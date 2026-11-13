"""Block 3 — post-hoc calibration pipeline (T9).

Two protocols are supported, and the report records which one produced it:

* **held_out fit** (PLAN §2.3): fit on ``held_out`` predictions, report on
  ``eval`` predictions — pass ``--fit-dir`` to the calibrate command.
* **K-fold cross-fitting** (default, no extra predictions needed): every
  reported probability is produced by a calibrator fitted without that sample.

Both report **raw vs recalibrated** ECE / Brier / NLL plus discrimination
metrics (AUROC, macro-F1, balanced accuracy for ``noul``; macro-F1, QWK, MAE
for ``score``) and 95% percentile bootstrap intervals.

Methods
-------
* ``noul`` (binary): **Platt scaling** on the log-odds of P(yes).
* ``score`` (ordinal 0-5): **temperature scaling** on the model's raw logits.

The fitted parameters are tiny (two scalars / one scalar) and are written to
``calibration_report.json`` so the deployment can apply them at serving time.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

LABELS: Dict[str, Tuple[str, ...]] = {
    "noul": ("no", "yes"),
    "score": tuple(str(level) for level in range(6)),
}
DEFAULT_METHOD = {"noul": "platt", "score": "temperature"}


# --------------------------------------------------------------------------- #
# calibration maps
# --------------------------------------------------------------------------- #
@dataclass
class PlattScaler:
    """``P_cal(yes) = sigmoid(a * logit(p_yes) + b)``."""

    a: float = 1.0
    b: float = 0.0

    def fit(self, p_yes: np.ndarray, y: np.ndarray, l2: float = 1e-3, iters: int = 100) -> "PlattScaler":
        p = np.clip(np.asarray(p_yes, dtype=float), 1e-6, 1.0 - 1e-6)
        target = np.asarray(y, dtype=float)
        if np.unique(target).size < 2:
            self.a, self.b = 1.0, 0.0
            return self
        x = np.log(p / (1.0 - p))
        a, b = 1.0, 0.0
        objective = _platt_objective(a, b, x, target, l2)
        for _ in range(iters):
            z = a * x + b
            q = 1.0 / (1.0 + np.exp(-z))
            weight = q * (1.0 - q)
            g_a = float(np.sum((q - target) * x) + l2 * a)
            g_b = float(np.sum(q - target))
            h_aa = float(np.sum(weight * x * x) + l2) + 1e-12
            h_ab = float(np.sum(weight * x))
            h_bb = float(np.sum(weight)) + 1e-12
            det = h_aa * h_bb - h_ab * h_ab
            if det <= 0:
                break
            step_a = (h_bb * g_a - h_ab * g_b) / det
            step_b = (-h_ab * g_a + h_aa * g_b) / det
            scale = 1.0
            for _ in range(20):
                candidate_a, candidate_b = a - scale * step_a, b - scale * step_b
                if _platt_objective(candidate_a, candidate_b, x, target, l2) <= objective:
                    break
                scale *= 0.5
            else:
                break
            a, b = candidate_a, candidate_b
            new_objective = _platt_objective(a, b, x, target, l2)
            if abs(new_objective - objective) < 1e-11:
                objective = new_objective
                break
            objective = new_objective
        self.a, self.b = float(a), float(b)
        return self

    def transform(self, p_yes: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p_yes, dtype=float), 1e-6, 1.0 - 1e-6)
        z = self.a * np.log(p / (1.0 - p)) + self.b
        return 1.0 / (1.0 + np.exp(-z))


def _platt_objective(a: float, b: float, x: np.ndarray, y: np.ndarray, l2: float) -> float:
    z = a * x + b
    loss = np.logaddexp(0.0, z) - y * z
    return float(np.mean(loss) + 0.5 * l2 * a * a)


#: Search bounds for the temperature. The upper bound is deliberately far past
#: any sensible optimum: a fit that rails against it means the logits carry no
#: usable signal and the loss is minimised by flattening towards uniform, which
#: is reported as ``at_boundary`` rather than silently clipped.
TEMPERATURE_BOUNDS = (0.05, 1.0e4)


@dataclass
class TemperatureScaler:
    """``softmax(logits / T)`` over the native option slots."""

    temperature: float = 1.0
    at_boundary: bool = False

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "TemperatureScaler":
        logits = np.asarray(logits, dtype=float)
        labels = np.asarray(labels, dtype=int)
        if logits.ndim != 2 or len(labels) == 0:
            raise ValueError("Expected a 2-D logit matrix and one label per row.")
        low, high = TEMPERATURE_BOUNDS
        grid = np.exp(np.linspace(math.log(low), math.log(high), 801))
        losses = [nll_with_temperature(logits, labels, temperature) for temperature in grid]
        best = int(np.argmin(losses))
        self.temperature = float(grid[best])
        self.at_boundary = best in (0, len(grid) - 1)
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        logits = np.asarray(logits, dtype=float) / self.temperature
        logits = logits - logits.max(axis=1, keepdims=True)
        shifted = np.exp(logits)
        return shifted / shifted.sum(axis=1, keepdims=True)


def nll_from_probs(probs: np.ndarray, labels: np.ndarray) -> float:
    """Categorical NLL of an explicit probability matrix."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    rows = np.arange(len(labels))
    return float(-np.mean(np.log(np.clip(probs[rows, labels], 1e-12, None))))


def nll_with_temperature(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    z = np.asarray(logits, dtype=float) / temperature
    z = z - z.max(axis=1, keepdims=True)
    log_norm = np.log(np.exp(z).sum(axis=1))
    return float(np.mean(log_norm - z[np.arange(len(labels)), labels]))


# --------------------------------------------------------------------------- #
# record helpers
# --------------------------------------------------------------------------- #
def probability_matrix(records: Sequence[Mapping[str, Any]], kind: str) -> np.ndarray:
    labels = LABELS[kind]
    rows = []
    for record in records:
        prob = record.get("prob") or {}
        row = [float(prob.get(label, 0.0)) for label in labels]
        total = math.fsum(row)
        if total <= 0.0:
            raise ValueError(f"record {record.get('id')!r} has no usable probabilities")
        rows.append([value / total for value in row])
    return np.asarray(rows, dtype=float)


def logit_matrix(records: Sequence[Mapping[str, Any]], kind: str) -> np.ndarray:
    labels = LABELS[kind]
    rows = []
    for record in records:
        logits = record.get("logits") or {}
        row = [float(logits[label]) for label in labels]
        rows.append(row)
    return np.asarray(rows, dtype=float)


def label_indices(records: Sequence[Mapping[str, Any]], kind: str) -> np.ndarray:
    labels = LABELS[kind]
    index = {label: i for i, label in enumerate(labels)}
    values = []
    for record in records:
        truth = record["ground_truth"]
        key = str(int(truth)) if kind == "score" else str(truth).strip().lower()
        if key not in index:
            raise ValueError(f"record {record.get('id')!r} has invalid label {truth!r}")
        values.append(index[key])
    return np.asarray(values, dtype=int)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def expected_calibration_error(confidences: np.ndarray, corrects: np.ndarray, n_bins: int = 15) -> float:
    """Top-label ECE with the same 15-bin scheme as ``evaluation.metrics``."""
    confidences = np.asarray(confidences, dtype=float)
    corrects = np.asarray(corrects, dtype=float)
    if confidences.size == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        low, high = edges[i], edges[i + 1]
        mask = (confidences >= low) & (
            confidences <= high if i == n_bins - 1 else confidences < high
        )
        size = int(mask.sum())
        if size:
            ece += (size / confidences.size) * abs(
                float(corrects[mask].mean()) - float(confidences[mask].mean())
            )
    return float(ece)


def reliability_bins(
    confidences: np.ndarray, corrects: np.ndarray, n_bins: int = 15
) -> Dict[str, List[float]]:
    confidences = np.asarray(confidences, dtype=float)
    corrects = np.asarray(corrects, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers, accuracies, confs, sizes = [], [], [], []
    for i in range(n_bins):
        low, high = edges[i], edges[i + 1]
        mask = (confidences >= low) & (
            confidences <= high if i == n_bins - 1 else confidences < high
        )
        centers.append(float((low + high) / 2.0))
        sizes.append(int(mask.sum()))
        accuracies.append(float(corrects[mask].mean()) if mask.any() else float("nan"))
        confs.append(float(confidences[mask].mean()) if mask.any() else float("nan"))
    return {"centers": centers, "accuracy": accuracies, "confidence": confs, "size": sizes}


def calibration_scores(probs: np.ndarray, labels: np.ndarray, kind: str) -> Dict[str, float]:
    probs = np.asarray(probs, dtype=float)
    predictions = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    corrects = (predictions == labels).astype(float)
    rows = np.arange(len(labels))
    if kind == "noul":
        p_yes = probs[:, 1]
        brier = float(np.mean((p_yes - labels.astype(float)) ** 2))
    else:
        one_hot = np.zeros_like(probs)
        one_hot[rows, labels] = 1.0
        brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
    return {
        "n": int(len(labels)),
        "accuracy": float(corrects.mean()),
        "ece_15bin": expected_calibration_error(confidences, corrects),
        "brier": brier,
        "nll": nll_from_probs(probs, labels),
        "mean_confidence": float(confidences.mean()),
    }


# --------------------------------------------------------------------------- #
# discrimination metrics (threshold-free where possible)
# --------------------------------------------------------------------------- #
def _confusion(predictions: np.ndarray, labels: np.ndarray, n_classes: int) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=float)
    np.add.at(matrix, (np.asarray(labels, dtype=int), np.asarray(predictions, dtype=int)), 1.0)
    return matrix


def macro_f1(predictions: np.ndarray, labels: np.ndarray, n_classes: int) -> float:
    """Unweighted mean of per-class F1, over classes present in the labels."""
    matrix = _confusion(predictions, labels, n_classes)
    tp = np.diag(matrix)
    fp = matrix.sum(axis=0) - tp
    fn = matrix.sum(axis=1) - tp
    denominator = 2.0 * tp + fp + fn
    per_class = np.divide(2.0 * tp, denominator, out=np.zeros_like(tp), where=denominator > 0)
    present = (tp + fn) > 0
    return float(per_class[present].mean()) if present.any() else float("nan")


def balanced_accuracy(predictions: np.ndarray, labels: np.ndarray, n_classes: int) -> float:
    matrix = _confusion(predictions, labels, n_classes)
    tp = np.diag(matrix)
    fn = matrix.sum(axis=1) - tp
    present = (tp + fn) > 0
    recalls = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=present)
    return float(recalls[present].mean()) if present.any() else float("nan")


def quadratic_weighted_kappa(predictions: np.ndarray, labels: np.ndarray, n_classes: int) -> float:
    observed = _confusion(predictions, labels, n_classes)
    total = observed.sum()
    if total == 0:
        return float("nan")
    row = observed.sum(axis=1)
    column = observed.sum(axis=0)
    expected = np.outer(row, column) / total
    positions = np.arange(n_classes)
    weights = (positions[:, None] - positions[None, :]) ** 2 / max((n_classes - 1) ** 2, 1)
    denominator = float((weights * expected).sum())
    return 1.0 - float((weights * observed).sum()) / denominator if denominator > 0 else float("nan")


def binary_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUC with average ranks for ties (same value as sklearn)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    combined = np.concatenate([positive, negative])
    order = np.argsort(combined, kind="mergesort")
    sorted_values = combined[order]
    ranks = np.empty(combined.size, dtype=float)
    start = 0
    while start < sorted_values.size:
        stop = start
        while stop + 1 < sorted_values.size and sorted_values[stop + 1] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop + 1]] = 0.5 * (start + stop) + 1.0
        start = stop + 1
    rank_sum = ranks[:positive.size].sum()
    return float(
        (rank_sum - positive.size * (positive.size + 1) / 2.0) / (positive.size * negative.size)
    )


def discrimination_scores(probs: np.ndarray, labels: np.ndarray, kind: str) -> Dict[str, float]:
    """Threshold-free / threshold-aware quality, independent of calibration."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    predictions = probs.argmax(axis=1)
    if kind == "noul":
        return {
            "auroc": binary_auroc(probs[:, 1], labels),
            "macro_f1": macro_f1(predictions, labels, 2),
            "balanced_accuracy": balanced_accuracy(predictions, labels, 2),
        }
    n_classes = probs.shape[1]
    return {
        "macro_f1": macro_f1(predictions, labels, n_classes),
        "qwk": quadratic_weighted_kappa(predictions, labels, n_classes),
        "mae": float(np.mean(np.abs(predictions - labels))),
    }


def effective_uniform(kind: str, scores: Mapping[str, float]) -> bool:
    """True when the recalibrated distribution is indistinguishable from uniform.

    Temperature scaling can rail just *inside* the search bound (e.g. T≈32 on a
    six-way head) while its Brier score still equals the uniform predictor's
    ``(K-1)/K``. Low ECE is then an artefact of ignoring the input, not evidence
    of good calibration, so the report must say so even without ``at_boundary``.
    """
    n_classes = len(LABELS[kind])
    return float(scores["brier"]) >= (n_classes - 1) / n_classes - 1e-2


CI_KEYS = ("ece_15bin", "brier", "auroc", "macro_f1")


def bootstrap_ci(
    probs: np.ndarray,
    labels: np.ndarray,
    kind: str,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Dict[str, List[float]]:
    """Percentile bootstrap CI for the reported metrics.

    Resamples the *fixed* out-of-fold probabilities, so the interval measures
    sampling variability of the metric on this eval set; it does not include
    uncertainty from re-fitting the calibrator.
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if n_boot <= 0 or len(labels) < 2:
        return {}
    rng = np.random.default_rng(seed)
    collected: Dict[str, List[float]] = {key: [] for key in CI_KEYS}
    for _ in range(n_boot):
        indices = rng.integers(0, len(labels), size=len(labels))
        sample_labels = labels[indices]
        if np.unique(sample_labels).size < 2:
            continue  # AUC/macro-F1 undefined on a single-class resample
        sample = {
            **calibration_scores(probs[indices], sample_labels, kind),
            **discrimination_scores(probs[indices], sample_labels, kind),
        }
        for key in CI_KEYS:
            value = sample.get(key)
            if value is not None and np.isfinite(value):
                collected[key].append(float(value))
    return {
        key: [float(np.percentile(values, 100 * alpha / 2)),
              float(np.percentile(values, 100 * (1 - alpha / 2)))]
        for key, values in collected.items()
        if len(values) > 1
    }


# --------------------------------------------------------------------------- #
# calibration fitting protocols
# --------------------------------------------------------------------------- #
def fit_on(
    fit_records: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    kind: str,
    method: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """PLAN §2.3 protocol: fit on ``held_out``, report on ``eval``."""
    method = method or DEFAULT_METHOD[kind]
    fit_labels = label_indices(fit_records, kind)
    fit_probs = probability_matrix(fit_records, kind)
    probs = probability_matrix(records, kind)
    calibrated = np.empty_like(probs)
    if method == "platt":
        scaler = PlattScaler().fit(fit_probs[:, 1], fit_labels)
        p_yes = scaler.transform(probs[:, 1])
        calibrated[:, 1] = p_yes
        calibrated[:, 0] = 1.0 - p_yes
        params: Dict[str, float] = {"a": scaler.a, "b": scaler.b}
    elif method == "temperature":
        scaler = TemperatureScaler().fit(logit_matrix(fit_records, kind), fit_labels)
        calibrated = scaler.transform(logit_matrix(records, kind))
        params = {"temperature": scaler.temperature, "at_boundary": scaler.at_boundary}
    else:
        raise ValueError(f"unknown calibration method: {method!r}")
    return calibrated, params


# --------------------------------------------------------------------------- #
# cross-fitted calibration
# --------------------------------------------------------------------------- #
def _stratified_folds(labels: np.ndarray, n_folds: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    fold_of = np.empty(len(labels), dtype=int)
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        fold_of[indices] = np.arange(len(indices)) % n_folds
    return [np.flatnonzero(fold_of == fold) for fold in range(n_folds)]


@dataclass
class CalibrationReport:
    model: str
    kind: str
    method: str
    n_folds: int
    params: Dict[str, float]
    raw: Dict[str, float]
    calibrated: Dict[str, float]
    fold_params: List[Dict[str, float]] = field(default_factory=list)
    protocol: str = "cross_fitted"
    ci95: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "kind": self.kind,
            "method": self.method,
            "protocol": self.protocol,
            "n_folds": self.n_folds,
            "n": self.raw["n"],
            "params": self.params,
            "raw": self.raw,
            "calibrated": self.calibrated,
            "delta": {
                key: self.calibrated[key] - self.raw[key]
                for key in ("ece_15bin", "brier", "nll")
            },
            "ci95": self.ci95,
            "fold_params": self.fold_params,
        }


def cross_fitted(
    records: Sequence[Mapping[str, Any]],
    kind: str,
    method: Optional[str] = None,
    n_folds: int = 5,
    seed: int = 0,
) -> Tuple[np.ndarray, Dict[str, float], List[Dict[str, float]]]:
    """Return out-of-fold calibrated probabilities, pooled params, per-fold params."""
    method = method or DEFAULT_METHOD[kind]
    labels = label_indices(records, kind)
    probs = probability_matrix(records, kind)
    n_folds = max(2, min(n_folds, len(records)))
    folds = _stratified_folds(labels, n_folds, seed)
    calibrated = np.empty_like(probs)
    fold_params: List[Dict[str, float]] = []

    for fold, held_out in enumerate(folds):
        train = np.setdiff1d(np.arange(len(records)), held_out)
        if method == "platt":
            scaler = PlattScaler().fit(probs[train, 1], labels[train])
            p_yes = scaler.transform(probs[held_out, 1])
            calibrated[held_out, 1] = p_yes
            calibrated[held_out, 0] = 1.0 - p_yes
            fold_params.append({"fold": fold, "a": scaler.a, "b": scaler.b})
        elif method == "temperature":
            logits = logit_matrix(records, kind)
            scaler = TemperatureScaler().fit(logits[train], labels[train])
            calibrated[held_out] = scaler.transform(logits[held_out])
            fold_params.append(
                {
                    "fold": fold,
                    "temperature": scaler.temperature,
                    "at_boundary": scaler.at_boundary,
                }
            )
        else:
            raise ValueError(f"unknown calibration method: {method!r}")

    # Pooled parameters: fit once on everything, for deployment/serving.
    if method == "platt":
        pooled = PlattScaler().fit(probs[:, 1], labels)
        params = {"a": pooled.a, "b": pooled.b}
    else:
        pooled = TemperatureScaler().fit(logit_matrix(records, kind), labels)
        params = {"temperature": pooled.temperature, "at_boundary": pooled.at_boundary}
    return calibrated, params, fold_params


def evaluate(
    records: Sequence[Mapping[str, Any]],
    kind: str,
    model: str,
    method: Optional[str] = None,
    n_folds: int = 5,
    seed: int = 0,
    fit_records: Optional[Sequence[Mapping[str, Any]]] = None,
    n_boot: int = 1000,
) -> CalibrationReport:
    """Build one model x kind report.

    With ``fit_records`` the calibrator is fitted on that split only and applied
    to ``records`` (PLAN §2.3: fit on held_out, report on eval); without it the
    probabilities are K-fold cross-fitted on ``records`` itself.
    """
    method = method or DEFAULT_METHOD[kind]
    labels = label_indices(records, kind)
    raw_probs = probability_matrix(records, kind)
    raw = {**calibration_scores(raw_probs, labels, kind),
           **discrimination_scores(raw_probs, labels, kind)}
    if fit_records is not None:
        calibrated_probs, params = fit_on(fit_records, records, kind, method)
        fold_params: List[Dict[str, float]] = []
        protocol = "held_out_fit"
        folds = 0
    else:
        calibrated_probs, params, fold_params = cross_fitted(records, kind, method, n_folds, seed)
        protocol = "cross_fitted"
        folds = min(max(2, n_folds), len(records))
    calibrated = {**calibration_scores(calibrated_probs, labels, kind),
                  **discrimination_scores(calibrated_probs, labels, kind)}
    ci95 = {
        "raw": bootstrap_ci(raw_probs, labels, kind, n_boot=n_boot, seed=seed),
        "calibrated": bootstrap_ci(calibrated_probs, labels, kind, n_boot=n_boot, seed=seed),
    }
    return CalibrationReport(
        model=model,
        kind=kind,
        method=method,
        n_folds=folds,
        params=params,
        raw=raw,
        calibrated=calibrated,
        fold_params=fold_params,
        protocol=protocol,
        ci95=ci95,
    )


def run_calibration(
    records_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
    out_dir: Path,
    n_folds: int = 5,
    seed: int = 0,
    make_plot: bool = True,
    fit_records_by_model: Optional[Mapping[str, Sequence[Mapping[str, Any]]]] = None,
    n_boot: int = 1000,
) -> List[CalibrationReport]:
    reports: List[CalibrationReport] = []
    for model, records in records_by_model.items():
        fit_records = list((fit_records_by_model or {}).get(model, []))
        for kind in ("noul", "score"):
            subset = [record for record in records if record.get("type") == kind]
            if len(subset) < 2:
                continue
            fit_subset = [record for record in fit_records if record.get("type") == kind]
            reports.append(
                evaluate(
                    subset,
                    kind,
                    model,
                    n_folds=n_folds,
                    seed=seed,
                    fit_records=fit_subset if len(fit_subset) >= 2 else None,
                    n_boot=n_boot,
                )
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "n_folds": n_folds,
        "bootstrap": n_boot,
        "protocol": "held_out_fit" if fit_records_by_model else "cross_fitted",
        "reports": [report.to_dict() for report in reports],
    }
    (out_dir / "calibration_report.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "calibration_report.md").write_text(_markdown(payload), encoding="utf-8")
    if make_plot and reports:
        from .plots import reliability_grid

        reliability_grid(reports, records_by_model, out_dir / "reliability.png", seed=seed)
    return reports


def _with_ci(value: float, interval: Optional[Sequence[float]]) -> str:
    if not interval:
        return f"{value:.4f}"
    return f"{value:.4f} [{interval[0]:.4f}, {interval[1]:.4f}]"


def _markdown(payload: Mapping[str, Any]) -> str:
    protocol = payload.get("protocol", "cross_fitted")
    if protocol == "held_out_fit":
        protocol_note = "Fit on `held_out`, reported on `eval` (PLAN §2.3)."
    else:
        protocol_note = (
            f"Cross-fitting: {payload['n_folds']}-fold, seed {payload['seed']}. "
            "Every reported probability comes from a calibrator fitted without that sample."
        )
    bootstrap = payload.get("bootstrap", 0)
    ci_note = (
        f" Brackets are percentile 95% bootstrap intervals ({bootstrap} resamples, seed {payload['seed']})."
        if bootstrap else ""
    )
    lines = [
        "# Calibration report (T9)",
        "",
        protocol_note + ci_note,
        "",
        "| model | kind | method | n | ECE raw | ECE calibrated | Brier raw | Brier calibrated |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for report in payload["reports"]:
        raw_ci = report.get("ci95", {}).get("raw", {})
        cal_ci = report.get("ci95", {}).get("calibrated", {})
        lines.append(
            "| {model} | {kind} | {method} | {n} | {ece_raw} | {ece_cal} | "
            "{brier_raw} | {brier_cal} |".format(
                model=report["model"],
                kind=report["kind"],
                method=report["method"],
                n=report["n"],
                ece_raw=_with_ci(report["raw"]["ece_15bin"], raw_ci.get("ece_15bin")),
                ece_cal=_with_ci(report["calibrated"]["ece_15bin"], cal_ci.get("ece_15bin")),
                brier_raw=_with_ci(report["raw"]["brier"], raw_ci.get("brier")),
                brier_cal=_with_ci(report["calibrated"]["brier"], cal_ci.get("brier")),
            )
        )
    lines += [
        "",
        "Discrimination (does not improve under calibration by construction):",
        "",
        "| model | kind | AUC raw | AUC calibrated | macro-F1 raw | macro-F1 calibrated | QWK raw | QWK calibrated | MAE raw | MAE calibrated |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for report in payload["reports"]:
        def cell(key: str, state: str) -> str:
            value = report[state].get(key)
            return f"{value:.4f}" if value is not None and np.isfinite(value) else "—"
        lines.append(
            "| {model} | {kind} | {auroc_raw} | {auroc_cal} | {f1_raw} | {f1_cal} | "
            "{qwk_raw} | {qwk_cal} | {mae_raw} | {mae_cal} |".format(
                model=report["model"],
                kind=report["kind"],
                auroc_raw=cell("auroc", "raw"),
                auroc_cal=cell("auroc", "calibrated"),
                f1_raw=cell("macro_f1", "raw"),
                f1_cal=cell("macro_f1", "calibrated"),
                qwk_raw=cell("qwk", "raw"),
                qwk_cal=cell("qwk", "calibrated"),
                mae_raw=cell("mae", "raw"),
                mae_cal=cell("mae", "calibrated"),
            )
        )
    railed = [
        f"- **{r['model']}/{r['kind']}**: temperature {r['params']['temperature']:.3g} sits on the "
        f"search bound {TEMPERATURE_BOUNDS[0]:g}-{TEMPERATURE_BOUNDS[1]:g}. The NLL is still "
        "decreasing at the edge, so the best fit is the uniform distribution: these logits carry "
        "**no usable signal** for these labels, and the low ECE is not evidence of a good model."
        for r in payload["reports"]
        if r["params"].get("at_boundary")
    ]
    effectively_uniform = [
        f"- **{r['model']}/{r['kind']}**: temperature {r['params'].get('temperature', float('nan')):.3g} "
        f"stays inside the bound, but the calibrated Brier {r['calibrated']['brier']:.4f} is within "
        f"0.01 of the uniform predictor's {((len(LABELS[r['kind']]) - 1) / len(LABELS[r['kind']])):.4f}. "
        "The recalibrated distribution is effectively uniform, so the low ECE is an artefact and the "
        "decision metrics are unchanged by construction."
        for r in payload["reports"]
        if r["method"] == "temperature"
        and not r["params"].get("at_boundary")
        and effective_uniform(r["kind"], r["calibrated"])
    ]
    if railed or effectively_uniform:
        lines += ["", "## Warnings", ""] + railed + effectively_uniform
    lines += [
        "",
        "Fitted parameters (pooled, for serving):",
        "",
        "```json",
        json.dumps({f"{r['model']}/{r['kind']}": r["params"] for r in payload["reports"]}, indent=2),
        "```",
        "",
    ]
    return "\n".join(lines)
