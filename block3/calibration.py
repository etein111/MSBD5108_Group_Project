"""Block 3 — post-hoc calibration pipeline (T9).

Fits a calibrator on ``held_out`` and reports **raw vs recalibrated** ECE and
Brier score for both models, using K-fold cross-fitting so every reported
probability is produced by a calibrator that never saw that sample.

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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "kind": self.kind,
            "method": self.method,
            "n_folds": self.n_folds,
            "n": self.raw["n"],
            "params": self.params,
            "raw": self.raw,
            "calibrated": self.calibrated,
            "delta": {
                key: self.calibrated[key] - self.raw[key]
                for key in ("ece_15bin", "brier", "nll")
            },
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
) -> CalibrationReport:
    method = method or DEFAULT_METHOD[kind]
    labels = label_indices(records, kind)
    raw = calibration_scores(probability_matrix(records, kind), labels, kind)
    calibrated_probs, params, fold_params = cross_fitted(records, kind, method, n_folds, seed)
    calibrated = calibration_scores(calibrated_probs, labels, kind)
    return CalibrationReport(
        model=model,
        kind=kind,
        method=method,
        n_folds=min(max(2, n_folds), len(records)),
        params=params,
        raw=raw,
        calibrated=calibrated,
        fold_params=fold_params,
    )


def run_calibration(
    records_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
    out_dir: Path,
    n_folds: int = 5,
    seed: int = 0,
    make_plot: bool = True,
) -> List[CalibrationReport]:
    reports: List[CalibrationReport] = []
    for model, records in records_by_model.items():
        for kind in ("noul", "score"):
            subset = [record for record in records if record.get("type") == kind]
            if len(subset) < 2:
                continue
            reports.append(evaluate(subset, kind, model, n_folds=n_folds, seed=seed))
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "n_folds": n_folds,
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


def _markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Calibration report (T9)",
        "",
        f"Cross-fitting: {payload['n_folds']}-fold, seed {payload['seed']}. "
        "Every reported probability comes from a calibrator fitted without that sample.",
        "",
        "| model | kind | method | n | ECE raw | ECE calibrated | Brier raw | Brier calibrated |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for report in payload["reports"]:
        lines.append(
            "| {model} | {kind} | {method} | {n} | {ece_raw:.4f} | {ece_cal:.4f} | "
            "{brier_raw:.4f} | {brier_cal:.4f} |".format(
                model=report["model"],
                kind=report["kind"],
                method=report["method"],
                n=report["n"],
                ece_raw=report["raw"]["ece_15bin"],
                ece_cal=report["calibrated"]["ece_15bin"],
                brier_raw=report["raw"]["brier"],
                brier_cal=report["calibrated"]["brier"],
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
    if railed:
        lines += ["", "## Warnings", ""] + railed
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
