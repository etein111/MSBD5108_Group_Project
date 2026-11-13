"""Block 3 — static figures for the calibration report.

Reliability diagram (one panel per model x decision type), raw vs recalibrated.
Two-series categorical palette: Okabe-Ito blue/vermillion, a published
colour-blind-safe pair; the reference diagonal is neutral gray.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from .calibration import (
    CalibrationReport,
    cross_fitted,
    effective_uniform,
    label_indices,
    probability_matrix,
    reliability_bins,
)

RAW_COLOR = "#0072B2"
CALIBRATED_COLOR = "#D55E00"
REFERENCE_COLOR = "#6B6B6B"
INK = "#1F2328"
MUTED = "#5A6169"


def _curve(probs: np.ndarray, labels: np.ndarray, n_bins: int) -> Dict[str, List[float]]:
    confidences = probs.max(axis=1)
    corrects = (probs.argmax(axis=1) == labels).astype(float)
    return reliability_bins(confidences, corrects, n_bins=n_bins)


def reliability_grid(
    reports: Sequence[CalibrationReport],
    records_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
    path: Path,
    n_bins: int = 15,
    seed: int = 0,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = min(2, len(reports))
    rows = math.ceil(len(reports) / columns)
    figure, axes = plt.subplots(
        rows, columns, figsize=(5.2 * columns, 4.6 * rows), dpi=200, squeeze=False
    )

    for position, report in enumerate(reports):
        axis = axes[position // columns][position % columns]
        records = [
            record
            for record in records_by_model[report.model]
            if record.get("type") == report.kind
        ]
        labels = label_indices(records, report.kind)
        calibrated, _, _ = cross_fitted(
            records, report.kind, report.method, report.n_folds, seed
        )
        raw_curve = _curve(probability_matrix(records, report.kind), labels, n_bins)
        calibrated_curve = _curve(calibrated, labels, n_bins)

        axis.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, color=REFERENCE_COLOR,
                  label="Perfect calibration", zorder=1)
        for curve, color, name, ece in (
            (raw_curve, RAW_COLOR, "Raw", report.raw["ece_15bin"]),
            (calibrated_curve, CALIBRATED_COLOR, "Recalibrated", report.calibrated["ece_15bin"]),
        ):
            xs = [x for x, size in zip(curve["confidence"], curve["size"]) if size]
            ys = [y for y, size in zip(curve["accuracy"], curve["size"]) if size]
            axis.plot(xs, ys, color=color, linewidth=2.0, marker="o", markersize=4.5,
                      markeredgecolor="white", markeredgewidth=0.8,
                      label=f"{name} (ECE {ece:.3f})", zorder=2)

        axis.set_xlim(-0.02, 1.02)
        axis.set_ylim(-0.02, 1.02)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("Mean predicted confidence", fontsize=10, color=MUTED)
        axis.set_ylabel("Empirical accuracy", fontsize=10, color=MUTED)
        axis.set_title(
            f"{report.model.upper()} · {report.kind}  (n={report.raw['n']})",
            fontsize=11, color=INK, fontweight="bold", loc="left",
        )
        axis.tick_params(colors=MUTED, labelsize=9)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            axis.spines[spine].set_color("#C9CED6")
        axis.grid(True, linestyle=":", linewidth=0.6, color="#D8DCE2", zorder=0)
        axis.legend(fontsize=8.5, frameon=False, loc="upper left")
        uniform_after_scaling = (
            report.method in ("temperature", "ordinal")
            and effective_uniform(report.kind, report.calibrated)
        )
        if report.params.get("at_boundary") or uniform_after_scaling:
            # A railed or effectively-uniform fit means the recalibrated point is
            # an artefact, not evidence of a good model.
            note = (
                "fit at search bound\n→ degenerate calibration\n(no usable signal)"
                if report.params.get("at_boundary")
                else "calibrated distribution ≈ uniform\n(no usable signal)"
            )
            axis.text(
                0.03, 0.75, note,
                transform=axis.transAxes, ha="left", va="top",
                fontsize=7.5, color=MUTED, style="italic",
            )

    for position in range(len(reports), rows * columns):
        axes[position // columns][position % columns].set_visible(False)

    figure.suptitle(
        "Reliability diagram — raw vs cross-fitted recalibration",
        fontsize=12.5, color=INK, x=0.02, ha="left",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, facecolor="white")
    plt.close(figure)
    return path
