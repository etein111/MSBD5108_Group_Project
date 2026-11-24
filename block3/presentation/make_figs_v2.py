"""Reliability grid for the four-task dataset (2 models x 4 tasks, held-out fit).

Phishing comes from the length-matched rebuild (evaluation/data_phishing, preds in
/tmp/phish); the other three tasks come from data_v2 (preds in /tmp/v2).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from block3.calibration import (
    calibration_scores, fit_on, label_indices, probability_matrix, reliability_bins,
)

BLUE, VERM, GRAY, INK, MUTED = "#0072B2", "#D55E00", "#6B6B6B", "#1F2328", "#5A6169"
TASKS = [
    ("sms_spam", "noul", "platt", "SMS spam"),
    ("phishing", "noul", "platt", "Phishing (length-matched)"),
    ("noul", "noul", "platt", "Toxicity (binary)"),
    ("score", "score", "ordinal", "Toxicity count"),
]
OUT = Path("block3/outputs/presentation")


def load(path: Path) -> list:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def merge(*paths: Path) -> dict:
    data = {}
    for path in paths:
        if Path(path).exists():
            for row in load(Path(path)):
                data[row["id"]] = row
    return data


def attach(records, data):
    out = []
    for record in records:
        # Drop the old CEAS phishing records (saturated); the phishing panel uses
        # only the length-matched rebuild from /tmp/phish.
        if record.get("id", "").startswith("noul_phishing_email_"):
            continue
        record["task"] = data.get(record["id"], {}).get("task") or record.get("type")
        if record.get("status") == "success" and record.get("prob"):
            out.append(record)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = merge(Path("evaluation/data_v2/eval.jsonl"), Path("/tmp/phish/eval.jsonl"))
    fit_data = merge(Path("evaluation/data_v2/held_out.jsonl"), Path("/tmp/phish/held_out.jsonl"))

    fig, axes = plt.subplots(2, 4, figsize=(20, 9.2), dpi=150, squeeze=False)
    for row, model in enumerate(("jev", "qwen")):
        preds = attach(
            load(Path(f"/tmp/v2/preds-v2-eval/{model}.jsonl"))
            + load(Path(f"/tmp/phish/preds-eval/{model}.jsonl")), data,
        )
        fits = attach(
            load(Path(f"/tmp/v2/preds-v2-fit/{model}.jsonl"))
            + load(Path(f"/tmp/phish/preds-fit/{model}.jsonl")), fit_data,
        )
        for col, (task, kind, method, name) in enumerate(TASKS):
            ax = axes[row][col]
            subset = [p for p in preds if p["task"] == task]
            fit_subset = [p for p in fits if p["task"] == task]
            labels = label_indices(subset, kind)
            raw = probability_matrix(subset, kind)
            calibrated, _ = fit_on(fit_subset, subset, kind, method)
            raw_s = calibration_scores(raw, labels, kind)
            cal_s = calibration_scores(calibrated, labels, kind)

            def curve(probs):
                conf = probs.max(axis=1)
                correct = (probs.argmax(axis=1) == labels).astype(float)
                return reliability_bins(conf, correct, n_bins=15)

            for probs, color, label, ece in (
                (raw, BLUE, "Raw", raw_s["ece_15bin"]),
                (calibrated, VERM, "Recalibrated", cal_s["ece_15bin"]),
            ):
                c = curve(probs)
                xs = [x for x, s in zip(c["confidence"], c["size"]) if s]
                ys = [y for y, s in zip(c["accuracy"], c["size"]) if s]
                ax.plot(xs, ys, color=color, lw=2, marker="o", ms=4,
                        markeredgecolor="white", markeredgewidth=0.6,
                        label=f"{label} (ECE {ece:.3f})", zorder=3)
            ax.plot([0, 1], [0, 1], "--", color=GRAY, lw=1.2, label="Perfect", zorder=1)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_aspect("equal")
            ax.set_xlabel("Mean predicted confidence", fontsize=8.5, color=MUTED)
            ax.set_ylabel("Empirical accuracy", fontsize=8.5, color=MUTED)
            ax.set_title(f"{model.upper()} · {name} (n={len(subset)})",
                         fontsize=9.5, color=INK, fontweight="bold", loc="left")
            ax.tick_params(colors=MUTED, labelsize=8)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            ax.grid(True, linestyle=":", linewidth=0.5, color="#D8DCE2", zorder=0)
            ax.legend(fontsize=7, frameon=False, loc="upper left")

    fig.suptitle("Reliability — raw vs held-out recalibration (2 models x 4 tasks)",
                 x=0.02, ha="left", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = OUT / "reliability_v2.png"
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    print("wrote", path)


if __name__ == "__main__":
    main()
