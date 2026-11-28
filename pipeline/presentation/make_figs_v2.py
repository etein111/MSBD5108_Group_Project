"""Reliability figures for the four-task dataset, split into two readable 2x2 grids.

Phishing comes from the length-matched rebuild (preds in /tmp/phish); the other
three tasks come from data_v2 (preds in /tmp/v2).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline.calibration import (
    calibration_scores, fit_on, label_indices, probability_matrix, reliability_bins,
)

BLUE, VERM, GRAY, INK, MUTED = "#0072B2", "#D55E00", "#6B6B6B", "#1F2328", "#5A6169"
GROUPS = {
    "email": [("sms_spam", "noul", "platt", "SMS spam"),
              ("phishing", "noul", "platt", "Phishing (length-matched)")],
    "toxicity": [("noul", "noul", "platt", "Toxicity (binary)"),
                 ("score", "score", "ordinal", "Toxicity count")],
}
OUT = Path("pipeline/outputs/presentation")


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
        if record.get("id", "").startswith("noul_phishing_email_"):
            continue
        record["task"] = data.get(record["id"], {}).get("task") or record.get("type")
        if record.get("status") == "success" and record.get("prob"):
            out.append(record)
    return out


def draw(task, kind, method, name, ax, preds, fits):
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
        ax.plot(xs, ys, color=color, lw=2.6, marker="o", ms=6,
                markeredgecolor="white", markeredgewidth=0.7,
                label=f"{label} (ECE {ece:.3f})", zorder=3)
    ax.plot([0, 1], [0, 1], "--", color=GRAY, lw=1.4, label="Perfect", zorder=1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xlabel("Mean predicted confidence", fontsize=14, color=MUTED)
    ax.set_ylabel("Empirical accuracy", fontsize=14, color=MUTED)
    ax.set_title(f"{name} (n={len(subset)})", fontsize=16, color=INK,
                 fontweight="bold", loc="left")
    ax.tick_params(colors=MUTED, labelsize=13)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(True, linestyle=":", linewidth=0.6, color="#D8DCE2", zorder=0)
    ax.legend(fontsize=12, frameon=False, loc="upper left")

    # A fit that predicts one class for every input collapses the reliability
    # curve to a single point; say so, rather than let it read as good calibration.
    cal_argmax = calibrated.argmax(axis=1)
    if len(set(cal_argmax.tolist())) == 1:
        ax.text(0.28, 0.07, f"degenerate fit: all predictions\n= class {int(cal_argmax[0])} (no ordinal signal)",
                transform=ax.transAxes, fontsize=10.5, color=MUTED, style="italic",
                ha="left", va="bottom")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = merge(Path("evaluation/data_v2/eval.jsonl"), Path("/tmp/phish/eval.jsonl"))
    fit_data = merge(Path("evaluation/data_v2/held_out.jsonl"), Path("/tmp/phish/held_out.jsonl"))
    preds = {m: attach(load(Path(f"/tmp/v2/preds-v2-eval/{m}.jsonl"))
                       + load(Path(f"/tmp/phish/preds-eval/{m}.jsonl")), data)
             for m in ("jev", "qwen")}
    fits = {m: attach(load(Path(f"/tmp/v2/preds-v2-fit/{m}.jsonl"))
                      + load(Path(f"/tmp/phish/preds-fit/{m}.jsonl")), fit_data)
            for m in ("jev", "qwen")}

    for group, tasks in GROUPS.items():
        fig, axes = plt.subplots(2, 2, figsize=(13, 11.2), dpi=170, squeeze=False)
        for row, model in enumerate(("jev", "qwen")):
            for col, (task, kind, method, name) in enumerate(tasks):
                ax = axes[row][col]
                draw(task, kind, method, f"{model.upper()} · {name}", ax, preds[model], fits[model])
        fig.suptitle(f"Reliability — raw vs held-out recalibration ({group} tasks)",
                     x=0.02, ha="left", fontsize=17, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        path = OUT / f"reliability_{group}.png"
        fig.savefig(path, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        print("wrote", path)


if __name__ == "__main__":
    main()
