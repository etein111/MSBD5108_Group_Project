"""Per-task calibration report for the four-task dataset.

The cluster runner does not copy ``task`` into its predictions, so we join the
predictions back to the data split by ``id`` and then calibrate each task
separately (binary tasks with Platt, the score task with ordinal scaling).

    python -m block3.report_by_task \
        --pred-dir outputs/preds-v2-eval --data evaluation/data_v2/eval.jsonl \
        --fit-dir outputs/preds-v2-fit --fit-data evaluation/data_v2/held_out.jsonl \
        --out outputs/calibration-v2/by_task.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from .calibration import evaluate

# task -> (decision kind, calibration method)
TASKS = {
    "sms_spam": ("noul", "platt"),
    "phishing": ("noul", "platt"),
    "noul": ("noul", "platt"),
    "score": ("score", "ordinal"),
}


def load(path: Path) -> List[Dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def attach_task(records: List[Dict[str, Any]], data_by_id: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    for record in records:
        source = data_by_id.get(record["id"], {})
        record["task"] = source.get("task") or record.get("type")
        record.setdefault("source", source.get("source"))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--fit-dir", type=Path, default=None)
    parser.add_argument("--fit-data", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-boot", type=int, default=0)
    args = parser.parse_args()

    data_by_id = {row["id"]: row for row in load(args.data)}
    fit_by_id = {row["id"]: row for row in load(args.fit_data)} if args.fit_data else {}

    rows = []
    for model in ("jev", "qwen"):
        preds = attach_task(load(args.pred_dir / f"{model}.jsonl"), data_by_id)
        fits = attach_task(load(args.fit_dir / f"{model}.jsonl"), fit_by_id) if args.fit_dir else []
        for task, (kind, method) in TASKS.items():
            subset = [r for r in preds if r.get("task") == task]
            if len(subset) < 2:
                continue
            fit_subset = [r for r in fits if r.get("task") == task]
            report = evaluate(
                subset, kind, model, method=method,
                fit_records=fit_subset if len(fit_subset) >= 2 else None,
                n_boot=args.n_boot,
            )
            item = report.to_dict()
            item["task"] = task
            rows.append(item)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    print(f"{'model':5} {'task':9} {'n':>5} {'ACC':>6} {'ACC_cal':>7} {'ECE':>6} {'ECE_cal':>7} "
          f"{'Brier':>7} {'Brier_cal':>9} {'extra':>22}")
    for r in rows:
        raw, cal = r["raw"], r["calibrated"]
        extra = (f"QWK {raw.get('qwk','-')}->{cal.get('qwk','-')}" if r["kind"] == "score"
                 else f"AUC {raw.get('auroc','-')}->{cal.get('auroc','-')}")
        print(f"{r['model']:5} {r['task']:9} {r['n']:5d} "
              f"{raw['accuracy']:6.4f} {cal['accuracy']:7.4f} "
              f"{raw['ece_15bin']:6.4f} {cal['ece_15bin']:7.4f} "
              f"{raw['brier']:7.4f} {cal['brier']:9.4f} {extra}")


if __name__ == "__main__":
    main()
