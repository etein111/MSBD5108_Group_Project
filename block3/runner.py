"""Block 3 — prediction runner and calibration CLI.

Examples
--------
Start the service first (``python -m deployment.server`` on a CUDA machine), then::

    python -m block3.runner predict --model jev  --out block3/outputs/preds/jev.jsonl
    python -m block3.runner predict --model qwen --out block3/outputs/preds/qwen.jsonl
    python -m block3.runner calibrate --pred-dir block3/outputs/preds --out-dir block3/outputs/calibration

Development without a GPU: ``--model mock`` exercises the whole pipeline offline.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .adapters import JevAdapter, MockAdapter, Prediction, QwenAdapter, load_demonstrations, prompt_for_record
from .probes import summarize_option_probes


def load_records(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_adapter(
    records: Sequence[Mapping[str, Any]],
    adapter: Any,
    limit: Optional[int] = None,
    demonstrations: Sequence[Mapping[str, Any]] = (),
) -> List[Dict[str, Any]]:
    """Run one adapter over a JevBench JSONL dataset and return rich records."""
    if not hasattr(adapter, "predict_full"):
        raise TypeError(
            "Adapter must expose predict_full(...) — use JevAdapter, QwenAdapter or MockAdapter."
        )
    selected = records[:limit] if limit else records
    results: List[Dict[str, Any]] = []
    for item in selected:
        record: Dict[str, Any] = {
            "id": item["id"],
            "type": item["type"],
            "text": item["text"],
            "ground_truth": item["label"],
            "model": getattr(adapter, "model", "unknown"),
        }
        for key in ("probe", "pair_id", "variant", "direction", "origin_id", "source_type"):
            if key in item:
                record[key] = item[key]
        try:
            prediction: Prediction = adapter.predict_full(
                prompt_for_record(item, demonstrations), item["text"], item["type"], item.get("options")
            )
            record.update(
                prediction=prediction.decision,
                prob=prediction.prob,
                logits=prediction.logits,
                latency_ms=prediction.latency_ms,
                status="success",
            )
        except Exception as error:  # recorded, not raised: one bad input must not stop a run
            record.update(
                prediction=None,
                prob=None,
                logits=None,
                latency_ms=0.0,
                status=f"error: {error}",
            )
        results.append(record)
    return results


def summarize(results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    ok = [record for record in results if record["status"] == "success"]
    latencies = [record["latency_ms"] for record in ok]
    by_type: Dict[str, int] = {}
    for record in ok:
        by_type[record["type"]] = by_type.get(record["type"], 0) + 1
    summary = {
        "n": len(results),
        "ok": len(ok),
        "errors": len(results) - len(ok),
        "by_type": by_type,
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 2) if latencies else None,
            "p50": round(statistics.median(latencies), 2) if latencies else None,
            "max": round(max(latencies), 2) if latencies else None,
        },
    }
    if option_probes := summarize_option_probes(results):
        summary["option_probes"] = option_probes
    return summary


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _adapter_for(name: str, base_url: str) -> Any:
    if name == "jev":
        return JevAdapter(base_url)
    if name == "qwen":
        return QwenAdapter(base_url)
    if name == "mock":
        return MockAdapter()
    raise ValueError(f"unknown model {name!r}")


def _predict(args: argparse.Namespace) -> None:
    demonstrations = load_demonstrations(args.shots_file, args.shots_per_type) if args.shots_file else ()
    adapter = _adapter_for(args.model, args.base_url)
    records = load_records(args.input)
    results = run_adapter(records, adapter, args.limit, demonstrations)
    write_jsonl(args.out, results)
    print(json.dumps({"out": str(args.out), **summarize(results)}, indent=2))


def _predictions_by_model(directory: Path) -> Dict[str, List[Dict[str, Any]]]:
    files = sorted(Path(directory).glob("*.jsonl"))
    if not files:
        raise SystemExit(f"no prediction files in {directory}")
    records_by_model: Dict[str, List[Dict[str, Any]]] = {}
    for path in files:
        records = [r for r in load_records(path) if r.get("status") == "success"]
        if not records:
            continue
        model = records[0].get("model") or path.stem
        records_by_model.setdefault(model, []).extend(records)
    return records_by_model


def _calibrate(args: argparse.Namespace) -> None:
    from .calibration import run_calibration

    records_by_model = _predictions_by_model(Path(args.pred_dir))
    fit_records_by_model = (
        _predictions_by_model(Path(args.fit_dir)) if args.fit_dir else None
    )
    reports = run_calibration(
        records_by_model,
        Path(args.out_dir),
        n_folds=args.folds,
        seed=args.seed,
        make_plot=not args.no_plot,
        fit_records_by_model=fit_records_by_model,
        n_boot=args.bootstrap,
    )
    summary = [
        {
            "model": report.model,
            "kind": report.kind,
            "method": report.method,
            "protocol": report.protocol,
            "ece_raw": round(report.raw["ece_15bin"], 4),
            "ece_calibrated": round(report.calibrated["ece_15bin"], 4),
            "brier_raw": round(report.raw["brier"], 4),
            "brier_calibrated": round(report.calibrated["brier"], 4),
            **{
                key: round(float(value), 4)
                for key, value in report.raw.items()
                if key in ("auroc", "macro_f1", "qwk", "mae") and value == value
            },
        }
        for report in reports
    ]
    print(json.dumps(summary, indent=2))


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Block 3 prediction / calibration runner.")
    sub = parser.add_subparsers(dest="command", required=True)

    predict = sub.add_parser("predict", help="call a model adapter over a JSONL dataset")
    predict.add_argument("--model", choices=("jev", "qwen", "mock"), required=True)
    predict.add_argument("--input", type=Path, default=Path("evaluation/data/eval.jsonl"))
    predict.add_argument("--out", type=Path, required=True)
    predict.add_argument("--base-url", default="http://127.0.0.1:8000")
    predict.add_argument("--limit", type=int, default=None)
    predict.add_argument("--shots-file", type=Path, default=None,
                         help="train split used for shared few-shot demonstrations")
    predict.add_argument("--shots-per-type", type=int, default=4)
    predict.set_defaults(handler=_predict)

    calibrate = sub.add_parser("calibrate", help="cross-fitted raw-vs-recalibrated report")
    calibrate.add_argument("--pred-dir", type=Path, default=Path("block3/outputs/preds"))
    calibrate.add_argument("--out-dir", type=Path, default=Path("block3/outputs/calibration"))
    calibrate.add_argument("--folds", type=int, default=5)
    calibrate.add_argument("--seed", type=int, default=0)
    calibrate.add_argument("--no-plot", action="store_true")
    calibrate.add_argument("--fit-dir", type=Path, default=None,
                           help="prediction directory used to fit the calibrator (e.g. held_out); "
                                "default is K-fold cross-fitting inside --pred-dir")
    calibrate.add_argument("--bootstrap", type=int, default=1000,
                           help="bootstrap resamples for the 95%% CI (0 disables)")
    calibrate.set_defaults(handler=_calibrate)

    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
