"""Batch inference on a GPU node (SuperPOD / any CUDA box).

Loads ``deployment.runtime.Runtime`` once and runs JEV and the frozen Qwen
backbone over a JevBench JSONL dataset in-process — no HTTP server, no port.

Output records use the same schema ``block3.runner`` writes, so
``python -m block3 calibrate --pred-dir <out-dir>`` consumes them directly.

The run is resumable: ids already present in an output file are skipped, and
every record is flushed immediately, so a wall-clock timeout keeps what it
produced.

Example (inside an sbatch job)::

    python -m block3.cluster_run \
        --input evaluation/data/eval.jsonl \
        --out-dir /scratch/.../outputs/preds \
        --cache-dir /scratch/.../hf \
        --shots-file evaluation/data/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .adapters import augment_with_shots, load_demonstrations


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def existing_ids(path: Path) -> set:
    if not path.exists():
        return set()
    return {record.get("id") for record in load_jsonl(path)}


def select(records: Sequence[Mapping[str, Any]], limit_per_type: Optional[int]) -> List[Mapping[str, Any]]:
    if not limit_per_type:
        return list(records)
    chosen: List[Mapping[str, Any]] = []
    seen: Dict[str, int] = {}
    for record in records:
        kind = record["type"]
        if seen.get(kind, 0) < limit_per_type:
            chosen.append(record)
            seen[kind] = seen.get(kind, 0) + 1
    return chosen


def percentile(values: Sequence[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[position]


def run_model(runtime, model: str, records, out_path: Path, demonstrations) -> Dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = existing_ids(out_path)
    pending = [record for record in records if record["id"] not in done]
    print(f"[{model}] {len(done)} already done, {len(pending)} pending", flush=True)

    latencies: List[float] = []
    failures = 0
    started = time.time()
    with out_path.open("a", encoding="utf-8") as handle:
        for index, item in enumerate(pending, start=1):
            prompt = item.get("prompt", "")
            if model == "qwen":
                prompt = augment_with_shots(prompt, item["type"], demonstrations)
            record: Dict[str, Any] = {
                "id": item["id"],
                "type": item["type"],
                "text": item["text"],
                "ground_truth": item["label"],
                "model": model,
            }
            for key in ("probe", "pair_id", "variant", "direction", "origin_id"):
                if key in item:
                    record[key] = item[key]
            try:
                prediction, _token_ids = runtime.predict(
                    model, item["type"], item["text"], prompt
                )
                record.update(
                    prediction=prediction.decision,
                    prob=prediction.prob,
                    logits=prediction.logits,
                    latency_ms=prediction.latency,
                    status="success",
                )
                latencies.append(prediction.latency)
            except Exception as error:  # one bad input must not kill the batch
                failures += 1
                record.update(
                    prediction=None, prob=None, logits=None, latency_ms=0.0,
                    status=f"error: {type(error).__name__}: {error}",
                )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 100 == 0 or index == len(pending):
                elapsed = time.time() - started
                rate = index / elapsed if elapsed else 0.0
                eta = (len(pending) - index) / rate if rate else 0.0
                print(
                    f"[{model}] {index}/{len(pending)}  {rate:.2f} rec/s  "
                    f"elapsed {elapsed/60:.1f} min  eta {eta/60:.1f} min  "
                    f"failures {failures}",
                    flush=True,
                )
    summary = {
        "model": model,
        "records": len(records),
        "ran": len(pending),
        "failures": failures,
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 2) if latencies else None,
            "p50": round(statistics.median(latencies), 2) if latencies else None,
            "p95": round(percentile(latencies, 0.95), 2) if latencies else None,
        },
        "wall_minutes": round((time.time() - started) / 60, 2),
        "out": str(out_path),
    }
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Batch-infer JEV and Qwen on a GPU node.")
    parser.add_argument("--input", type=Path, default=Path("evaluation/data/eval.jsonl"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["jev", "qwen"], choices=["jev", "qwen"])
    parser.add_argument("--limit-per-type", type=int, default=None)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--shots-file", type=Path, default=None)
    parser.add_argument("--shots-per-type", type=int, default=4)
    args = parser.parse_args(argv)

    from deployment.runtime import Runtime

    records = select(load_jsonl(args.input), args.limit_per_type)
    demonstrations = (
        load_demonstrations(args.shots_file, args.shots_per_type) if args.shots_file else []
    )
    print(f"dataset: {len(records)} records; few-shot demos: {len(demonstrations)}", flush=True)

    runtime = Runtime(args.cache_dir, args.max_input_tokens)
    summaries = [
        run_model(runtime, model, records, args.out_dir / f"{model}.jsonl", demonstrations)
        for model in args.models
    ]
    (args.out_dir / "run_summary.json").write_text(
        json.dumps({"records": len(records), "runs": summaries}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
