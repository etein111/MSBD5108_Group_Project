"""Run a reproducible CPU pilot against the checked-out Block 3 code."""

import argparse
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-source", type=int, default=10)
    args = parser.parse_args()
    if args.per_source < 2:
        parser.error("--per-source must be at least 2")

    repo = args.repo.resolve()
    sys.path.insert(0, str(repo))
    from block3.adapters import MockAdapter, load_demonstrations
    from block3.runner import load_records, run_adapter, write_jsonl
    from evaluation.metrics import JevBenchMetrics

    data = repo / "evaluation" / "data"
    files = {split: data / f"{split}.jsonl" for split in ("train", "held_out", "eval")}
    splits = {split: load_records(path) for split, path in files.items()}
    demonstrations = load_demonstrations(files["train"], per_type=4)
    rng = random.Random(args.seed)
    selected = {}
    coverage = {}
    for split, rows in splits.items():
        groups = defaultdict(list)
        for row in rows:
            groups[(row["type"], row["source"])].append(row)
        coverage[split] = {f"{kind}/{source}": len(group) for (kind, source), group in sorted(groups.items())}
        if split != "train":
            selected[split] = [
                row for _, group in sorted(groups.items())
                for row in rng.sample(group, min(args.per_source, len(group)))
            ]

    out = args.out.resolve()
    summary = {
        "mode": "mock",
        "commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        "seed": args.seed,
        "requested_per_source": args.per_source,
        "few_shot_count": len(demonstrations),
        "few_shot_sha256": hashlib.sha256(
            json.dumps(demonstrations, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "dataset_sha256": {split: hashlib.sha256(path.read_bytes()).hexdigest() for split, path in files.items()},
        "coverage": coverage,
        "runs": {},
    }
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    for split, sample in selected.items():
        sources = {row["id"]: row["source"] for row in sample}
        if len(sources) != len(sample):
            raise ValueError(f"Duplicate sample ID in {split}")
        for model in ("mock-a", "mock-b"):
            results = run_adapter(sample, MockAdapter(model, args.seed), demonstrations=demonstrations)
            for row in results:
                row.update(split=split, source=sources[row["id"]], probe="clean", seed=args.seed)
                row.pop("text")
            write_jsonl(out / split / f"{model}.jsonl", results)
            statuses = Counter(row["status"] for row in results)
            summary["runs"][f"{split}/{model}"] = dict(statuses)
            if statuses["success"] != len(sample):
                raise RuntimeError(f"{split}/{model}: {statuses}")
            metric_rows = [
                {**row, "probability": row["prob"], "latency_seconds": row["latency_ms"] / 1000}
                for row in results
            ]
            metrics = JevBenchMetrics(metric_rows).compute_all_metrics()
            if set(metrics) != {"noul", "score"}:
                raise RuntimeError(f"{split}/{model} does not cover both decision types")
    (out / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["runs"], indent=2))


if __name__ == "__main__":
    main()
