"""Inspect a HuggingFace email dataset's schema, label balance, sources, and lengths."""

from __future__ import annotations

import collections
import statistics
import sys

from datasets import load_dataset


def main() -> None:
    name = sys.argv[1]
    ds = load_dataset(name)
    print("=== dataset:", name, "===")
    for split in ds:
        data = ds[split]
        print(f"\n--- split {split}: {len(data)} rows ---")
        print("features:", data.column_names)
        row0 = data[0]
        for col, val in row0.items():
            s = str(val).replace("\n", " ")[:120]
            print(f"  {col} ({type(val).__name__}): {s}")

        # guess columns
        cols = data.column_names
        text_col = next((c for c in ("Email Text", "text", "email", "body", "content") if c in cols), None)
        label_col = next((c for c in ("Email Type", "label", "labels", "type", "spam") if c in cols), None)
        source_col = next((c for c in ("source", "corpus", "origin", "dataset") if c in cols), None)
        print("  guessed text/label/source:", text_col, label_col, source_col)
        if label_col:
            print("  label counts:", dict(collections.Counter(str(v) for v in data[label_col])))
        if source_col:
            print("  source counts:", dict(collections.Counter(str(v) for v in data[source_col]).most_common(20)))
        if text_col and label_col:
            by = collections.defaultdict(list)
            for t, l in zip(data[text_col], data[label_col]):
                by[str(l)].append(len(str(t)))
            for lab, lens in sorted(by.items()):
                print(f"  len[{lab}]: n={len(lens)} median={statistics.median(lens):.0f} "
                      f"p25={statistics.quantiles(lens, n=4)[0]:.0f} p75={statistics.quantiles(lens, n=4)[2]:.0f}")
        # samples per label
        if text_col and label_col:
            seen = set()
            for t, l in zip(data[text_col], data[label_col]):
                l = str(l)
                if l in seen:
                    continue
                seen.add(l)
                print(f"  sample[{l}]:", repr(str(t)[:200]))
                if len(seen) >= 4:
                    break


if __name__ == "__main__":
    main()
