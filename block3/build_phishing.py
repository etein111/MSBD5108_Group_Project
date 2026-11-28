"""Build a balanced, length-matched phishing split for the JevBench harness.

The public "phishing" corpora conflate phishing/spam with mailing-list ham and
differ in length between classes, which lets a model win on format alone. This
builder:

  * normalises a HF corpus to (text, label),
  * de-duplicates by text,
  * length-stratifies and balances the two classes (equal counts per length bin),
  * splits 60/20/20 into train/held_out/eval,
  * writes JevBench ``noul`` records with the phishing prompt.

Usage (on the cluster, env with ``datasets``/``huggingface_hub``):
    python -m block3.build_phishing --source zefang --out evaluation/data_phishing
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

PROMPT = "Is this email a phishing email? (no / yes)"


def load_pairs(source: str) -> list:
    """Return a list of (text, label) where label is 'yes' (phishing) or 'no'."""
    if source == "zefang":
        from datasets import load_dataset
        ds = load_dataset("zefang-liu/phishing-email-dataset", split="train")
        pairs = []
        for row in ds:
            lab = str(row["Email Type"]).strip().lower()
            pairs.append((str(row["Email Text"]), "yes" if "phishing" in lab else "no"))
        return pairs
    if source == "ealvaradob":
        from huggingface_hub import hf_hub_download
        path = hf_hub_download("ealvaradob/phishing-dataset", "texts.json", repo_type="dataset")
        data = json.load(open(path))
        return [(str(r["text"]), "yes" if int(r["label"]) == 1 else "no") for r in data]
    raise ValueError(source)


def dedup(pairs: list) -> list:
    seen, out = set(), []
    for text, label in pairs:
        text = text.strip()
        if len(text) < 20:
            continue
        key = (label, text)
        if key in seen:
            continue
        seen.add(key)
        out.append((text, label))
    return out


def length_matched(pairs: list, target_total: int, n_bins: int, seed: int) -> list:
    rng = random.Random(seed)
    lengths = np.array([len(t) for t, _ in pairs])
    edges = np.quantile(lengths, np.linspace(0, 1, n_bins + 1))
    bins = defaultdict(lambda: {"yes": [], "no": []})
    for text, label in pairs:
        b = int(np.clip(np.digitize(len(text), edges[1:-1]), 0, n_bins - 1))
        bins[b][label].append((text, label))
    cap = max(1, target_total // (2 * n_bins))
    out = []
    for b in sorted(bins):
        for label in ("yes", "no"):
            bucket = bins[b][label]
            rng.shuffle(bucket)
            take = min(cap, len(bucket))
            out.extend(bucket[:take])
    rng.shuffle(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("zefang", "ealvaradob"), default="zefang")
    parser.add_argument("--out", type=Path, default=Path("evaluation/data_phishing"))
    parser.add_argument("--target-total", type=int, default=4000)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--max-chars", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", default=PROMPT)
    args = parser.parse_args()

    pairs = dedup(load_pairs(args.source))
    n_yes = sum(1 for _, l in pairs if l == "yes")
    print(f"loaded+dedup: {len(pairs)} (phishing {n_yes}, legit {len(pairs)-n_yes})")
    matched = length_matched(pairs, args.target_total, args.bins, args.seed)
    n_yes = sum(1 for _, l in matched if l == "yes")
    print(f"length-matched: {len(matched)} (phishing {n_yes}, legit {len(matched)-n_yes})")

    # cap length, drop overly long after capping is fine for the task
    matched = [(t[: args.max_chars], l) for t, l in matched]
    n = len(matched)
    n_train, n_hold = int(n * 0.6), int(n * 0.2)
    splits = {"train": matched[:n_train], "held_out": matched[n_train:n_train + n_hold],
              "eval": matched[n_train + n_hold:]}
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with (args.out / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for i, (text, label) in enumerate(rows):
                handle.write(json.dumps({
                    "id": f"phishing_{args.source}_{name}_{i:05d}",
                    "type": "noul", "source": args.source, "task": "phishing",
                    "prompt": args.prompt, "text": text, "label": label,
                    "options": ["no", "yes"],
                }, ensure_ascii=False) + "\n")
        pos = sum(1 for _, l in rows if l == "yes")
        print(f"  {name}: {len(rows)} rows (phishing {pos}, legit {len(rows)-pos})")


if __name__ == "__main__":
    main()
