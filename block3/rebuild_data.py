"""Rebuild the four-task dataset (per the deck's "Data Selection") from existing splits.

Deck tasks:
  1. SMS spam detection   (sms_spam + enron_spam)  -> binary no/yes
  2. Toxicity count score (jigsaw)                  -> 0..5
  3. Phishing detection   (phishing_email)          -> binary no/yes
  4. Binary toxicity noul (jigsaw `toxic` label)    -> no/yes

The Jigsaw rows already carry both the subtype labels (score) and the umbrella
`toxic` label (noul), so a single comment serves the two aligned tasks. We derive
the binary-toxicity records from the existing `score` records, which keeps each
comment inside the same train/held_out/eval split (no cross-split leakage).

Reads  evaluation/data/{train,held_out,eval}.jsonl
Writes evaluation/data_v2/{train,held_out,eval}.jsonl  (source files are left untouched)
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

SRC = Path("evaluation/data")
DST = Path("evaluation/data_v2")

SMS_PROMPT = "Is this message spam? (no / yes)"
PHISHING_PROMPT = "Is this email a phishing email? (no / yes)"
TOXICITY_PROMPT = "Does this text contain toxic content? (no / yes)"
SCORE_PROMPT = ("Count the applicable toxicity categories (0 to 5): "
                "severe toxic, obscene, threat, insult, identity hate.")


def rebuild_split(split: str) -> int:
    out = []
    for line in (SRC / f"{split}.jsonl").open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        kind = r["type"]
        if kind == "noul":
            src = r.get("source", "")
            if src in ("sms_spam", "enron_spam"):
                r["prompt"] = SMS_PROMPT
                r["task"] = "sms_spam"
            elif src in ("phishing_email", "phishing_email_fallback"):
                r["prompt"] = PHISHING_PROMPT
                r["task"] = "phishing"
            else:
                raise ValueError(f"{split}: unexpected noul source {src!r}")
            out.append(r)
        elif kind == "score":
            r["prompt"] = SCORE_PROMPT
            r["task"] = "score"
            out.append(r)
            toxic = int(r.get("toxic_label", 0))
            out.append({
                "id": r["id"].replace("score_", "noul_", 1),
                "type": "noul",
                "source": "jigsaw_toxicity",
                "task": "noul",
                "prompt": TOXICITY_PROMPT,
                "text": r["text"],
                "label": "yes" if toxic == 1 else "no",
                "options": ["no", "yes"],
                "toxic_label": toxic,
                "subtype_labels": r.get("subtype_labels"),
            })
        else:
            out.append(r)

    (DST / f"{split}.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n", encoding="utf-8"
    )
    return len(out)


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    for split in ("train", "held_out", "eval"):
        n = rebuild_split(split)
        # task-level composition for the deck
        rows = [json.loads(l) for l in (DST / f"{split}.jsonl").open(encoding="utf-8") if l.strip()]
        tasks = Counter()
        for r in rows:
            task = r.get("task", r["type"])
            if task == "sms_spam":
                tasks["sms_spam (sms+enron)"] += 1
            elif task == "phishing":
                tasks["phishing"] += 1
            elif task == "score":
                tasks["score (toxicity count)"] += 1
            elif task == "noul":
                tasks["noul (binary toxicity)"] += 1
            else:
                tasks[f"other:{task}"] += 1
        noul_labels = Counter(r["label"] for r in rows if r["type"] == "noul")
        score_labels = Counter(r["label"] for r in rows if r["type"] == "score")
        print(f"[{split}] {n} rows")
        print(f"   tasks: {dict(tasks)}")
        print(f"   noul labels: {dict(noul_labels)}")
        print(f"   score labels: {dict(sorted(score_labels.items(), key=lambda kv: int(kv[0]))) if score_labels else {}}")

    # sanity: binary-toxicity labels equal the umbrella toxic label, one comment per split
    for split in ("train", "held_out", "eval"):
        rows = [json.loads(l) for l in (DST / f"{split}.jsonl").open(encoding="utf-8") if l.strip()]
        score = {r["text"]: r for r in rows if r["type"] == "score"}
        noul_tox = [r for r in rows if r["type"] == "noul" and r.get("source") == "jigsaw_toxicity"]
        bad = 0
        for r in noul_tox:
            s = score.get(r["text"])
            if s is None or (r["label"] == "yes") != (int(s["toxic_label"]) == 1):
                bad += 1
        print(f"[{split}] binary-toxicity checks: {len(noul_tox)} rows, {bad} label mismatches")


if __name__ == "__main__":
    main()
