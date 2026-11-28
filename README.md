# JevBench — Typed Decisions vs a General-Purpose LLM for Content Safety

Compares **JEV-9B** (a typed decision model: frozen Qwen3.5-9B backbone + a small
decision head) against **Qwen3.5-9B** as a controlled baseline (same backbone, only
the decision mechanism differs). Both models receive the **same prompts and the same
few-shot examples**, so the comparison isolates the decision architecture.

The benchmark measures four dimensions: **accuracy, probability calibration,
inference cost, and latency**, plus **robustness** under four adversarial probes.

## Tasks

| Task | Decision type | Labels | Source |
| --- | --- | --- | --- |
| SMS spam | binary (`noul`) | `no` / `yes` | `sms_spam` + `enron_spam` |
| Phishing | binary (`noul`) | `no` / `yes` | public corpus, length-matched |
| Toxicity (binary) | binary (`noul`) | `no` / `yes` | Jigsaw umbrella `toxic` label |
| Toxicity (count) | ordinal (`score`) | `0`–`5` | Jigsaw five-subtype sum |

## Repository layout

| Path | Contents |
| --- | --- |
| `deployment/` | JEV-9B serving: FastAPI runtime, HTTP protocol, slot/verbalizer handling |
| `evaluation/` | Data pipeline, `JevBenchHarness`, metrics (ECE/Brier/AUROC/QWK, bootstrap CIs) |
| `pipeline/` | Model adapters, post-hoc calibration, adversarial probes, batch runner |
| `analysis/` | T11 patch and T12 statistical analysis, robustness comparisons |

`pipeline/README.md` documents the comparison pipeline in detail; `deployment/README.md`
and `evaluation/README.md` cover serving and evaluation respectively.

## Quick start

```bash
# 1) install the evaluation environment
pip install -r requirements-eval.txt

# 2) the evaluation data is committed (evaluation/data_v2 + evaluation/data_phishing);
#    evaluation/data_pipeline.py is the original two-task generator, kept for provenance

# 3) serve the model, then predict and calibrate (GPU setup: see deployment/README.md)
python -m deployment.server          # JEV-9B service on :8000 (run in one terminal)
python -m pipeline predict   --model jev --out pipeline/outputs/eval/jev.jsonl   # in another
python -m pipeline calibrate --pred-dir pipeline/outputs/eval --out-dir pipeline/outputs/calibration

# 4) offline tests (no GPU)
python -m unittest discover -s pipeline/tests
```

No GPU available? Use `--model mock` for `pipeline predict` to exercise the pipeline
end to end (the numbers are meaningless).

## Key results

Measured on HKUST SuperPOD (one H800, 4-bit NF4). JEV-9B ranks best on every task:

| Task (eval n) | JEV-9B AUC | Qwen3.5-9B AUC |
| --- | --- | --- |
| SMS spam (628) | **0.967** | 0.857 |
| Phishing, length-matched (797) | **0.914** | 0.713 |
| Toxicity binary (1,016) | **0.928** | 0.699 |

The toxicity-count task has no usable ordinal signal in Qwen (QWK = 0); JEV's only flaw
is a decision-head location bias, fixed by ordinal scaling (`softmax(logits/T + γ·i)`,
γ = −2.0) at report time. Full tables, reliability diagrams and methodology are in
[`pipeline/README.md`](pipeline/README.md).

## Team

GONG Ruiting · Ye Chuan · Zheng Zhiqi — HKUST MSBD 5018 group project.
Scope and plan: [`proposal.md`](proposal.md), [`PLAN.md`](PLAN.md).
