# Block 2: Data and Evaluation

## Install and run

Run these commands from this directory with the same Python environment used by the project:

```powershell
python -m pip install --upgrade -r requirements-eval.txt
python main.py
```

The pipeline downloads the source corpora, creates data/train.jsonl, data/held_out.jsonl, and data/eval.jsonl, then runs the harness with MockModelAdapter. The mock predictions and dashboard only check that the Block 2 pipeline runs; they are not real model results. The dashboard is written to outputs/mock_jevbench_evaluation_dashboard.png so it is clearly separated from dataset files.

## Data sources

The `noul` task samples across three sources. Spam or phishing is labeled `yes`; ham or safe is labeled `no`.

- `ucirvine/sms_spam`: SMS spam.
- `SetFit/enron_spam`: Enron spam email.
- `drorrabin/phishing_emails-data`: phishing and safe email. The pipeline falls back to `jhonrayo99/phishing-email-balanced-6000` if this source cannot load.

The score task loads the English Arsive/toxicity_classification_jigsaw dataset from Hugging Face (train split, about 26,000 rows). The score is the sum of severe_toxic, obscene, threat, insult, and identity_hate, producing a category count from 0 to 5. The umbrella toxic label is retained separately as toxic_label and is not added to the score. Each JSONL sample also keeps the five source subtype labels. The dataset card describes the training data as a balanced sample of toxic and clean comments.

Each sample includes a `source` field. The default total is 10,000 samples, split about evenly between `noul` and `score`; the `noul` allocation is distributed across its three sources. Call `load_and_prepare_data(total_samples=...)` to request a different total. Exact duplicate texts are removed before splitting, so the final JSONL total may be lower than the pre-deduplication request.

## Hugging Face Brotli download error

If a download fails with `output_buffer_limit`, upgrade dependencies in the same Python environment that runs `main.py`:

```powershell
python -m pip install --upgrade -r requirements-eval.txt
```
