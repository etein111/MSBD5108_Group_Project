# T12 Statistical Analysis

The analysis uses 162,270 merged T11 records from real model inference, covering JEV-9B and Qwen3.5-9B, three seeds, four tasks, and four probe types. Within each task, the two models were paired by original sample. There were 628 spam samples, 345 phishing samples, 1,016 binary toxicity samples, and 1,016 toxicity subtype count samples. The two toxicity tasks used the same Jigsaw texts and are reported separately.

## Accuracy

- **JEV had higher accuracy on clean inputs for all three binary tasks.** For spam, accuracy was 92.0% versus 32.6%, a difference of +59.4 percentage points [95% CI +54.3, +64.3], Holm-adjusted p = 0.00745. For phishing, accuracy was 99.7% versus 55.9%, a difference of +43.8 percentage points [95% CI +38.6, +49.3], Holm-adjusted p = 0.00745. For binary toxicity, accuracy was 81.7% versus 66.0%, a difference of +15.6 percentage points [95% CI +12.6, +18.9], Holm-adjusted p = 0.00745.
- **Qwen had higher exact-match accuracy on the toxicity subtype count task.** JEV scored 20.2% and Qwen scored 63.3%, a JEV-minus-Qwen difference of -43.1 percentage points [95% CI -46.6, -39.7], Holm-adjusted p = 0.00745. Qwen also had lower MAE, while JEV had higher QWK. The gold count was 0 for 672 of 1,016 samples, and Qwen predicted 0 for 960 of 1,016 samples at seed 0. This distribution provides context for reading exact-match accuracy alongside QWK.

Accuracy and Macro-F1 are expressed as percentages. Differences are in percentage points. MAE and QWK are reported on their original scales.

| Task | Metric | JEV [95% CI] | Qwen [95% CI] | JEV − Qwen [95% CI] | Holm-adjusted p |
| --- | --- | --- | --- | --- | --- |
| Spam | Accuracy | 92.0 [90.0, 94.1] | 32.6 [29.0, 36.5] | +59.4 [+54.3, +64.3] | 0.00745 |
| Spam | Macro-F1 | 89.8 [87.0, 92.4] | 27.0 [24.2, 29.9] | +62.8 [+58.7, +67.0] | 0.00745 |
| Phishing | Accuracy | 99.7 [99.1, 100.0] | 55.9 [50.4, 61.2] | +43.8 [+38.6, +49.3] | 0.00745 |
| Phishing | Macro-F1 | 99.7 [99.1, 100.0] | 48.0 [42.8, 53.1] | +51.7 [+46.7, +57.0] | 0.00745 |
| Binary toxicity | Accuracy | 81.7 [79.3, 84.1] | 66.0 [63.1, 69.0] | +15.6 [+12.6, +18.9] | 0.00745 |
| Binary toxicity | Macro-F1 | 81.6 [79.2, 84.0] | 64.0 [60.9, 66.9] | +17.6 [+14.3, +21.0] | 0.00745 |
| Toxicity subtype count | Accuracy | 20.2 [17.7, 22.6] | 63.3 [60.3, 66.2] | -43.1 [-46.6, -39.7] | 0.00745 |
| Toxicity subtype count | MAE | 1.947 [1.871, 2.025] | 0.685 [0.621, 0.750] | +1.262 [+1.153, +1.370] | 0.00745 |
| Toxicity subtype count | QWK | 0.210 [0.185, 0.235] | 0.008 [-0.031, 0.049] | +0.201 [+0.155, +0.246] | 0.00745 |

## Latency and Resource Cost

Across all three seeds, JEV had lower p50 latency and higher p95 latency and service time per thousand decisions on all four tasks. The table reports paired differences and adjusted p-values. **All 814 clean requests above 500 ms occurred for JEV at seed 0.** The jobs ran in sequential blocks by model and seed, with no randomization of model-block order or system conditions. The intervals capture sample-level uncertainty within the observed runs.

| Task | Metric | JEV [95% CI] | Qwen [95% CI] | JEV − Qwen [95% CI] | Holm-adjusted p |
| --- | --- | --- | --- | --- | --- |
| Spam | p50 (ms) | 152.3 [151.3, 152.8] | 157.0 [156.3, 158.0] | -4.7 [-5.4, -4.3] | 0.00745 |
| Spam | p95 (ms) | 864.8 [859.0, 868.5] | 194.4 [191.0, 198.6] | +670.5 [+665.7, +674.3] | 0.00745 |
| Spam | Service s per 1,000 decisions | 238.9 [229.9, 248.3] | 163.6 [162.6, 164.7] | +75.3 [+66.8, +84.2] | 0.00745 |
| Phishing | p50 (ms) | 162.0 [161.8, 162.2] | 165.5 [165.3, 165.7] | -3.5 [-3.7, -3.3] | 0.00745 |
| Phishing | p95 (ms) | 857.0 [853.0, 864.2] | 187.9 [187.0, 189.5] | +669.1 [+664.4, +676.7] | 0.00745 |
| Phishing | Service s per 1,000 decisions | 246.2 [234.6, 258.4] | 167.2 [166.6, 167.8] | +79.0 [+67.4, +91.4] | 0.00745 |
| Binary toxicity | p50 (ms) | 140.7 [140.2, 141.3] | 142.9 [142.4, 143.4] | -2.2 [-2.6, -1.7] | 0.00745 |
| Binary toxicity | p95 (ms) | 828.0 [824.6, 832.0] | 165.8 [163.3, 168.7] | +662.2 [+658.7, +666.2] | 0.00745 |
| Binary toxicity | Service s per 1,000 decisions | 199.4 [193.1, 205.8] | 145.9 [145.4, 146.4] | +53.6 [+47.4, +59.7] | 0.00745 |
| Toxicity subtype count | p50 (ms) | 139.5 [139.2, 139.7] | 144.0 [143.8, 144.2] | -4.5 [-4.7, -4.3] | 0.00745 |
| Toxicity subtype count | p95 (ms) | 831.6 [827.7, 834.9] | 168.5 [166.5, 170.9] | +663.2 [+659.5, +666.5] | 0.00745 |
| Toxicity subtype count | Service s per 1,000 decisions | 192.3 [186.3, 198.5] | 148.4 [147.9, 149.0] | +43.9 [+38.0, +49.9] | 0.00745 |

The **post hoc sensitivity analysis** uses seeds 1 and 2 only, with a separate Holm adjustment across 12 tests. The table shows how sensitive tail latency and the resource estimate are to seed 0. The analysis across all three seeds remains the primary analysis.

| Task | Metric | JEV [95% CI] | Qwen [95% CI] | JEV − Qwen [95% CI] | Holm-adjusted p |
| --- | --- | --- | --- | --- | --- |
| Spam | p95 (ms) | 188.3 [181.9, 192.5] | 190.9 [186.2, 194.5] | -2.5 [-4.9, -1.0] | 0.0098 |
| Spam | Service s per 1,000 decisions | 157.3 [156.2, 158.3] | 160.6 [159.6, 161.6] | -3.3 [-3.7, -3.0] | 0.0006 |
| Phishing | p95 (ms) | 169.8 [167.3, 170.8] | 172.5 [169.7, 175.1] | -2.7 [-6.6, +0.3] | 0.089 |
| Phishing | Service s per 1,000 decisions | 160.6 [160.1, 161.1] | 164.0 [163.5, 164.6] | -3.5 [-3.9, -3.1] | 0.0006 |
| Binary toxicity | p95 (ms) | 164.5 [162.6, 166.3] | 166.5 [163.9, 169.3] | -2.0 [-4.1, +0.2] | 0.089 |
| Binary toxicity | Service s per 1,000 decisions | 145.4 [144.9, 146.0] | 146.7 [146.2, 147.3] | -1.3 [-1.8, -0.8] | 0.0006 |
| Toxicity subtype count | p95 (ms) | 155.5 [153.3, 158.1] | 158.2 [156.7, 161.1] | -2.7 [-4.7, -1.1] | 0.0015 |
| Toxicity subtype count | Service s per 1,000 decisions | 141.7 [141.2, 142.2] | 145.5 [145.0, 146.1] | -3.9 [-4.2, -3.5] | 0.0006 |

Resource cost is expressed as the mean service time in seconds on clean requests multiplied by 1,000. This service-time proxy includes tokenization and synchronization waits. Per-request GPU kernel timing and an hourly GPU rate were unavailable. The two Slurm jobs ran on different H800 nodes with different driver versions, so model comparisons are paired within each task. See `timing_by_seed.csv` for per-seed latency and tail counts.

## Raw Probability Calibration

JEV had lower ECE and Brier scores on spam and phishing. JEV also had a lower Brier score on binary toxicity, and Qwen had a lower Brier score on toxicity subtype counting. For both toxicity tasks, the Holm-adjusted p-value for the ECE difference exceeded the significance threshold.

| Task | Metric | JEV [95% CI] | Qwen [95% CI] | JEV − Qwen [95% CI] | Holm-adjusted p |
| --- | --- | --- | --- | --- | --- |
| Spam | ECE (%) | 16.5 [14.5, 18.5] | 34.7 [31.8, 38.0] | -18.2 [-21.7, -15.0] | 0.00745 |
| Spam | Brier | 0.093 [0.084, 0.103] | 0.300 [0.289, 0.311] | -0.207 [-0.224, -0.190] | 0.00745 |
| Phishing | ECE (%) | 11.4 [10.2, 12.6] | 38.8 [36.2, 41.3] | -27.4 [-30.7, -24.1] | 0.00745 |
| Phishing | Brier | 0.025 [0.021, 0.031] | 0.206 [0.187, 0.224] | -0.180 [-0.200, -0.160] | 0.00745 |
| Binary toxicity | ECE (%) | 15.7 [13.7, 17.7] | 17.6 [15.2, 20.3] | -1.9 [-5.2, +0.9] | 1 |
| Binary toxicity | Brier | 0.137 [0.129, 0.146] | 0.227 [0.222, 0.231] | -0.089 [-0.097, -0.082] | 0.00745 |
| Toxicity subtype count | ECE (%) | 19.3 [16.8, 21.8] | 23.7 [20.8, 26.6] | -4.4 [-8.8, +0.0] | 0.8462 |
| Toxicity subtype count | Brier | 0.953 [0.934, 0.971] | 0.626 [0.609, 0.642] | +0.327 [+0.305, +0.349] | 0.00745 |

![Raw probability reliability diagrams for the four tasks](T12-原始概率校准图.png)

For the binary tasks, the reliability diagrams compare `P(yes)` with the observed positive rate, and the Brier score is the mean squared error for the binary outcome. For toxicity subtype counting, the diagrams compare the maximum class probability with observed accuracy, and the Brier score averages the sum of squared errors across all classes. The analysis uses 15 equal-width bins. Error bars show pointwise 95% intervals for the observed rate in each bin. The figure displays bins with at least 10 distinct samples. See `calibration_bins.csv` for all bins. The analysis uses raw probabilities. [Guo et al. (2017)](https://proceedings.mlr.press/v70/guo17a.html)

## Robustness Across Four Probe Types

The primary robustness metric is the **paired difference between original and perturbed accuracy** (original minus perturbed, in percentage points). Positive values indicate an accuracy drop, and negative values indicate higher accuracy after perturbation. Each comparison uses matched samples. For option-based probes, the original records come from the corresponding `choice` setting. An `unsure` response counts as a mismatch with the gold label. In the toxicity subtype count task, option renaming uses verbal level descriptions, so the comparison reflects that specific wording change.

For phishing option renaming, accuracy dropped by 0.3 percentage points for JEV and 25.8 percentage points for Qwen. The JEV-minus-Qwen difference in the drop was -25.5 percentage points [95% CI -30.1, -21.2], Holm-adjusted p = 0.00745. The direction of the difference varies by task and probe. The full tables also report decision flip rate, total variation between probability distributions, change in prompt-injection target selection rate, and abstention rate before removal. These metrics include confidence intervals and tests of differences between the models.

| Task | Probe | JEV drop [95% CI] | JEV p | Qwen drop [95% CI] | Qwen p | JEV − Qwen drop [95% CI] | Difference p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Spam | Prompt injection | -0.5 [-1.3, 0.3] | 1 | 0.6 [-0.3, 1.6] | 1 | -1.1 [-2.4, +0.2] | 1 |
| Spam | Option renaming | 0.8 [-0.3, 2.1] | 1 | 0.6 [0.2, 1.3] | 1 | +0.2 [-1.1, +1.4] | 1 |
| Spam | Option-order permutation | -3.8 [-5.9, -1.8] | 0.0301 | -20.4 [-24.0, -16.7] | 0.00745 | +16.6 [+12.3, +20.5] | 0.00745 |
| Spam | Abstain removal | -1.6 [-2.7, -0.5] | 0.2465 | -0.3 [-0.8, 0.0] | 1 | -1.3 [-2.5, -0.2] | 1 |
| Phishing | Prompt injection | 0.0 [0.0, 0.0] | 1 | 7.8 [5.2, 10.7] | 0.00745 | -7.8 [-10.7, -5.2] | 0.00745 |
| Phishing | Option renaming | 0.3 [0.0, 0.9] | 1 | 25.8 [21.4, 30.4] | 0.00745 | -25.5 [-30.1, -21.2] | 0.00745 |
| Phishing | Option-order permutation | -0.3 [-0.9, 0.0] | 1 | -23.2 [-28.4, -18.0] | 0.00745 | +22.9 [+17.7, +28.1] | 0.00745 |
| Phishing | Abstain removal | 0.0 [0.0, 0.0] | 1 | 3.5 [1.7, 5.5] | 0.0315 | -3.5 [-5.5, -1.7] | 0.0138 |
| Binary toxicity | Prompt injection | -2.1 [-3.3, -0.9] | 0.0513 | 4.4 [0.9, 8.0] | 0.5223 | -6.5 [-10.2, -2.9] | 0.0328 |
| Binary toxicity | Option renaming | -1.2 [-2.0, -0.4] | 0.2607 | 1.0 [0.4, 1.7] | 0.07215 | -2.2 [-3.2, -1.2] | 0.00745 |
| Binary toxicity | Option-order permutation | -9.4 [-11.4, -7.6] | 0.00745 | -8.3 [-12.6, -3.7] | 0.0198 | -1.2 [-5.6, +3.1] | 1 |
| Binary toxicity | Abstain removal | 0.8 [0.0, 1.6] | 1 | 0.0 [-0.4, 0.4] | 1 | +0.8 [-0.1, +1.7] | 1 |
| Toxicity subtype count | Prompt injection | -8.8 [-10.8, -6.7] | 0.00745 | 10.2 [8.1, 12.5] | 0.00745 | -19.0 [-21.9, -16.0] | 0.00745 |
| Toxicity subtype count | Option renaming | 3.8 [1.9, 5.8] | 0.01175 | -9.1 [-12.5, -5.4] | 0.00745 | +12.9 [+8.9, +16.7] | 0.00745 |
| Toxicity subtype count | Option-order permutation | -5.9 [-7.8, -4.1] | 0.00745 | -21.2 [-24.8, -17.4] | 0.00745 | +15.3 [+11.1, +19.2] | 0.00745 |
| Toxicity subtype count | Abstain removal | 1.2 [0.3, 2.2] | 0.6588 | -2.5 [-4.0, -0.9] | 0.1098 | +3.6 [+1.8, +5.5] | 0.00745 |

## Statistical Methods

- The resampling unit is `sample_id` within each task. All repeats across the three seeds, both models, and both original and perturbed records are resampled together as a cluster. Original and perturbed records were also matched using `pair_id`. For each resample, p50 and p95 are recomputed from all requests in the selected clusters.
- Pointwise 95% confidence intervals use 5,000 paired percentile bootstrap resamples. Two-sided p-values use 19,999 paired random swaps. The reliability bins are used for visualization. Model comparisons swap all records for the two models within each sample cluster. Within-model tests of perturbation effects swap the original and perturbed conditions as a group. The Monte Carlo p-value is computed as (number of results at least as extreme + 1) / 20,000.
- The 149 primary tests use a single Holm adjustment with a significance threshold of 0.05. The 12 post hoc sensitivity tests form a separate family. The reported confidence intervals are pointwise. Statistical significance is assessed using adjusted p-values. [Paired permutation test definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html) · [Holm (1979)](https://www.jstor.org/stable/4615733)
- Gold labels for the Jigsaw subtype count and binary toxicity tasks come from the source data. The analysis script uses sample IDs and inference fields for statistical calculations. See `analysis_settings.json` for the analysis seed, binning parameters, and test counts. To reproduce the results, extract the T11 archive and run `python t12_analysis.py /path/to/all_results.jsonl ./t12-output` from the directory containing the analysis script. The script uses NumPy and Matplotlib.
