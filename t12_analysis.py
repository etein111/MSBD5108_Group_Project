"""T12 statistics from T11 JSONL. Run python t12_analysis.py INPUT_JSONL OUTPUT_DIR."""
import argparse
import csv
import json
import os
from pathlib import Path

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

TASKS = {"spam_binary": ("Spam", 628), "phishing_binary": ("Phishing", 345),
         "toxicity_binary": ("Toxicity binary", 1016), "toxicity_score": ("Toxicity count", 1016)}
PROBES = ("prompt_injection", "option_renaming", "option_order_permutation", "abstain_removal")
MODELS = ("jev", "qwen")
SEEDS = (0, 1, 2)
N_BOOT = 5000
N_PERM = 19999
RNG_SEED = 12026
BINS = 15


def load(path):
    data = {task: {} for task in TASKS}
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            assert row["backend"] == "local" and row["status"] == "success"
            assert row["model"] in MODELS and row["seed"] in SEEDS
            slot = (row["model"], row["seed"], row["probe"], row["variant"])
            sample = data[row["task"]].setdefault(row["sample_id"], {})
            assert slot not in sample
            sample[slot] = (str(row["decision"]), str(row["gold"]), row["probability"],
                          row["latency_ms"], row["gpu_seconds_proxy"], row["pair_id"])
            count += 1
    assert count == 162270
    for task, (_, n) in TASKS.items():
        assert len(data[task]) == n
        for sample in data[task].values():
            gold = {row[1] for row in sample.values()}
            assert len(gold) == 1 and len(sample) == 54
            for model in MODELS:
                for seed in SEEDS:
                    assert sample[model, seed, "clean", None][5] is None
                    for probe in PROBES:
                        first = sample[model, seed, probe, "original"]
                        second = sample[model, seed, probe, "perturbed"]
                        assert first[5] is not None and first[5] == second[5]
                        assert first[5] == sample["jev", 0, probe, "original"][5]
    return data


def positive(value):
    return np.abs(value)


def confidence_interval(samples):
    return np.quantile(samples, (.025, .975), axis=0).T


def resample_stats(a, b, stat, rng):
    """Apply a statistic of summed cluster features to paired bootstrap/swaps."""
    n = len(a)
    total_a, total_b = a.sum(axis=0), b.sum(axis=0)
    point_a, point_b = stat(total_a[None])[0], stat(total_b[None])[0]
    boot_a, boot_b = [], []
    for start in range(0, N_BOOT, 128):
        weights = rng.multinomial(n, np.full(n, 1/n), size=min(128, N_BOOT-start)).astype(float)
        boot_a.append(stat(weights @ a))
        boot_b.append(stat(weights @ b))
    boot_a, boot_b = np.vstack(boot_a), np.vstack(boot_b)
    exceed = np.zeros_like(point_a, dtype=int)
    combined = total_a + total_b
    diff = a - b
    for start in range(0, N_PERM, 128):
        signs = 2.0 * rng.integers(0, 2, size=(min(128, N_PERM-start), n), dtype=np.int8) - 1
        d = signs @ diff
        null = stat((combined+d)/2) - stat((combined-d)/2)
        exceed += (positive(null) >= positive(point_a-point_b) - 1e-12).sum(axis=0)
    return point_a, point_b, confidence_interval(boot_a), confidence_interval(boot_b), confidence_interval(boot_a-boot_b), (exceed+1)/(N_PERM+1)


def timed_stats(a, b, rng):
    def stat(batch):
        return np.quantile(batch.reshape(batch.shape[0], -1), (.5, .95), axis=1).T

    n = len(a)
    ap, bp = stat(a[None])[0], stat(b[None])[0]
    boot_a, boot_b = [], []
    for start in range(0, N_BOOT, 64):
        chosen = rng.integers(n, size=(min(64, N_BOOT-start), n))
        boot_a.append(stat(a[chosen]))
        boot_b.append(stat(b[chosen]))
    boot_a, boot_b = np.vstack(boot_a), np.vstack(boot_b)
    exceed = np.zeros(2, dtype=int)
    for start in range(0, N_PERM, 64):
        mask = rng.integers(0, 2, size=(min(64, N_PERM-start), n, 1), dtype=np.int8).astype(bool)
        delta = stat(np.where(mask, a, b)) - stat(np.where(mask, b, a))
        exceed += (positive(delta) >= positive(ap-bp)-1e-12).sum(axis=0)
    return ap, bp, confidence_interval(boot_a), confidence_interval(boot_b), confidence_interval(boot_a-boot_b), (exceed+1)/(N_PERM+1)


def comparisons(result, names, metadata):
    a, b, a_ci, b_ci, effect_ci, p = result
    for i, name in enumerate(names):
        yield {**metadata, "metric": name, "jev": float(a[i]), "jev_lo": float(a_ci[i, 0]),
               "jev_hi": float(a_ci[i, 1]), "qwen": float(b[i]), "qwen_lo": float(b_ci[i, 0]),
               "qwen_hi": float(b_ci[i, 1]), "diff_jev_minus_qwen": float(a[i]-b[i]),
               "diff_lo": float(effect_ci[i, 0]), "diff_hi": float(effect_ci[i, 1]), "p": float(p[i])}


def clean_features(samples, model, ordinal):
    labels = tuple(str(i) for i in range(6)) if ordinal else ("no", "yes")
    label_index = {k: v for v, k in enumerate(labels)}
    clean = [[s[model, seed, "clean", None] for seed in SEEDS] for s in samples]
    prediction = np.array([[label_index[r[0]] for r in rows] for rows in clean])
    truth = np.array([[label_index[r[1]] for r in rows] for rows in clean])
    probs = np.array([[[r[2][label] for label in labels] for r in rows] for rows in clean])
    duration = np.array([[r[3] for r in rows] for rows in clean])
    proxy = np.array([[r[4] for r in rows] for rows in clean])
    ok = (prediction == truth).astype(float)
    if ordinal:
        x = probs.max(axis=2)
        y = ok
        brier = ((probs - np.eye(6)[truth])**2).sum(axis=2)
    else:
        x = probs[:, :, 1]
        y = truth.astype(float)
        brier = (x-y)**2
    bins = np.minimum((x*BINS).astype(int), BINS-1)
    membership = np.eye(BINS)[bins]
    counts = membership.sum(axis=1)
    xsum = (membership*x[:, :, None]).sum(axis=1)
    ysum = (membership*y[:, :, None]).sum(axis=1)
    confusion = np.eye(len(labels)**2)[truth*len(labels)+prediction].sum(axis=1)/3
    features = np.column_stack((ok.mean(axis=1), brier.mean(axis=1), 1000*proxy.mean(axis=1),
                                (ysum-xsum)/3, confusion))
    return features, duration, (counts, xsum, ysum)


def clean_stat(ordinal, n):
    width = 6 if ordinal else 2
    distances = np.abs(np.arange(width)[:, None]-np.arange(width))
    sq = distances**2

    def stat(features):
        acc, brier, cost = features[:, :3].T/n
        ece = np.abs(features[:, 3:3+BINS]).sum(axis=1)/n
        confusion = features[:, 3+BINS:].reshape(-1, width, width)
        if ordinal:
            mae = (confusion*distances).sum(axis=(1, 2))/n
            observed = (confusion*sq).sum(axis=(1, 2))
            expected = (confusion.sum(axis=1)[:, None, :]*confusion.sum(axis=2)[:, :, None]*sq).sum(axis=(1, 2))/n
            return np.column_stack((acc, brier, ece, cost, mae, 1-observed/expected))
        f1 = 2*np.diagonal(confusion, axis1=1, axis2=2)/(confusion.sum(axis=1)+confusion.sum(axis=2))
        return np.column_stack((acc, brier, ece, cost, f1.mean(axis=1)))
    return stat


def reliability_rows(parts, task, model, rng):
    count, xsum, ysum = parts
    n = len(count)
    support = (count > 0).sum(axis=0)
    totals = count.sum(axis=0)
    bootstrap = []
    for start in range(0, N_BOOT, 128):
        w = rng.multinomial(n, np.full(n, 1/n), size=min(128, N_BOOT-start)).astype(float)
        den = w @ count
        bootstrap.append(np.divide(w@ysum, den, out=np.full_like(den, np.nan), where=den>0))
    bootstrap = np.vstack(bootstrap)
    rows = []
    for i, total in enumerate(totals):
        if total == 0:
            continue
        bounds = np.nanquantile(bootstrap[:, i], (.025, .975))
        rows.append(dict(task=task, model=model, bin=i+1, low=i/BINS, high=(i+1)/BINS,
                         sample_clusters=int(support[i]), records=int(total), forecast=float(xsum[:, i].sum()/total),
                         observed=float(ysum[:, i].sum()/total), lo=float(bounds[0]), hi=float(bounds[1])))
    return rows


def robustness_features(samples, model, probe):
    def measure(original, perturbed):
        pred0, gold, p0 = original[:3]
        pred1, gold1, p1 = perturbed[:3]
        assert gold == gold1
        a, b = float(pred0 == gold), float(pred1 == gold)
        tv = .5*sum(abs(p0.get(k, 0)-p1.get(k, 0)) for k in p0.keys() | p1.keys())
        core = (a, b, a-b, float(pred0 != pred1), tv)
        if probe == "abstain_removal":
            return (*core, float(pred0 == "unsure"))
        if probe == "prompt_injection":
            target = "5" if gold not in ("no", "yes") else "yes"
            return (*core, float(pred1 == target)-float(pred0 == target))
        return core

    return np.array([[measure(s[model, seed, probe, "original"], s[model, seed, probe, "perturbed"])
                      for seed in SEEDS] for s in samples]).mean(axis=1)


def within_drop(data, rng):
    """Measure an accuracy drop and test a within-model perturbation effect."""
    n = len(data)
    point = float(data.mean())
    boot, exceed = [], 0
    for start in range(0, N_BOOT, 128):
        w = rng.multinomial(n, np.full(n, 1/n), size=min(128, N_BOOT-start)).astype(float)
        boot.append(w@data/n)
    for start in range(0, N_PERM, 128):
        signs = 2.0*rng.integers(0, 2, size=(min(128, N_PERM-start), n), dtype=np.int8)-1
        exceed += (np.abs(signs@data/n) >= abs(point)-1e-12).sum()
    lo, hi = np.quantile(np.concatenate(boot), (.025, .975))
    return {"drop": point, "drop_lo": float(lo), "drop_hi": float(hi), "p": float((exceed+1)/(N_PERM+1))}


def holm(rows):
    ordered = sorted(rows, key=lambda r: r["p"])
    best = 0.
    for i, row in enumerate(ordered):
        best = max(best, min(1, row["p"]*(len(ordered)-i)))
        row["p_holm"] = best


def table(path, rows):
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_reliability(rows, path):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout="constrained")
    colors = {"jev": "#0b6e99", "qwen": "#ad4e1a"}
    for ax, (task, (title, n)) in zip(axes.ravel(), TASKS.items()):
        ax.plot([0, 1], [0, 1], linestyle="--", color="#818894", linewidth=1)
        for model in MODELS:
            visible = [r for r in rows if r["task"] == task and r["model"] == model and r["sample_clusters"] >= 10]
            ax.plot([r["forecast"] for r in visible], [r["observed"] for r in visible], "o",
                    color=colors[model], markersize=4, label=model.upper())
            ax.vlines([r["forecast"] for r in visible], [r["lo"] for r in visible], [r["hi"] for r in visible],
                      color=colors[model], alpha=.45, linewidth=1)
        ax.set_title(f"{title} · n = {n:,}", loc="left", fontweight="bold")
        ax.set(xlim=(0, 1), ylim=(0, 1))
        ax.set_aspect("equal")
        ax.grid(alpha=.19)
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xlabel("Mean confidence" if task == "toxicity_score" else "Mean predicted P(yes)")
        ax.set_ylabel("Observed accuracy" if task == "toxicity_score" else "Observed positive rate")
    axes[0, 0].legend(frameon=False, loc="upper left")
    fig.suptitle("Raw probability calibration · clean inputs", fontsize=16, fontweight="bold")
    fig.supxlabel("15 equal-width bins · pointwise 95% sample bootstrap intervals · bins with ≥10 samples shown", fontsize=9)
    fig.savefig(path, dpi=190)
    plt.close(fig)


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("input", type=Path)
    args.add_argument("output", type=Path)
    options = args.parse_args()
    options.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)
    data = load(options.input)
    clean, reliability, robust, within, sensitivity, by_seed = [], [], [], [], [], []
    for task, per_sample in data.items():
        samples = list(per_sample.values())
        n = len(samples)
        ordinal = task == "toxicity_score"
        a, lat_a, rel_a = clean_features(samples, "jev", ordinal)
        b, lat_b, rel_b = clean_features(samples, "qwen", ordinal)
        meta = {"task": task, "sample_clusters": n, "seed_repeats": 3}
        names = ["accuracy", "brier", "ece", "service_seconds_per_1000"] + (["mae", "qwk"] if ordinal else ["macro_f1"])
        clean.extend(comparisons(resample_stats(a, b, clean_stat(ordinal, n), rng), names, meta))
        clean.extend(comparisons(timed_stats(lat_a, lat_b, rng), ["p50_ms", "p95_ms"], meta))
        sensitivity_meta = {**meta, "seed_repeats": 2}
        sensitivity.extend(comparisons(timed_stats(lat_a[:, 1:], lat_b[:, 1:], rng),
                                       ["p50_ms", "p95_ms"], sensitivity_meta))
        sensitivity.extend(comparisons(
            resample_stats(lat_a[:, 1:].mean(axis=1, keepdims=True),
                           lat_b[:, 1:].mean(axis=1, keepdims=True), lambda f: f/n, rng),
            ["service_seconds_per_1000"], sensitivity_meta))
        for model, latency in zip(MODELS, (lat_a, lat_b)):
            for index, seed in enumerate(SEEDS):
                values = latency[:, index]
                p50, p95 = np.quantile(values, (.5, .95))
                by_seed.append({"task": task, "model": model, "seed": seed, "sample_clusters": n,
                                "p50_ms": p50, "p95_ms": p95, "service_seconds_per_1000": values.mean(),
                                "records_above_500ms": int((values > 500).sum())})
        for model, parts in zip(MODELS, (rel_a, rel_b)):
            reliability.extend(reliability_rows(parts, task, model, rng))
        for probe in PROBES:
            a = robustness_features(samples, "jev", probe)
            b = robustness_features(samples, "qwen", probe)
            names = ["original_accuracy", "perturbed_accuracy", "accuracy_drop", "flip_rate", "probability_tv"]
            if probe == "abstain_removal":
                names.append("original_abstain_rate")
            if probe == "prompt_injection":
                names.append("target_rate_change")
            robust.extend(comparisons(resample_stats(a, b, lambda f: f/n, rng), names, {**meta, "probe": probe}))
            for model, matrix in zip(MODELS, (a, b)):
                within.append({**meta, "probe": probe, "model": model, **within_drop(matrix[:, 2], rng)})
        print(f"Completed {task}: {n} paired samples", flush=True)
    holm(clean + robust + within)
    holm(sensitivity)
    table(options.output / "clean_comparisons.csv", clean)
    table(options.output / "robustness_comparisons.csv", robust)
    table(options.output / "robustness_within_model.csv", within)
    table(options.output / "calibration_bins.csv", reliability)
    table(options.output / "timing_by_seed.csv", by_seed)
    table(options.output / "timing_sensitivity_seeds_1_2.csv", sensitivity)
    plot_reliability(reliability, options.output / "T12-原始概率校准图.png")
    settings = {"bootstrap": N_BOOT, "paired_permutations": N_PERM, "analysis_seed": RNG_SEED,
                "holm_test_count": len(clean)+len(robust)+len(within), "sensitivity_holm_test_count": len(sensitivity), "reliability_bins": BINS}
    (options.output / "analysis_settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print(settings, flush=True)


if __name__ == "__main__":
    main()
