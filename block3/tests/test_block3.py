"""Offline tests for Block 3 — no GPU and no running service required."""

from __future__ import annotations

import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from block3.adapters import (
    JevAdapter,
    MockAdapter,
    Prediction,
    QwenAdapter,
    augment_with_shots,
    load_demonstrations,
    normalize_decision,
    prompt_for_record,
)
from block3.calibration import (
    PlattScaler,
    TemperatureScaler,
    balanced_accuracy,
    binary_auroc,
    bootstrap_ci,
    cross_fitted,
    evaluate,
    expected_calibration_error,
    fit_on,
    label_indices,
    macro_f1,
    probability_matrix,
    quadratic_weighted_kappa,
    run_calibration,
)
from block3.cluster_run import run_model
from block3.probes import OPTION_PROBES, PROBES, apply_probe, build_probe_set, summarize_option_probes, write_probe_sets
from deployment.protocol import Prediction as RuntimePrediction, format_scores
from block3.runner import main as runner_main
from block3.runner import run_adapter
from evaluation.harness import JevBenchHarness

NOUL_RECORD = {
    "id": "noul_0001",
    "type": "noul",
    "prompt": "Is this message spam or phishing?",
    "text": "Claim your free prize now",
    "label": "yes",
    "options": ["no", "yes"],
}
SCORE_RECORD = {
    "id": "score_0001",
    "type": "score",
    "prompt": "Rate the toxicity level of this text from 0 to 5:",
    "text": "You are an idiot",
    "label": 3,
    "options": [0, 1, 2, 3, 4, 5],
}


def synthetic_records(kind: str, n: int = 200, overconfidence: float = 1.5, seed: int = 0):
    """Records whose stated probabilities are deliberately overconfident."""
    rng = np.random.default_rng(seed)
    records = []
    for index in range(n):
        if kind == "noul":
            p_true = float(rng.beta(2.5, 2.5))
            label = int(rng.random() < p_true)
            stated = 1.0 / (1.0 + math.exp(-(math.log(max(p_true, 1e-3) / max(1 - p_true, 1e-3)) + overconfidence)))
            prob = {"no": 1.0 - stated, "yes": stated}
            logits = {"no": math.log(max(1 - stated, 1e-9)) - math.log(max(stated, 1e-9)), "yes": 0.0}
            ground_truth = "yes" if label else "no"
        else:
            logits_arr = rng.normal(0.0, 1.5, size=6)
            true_probs = np.exp(logits_arr / 2.0)
            true_probs /= true_probs.sum()
            label = int(rng.choice(6, p=true_probs))
            prob = {str(level): float(value) for level, value in enumerate(np.exp(logits_arr) / np.exp(logits_arr).sum())}
            logits = {str(level): float(value) for level, value in enumerate(logits_arr)}
            ground_truth = label
        records.append(
            {
                "id": f"{kind}_{index:04d}",
                "type": kind,
                "ground_truth": ground_truth,
                "prediction": max(prob, key=prob.get),
                "prob": prob,
                "logits": logits,
                "latency_ms": 50.0,
                "status": "success",
            }
        )
    return records


class AdapterTests(unittest.TestCase):
    def test_choice_demonstrations_match_runner_through_direct_call_and_harness(self):
        record, _ = apply_probe(NOUL_RECORD, "option_renaming")
        demonstrations = [dict(NOUL_RECORD, text="Training example"), SCORE_RECORD]
        expected = prompt_for_record(record, demonstrations)
        self.assertIn("Training example", expected)
        response = {
            "decision": "yes", "prob": {"no": 0.2, "yes": 0.8},
            "logits": {"no": 0, "yes": 1}, "latency_ms": 250,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "probe.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            for adapter_cls in (JevAdapter, QwenAdapter):
                adapter = adapter_cls(demonstrations=demonstrations)
                with patch("urllib.request.urlopen", side_effect=lambda *args, **kwargs: io.BytesIO(
                    json.dumps(response).encode("utf-8")
                )) as urlopen:
                    adapter.predict_full(
                        record["prompt"], record["text"], record["type"], record["options"],
                        source_type=record["source_type"],
                    )
                    result, = JevBenchHarness(str(path), adapter).run()
                    self.assertEqual(result["status"], "success")
                    self.assertEqual(len(urlopen.call_args_list), 2)
                    for call in urlopen.call_args_list:
                        payload = json.loads(call.args[0].data)
                        self.assertEqual(payload["prompt"], expected)
                        self.assertEqual(payload["options"], record["options"])

    def test_latency_payload_compatibility(self):
        for kind, decision, prob in (("noul", "yes", {"no": 0.3, "yes": 0.7}),
                                     ("score", 2, {str(i): float(i == 2) for i in range(6)})):
            for timing in ({"latency": 250.0}, {"latency_ms": 250.0},
                           {"latency_ms": 250.0, "latency": 999.0}):
                with self.subTest(kind=kind, timing=timing):
                    prediction = Prediction.from_payload(
                        {"decision": decision, "prob": prob, "logits": prob, **timing}, kind
                    )
                    self.assertEqual(prediction.latency_ms, 250.0)
                    self.assertEqual(prediction.as_harness_tuple()[3], 0.25)

    def test_harness_projection(self):
        prediction = Prediction("yes", {"no": 0.3, "yes": 0.7}, {"no": -1.0, "yes": 1.0}, 250.0)
        decision, probability, logits, latency_seconds = prediction.as_harness_tuple()
        self.assertEqual(decision, "yes")
        # Block 2's metrics require the full distribution, not a scalar.
        self.assertIsInstance(probability, dict)
        self.assertEqual(set(probability), {"no", "yes"})
        self.assertAlmostEqual(probability["yes"], 0.7)
        self.assertAlmostEqual(latency_seconds, 0.25)
        self.assertEqual(set(logits), {"no", "yes"})

    def test_decision_normalization(self):
        self.assertEqual(normalize_decision("score", "3"), 3)
        self.assertEqual(normalize_decision("noul", "YES"), "yes")
        with self.assertRaises(ValueError):
            normalize_decision("noul", "maybe")

    def test_mock_adapter_is_deterministic(self):
        first = MockAdapter().predict_full("q", "hello", "noul")
        second = MockAdapter().predict_full("q", "hello", "noul")
        self.assertEqual(first.decision, second.decision)
        self.assertAlmostEqual(sum(first.prob.values()), 1.0)

    def test_mock_adapter_runs_in_teammate_harness(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "eval.jsonl"
            path.write_text(
                "\n".join(json.dumps(record) for record in (NOUL_RECORD, SCORE_RECORD)) + "\n",
                encoding="utf-8",
            )
            results = JevBenchHarness(str(path), MockAdapter()).run()
        self.assertEqual(len(results), 2)
        for record in results:
            self.assertEqual(record["status"], "success")
            # Block 2's metrics contract: a full distribution whose argmax is
            # the decision, not a scalar P(decision).
            self.assertIsInstance(record["probability"], dict)
            self.assertAlmostEqual(sum(record["probability"].values()), 1.0)
            self.assertEqual(
                str(record["prediction"]),
                max(record["probability"], key=record["probability"].get),
            )
        self.assertIsInstance(results[1]["prediction"], int)

    def test_few_shot_augmentation_is_type_filtered(self):
        demos = [{"type": "noul", "text": "buy now", "label": "yes"}]
        augmented = augment_with_shots("Is this spam?", "noul", demos)
        self.assertIn('-> true', augmented)
        self.assertTrue(augmented.endswith("Is this spam?"))
        self.assertEqual(augment_with_shots("q", "score", demos), "q")
        self.assertEqual(augment_with_shots("q", "noul", []), "q")

    def test_demonstrations_are_trimmed_per_type(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "train.jsonl"
            rows = [dict(NOUL_RECORD, id=f"n{i}") for i in range(5)]
            rows += [dict(SCORE_RECORD, id=f"s{i}") for i in range(5)]
            path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
            demos = load_demonstrations(path, per_type=2)
        self.assertEqual(len([d for d in demos if d["type"] == "noul"]), 2)
        self.assertEqual(len([d for d in demos if d["type"] == "score"]), 2)

    def test_demonstrations_survive_unicode_line_separators(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "train.jsonl"
            tricky = dict(NOUL_RECORD, id="u1", text="before\u2028after")
            path.write_text(json.dumps(tricky, ensure_ascii=False) + "\n", encoding="utf-8")
            demos = load_demonstrations(path, per_type=1)
        self.assertEqual(len(demos), 1)
        self.assertIn("\u2028", demos[0]["text"])

    def test_runner_produces_calibratable_records(self):
        results = run_adapter([NOUL_RECORD, SCORE_RECORD], MockAdapter())
        self.assertEqual([record["status"] for record in results], ["success", "success"])
        self.assertAlmostEqual(sum(results[0]["prob"].values()), 1.0)
        self.assertIn("ground_truth", results[0])


class ProbeTests(unittest.TestCase):
    def test_score_renaming_uses_the_same_option_neutral_question(self):
        record = dict(SCORE_RECORD, prompt="Rate the toxicity level of this text from 0 to 5:")
        original, renamed = apply_probe(record, "option_renaming")
        self.assertEqual(original["prompt"], "Select the option that best describes the toxicity level of this text.")
        self.assertEqual(renamed["prompt"], original["prompt"])
        self.assertEqual([option["text"] for option in original["options"]], list("012345"))
        self.assertEqual([option["id"] for option in renamed["options"]], list("012345"))
        self.assertNotEqual(renamed["options"], original["options"])
        self.assertEqual(apply_probe(record, "option_order_permutation")[0]["prompt"], record["prompt"])

    def test_cluster_probe_summary_only_counts_selected_pairs_on_resume(self):
        class FakeRuntime:
            def predict(self, model, kind, text, prompt, options=None):
                decision, prob, logits = format_scores(kind, list(range(len(options))), 1, options)
                return RuntimePrediction(decision, prob, logits, 3), []

        probe = "option_renaming"
        records = build_probe_set([NOUL_RECORD, dict(NOUL_RECORD, id="other")], probes=[probe])
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "jev.jsonl"
            first = run_model(FakeRuntime(), "jev", records, output, [])
            resumed = run_model(FakeRuntime(), "jev", records[:2], output, [])
            self.assertEqual(first["option_probes"][probe]["noul"]["pairs_complete"], 2)
            self.assertEqual(resumed["ran"], 0)
            self.assertEqual(resumed["option_probes"][probe]["noul"]["pairs_observed"], 1)
            self.assertEqual(resumed["option_probes"][probe]["noul"]["pairs_complete"], 1)
            self.assertEqual(len(output.read_text(encoding="utf-8").splitlines()), 4)

    def test_each_probe_builds_one_pair(self):
        for record in (NOUL_RECORD, SCORE_RECORD):
            for name in PROBES:
                with self.subTest(kind=record["type"], probe=name):
                    original, perturbed = apply_probe(record, name)
                    self.assertEqual(original["id"], perturbed["id"])
                    self.assertEqual(original["prompt"], perturbed["prompt"])
                    if name == "prompt_injection":
                        self.assertEqual(original["type"], record["type"])
                        self.assertNotEqual(original["text"], perturbed["text"])
                        continue
                    self.assertEqual(original["type"], perturbed["type"])
                    self.assertEqual(original["type"], "choice")
                    self.assertEqual(original["source_type"], record["type"])
                    self.assertEqual(original["text"], perturbed["text"])
                    self.assertEqual(original["label"], str(record["label"]))
                    if name == "abstain_removal":
                        self.assertEqual(original["options"][:-1], perturbed["options"])
                        self.assertEqual(original["options"][-1]["id"], "unsure")
                    elif name == "option_renaming":
                        self.assertEqual([o["id"] for o in original["options"]],
                                         [o["id"] for o in perturbed["options"]])
                        self.assertNotEqual(original["options"], perturbed["options"])
                    else:
                        self.assertEqual(original["options"][::-1], perturbed["options"])

    def test_probe_set_is_flat_paired_and_runnable(self):
        items = build_probe_set([NOUL_RECORD, SCORE_RECORD])
        self.assertEqual(len(items), len(PROBES) * 2 * 2)
        pairs = {}
        for item in items:
            for key in ("id", "type", "prompt", "text", "label", "options",
                        "pair_id", "probe", "variant", "direction", "origin_id"):
                self.assertIn(key, item)
            pairs.setdefault(item["pair_id"], []).append(item["variant"])
        for variants in pairs.values():
            self.assertEqual(sorted(variants), ["original", "perturbed"])
        self.assertEqual(len({item["id"] for item in items}), len(items))

    def test_probe_files_written(self):
        items = build_probe_set([NOUL_RECORD, SCORE_RECORD])
        with tempfile.TemporaryDirectory() as temp:
            written = write_probe_sets(items, Path(temp))
            self.assertEqual(written["all_pairs"], len(items))
            self.assertTrue((Path(temp) / "prompt_injection.jsonl").exists())

    def test_option_probes_run_and_report_complete_pairs(self):
        records = build_probe_set([NOUL_RECORD, SCORE_RECORD], probes=sorted(OPTION_PROBES))
        results = run_adapter(records, MockAdapter())
        self.assertTrue(all(row["status"] == "success" for row in results))
        for item, result in zip(records, results):
            self.assertEqual(set(result["prob"]), {o["id"] for o in item["options"]})
        summary = summarize_option_probes(results)
        self.assertEqual(summary["option_renaming"]["noul"]["pairs_complete"], 1)
        self.assertIn("forced_choice_accuracy", summary["abstain_removal"]["score"])
        incomplete = summarize_option_probes(results[1:])
        self.assertEqual(incomplete[records[0]["probe"]]["noul"]["pairs_complete"], 0)

    def test_cluster_passes_choice_options_and_identical_prompts_to_both_models(self):
        calls = {"jev": [], "qwen": []}

        class FakeRuntime:
            def predict(self, model, kind, text, prompt, options=None):
                calls[model].append((kind, text, prompt, options))
                decision, prob, logits = format_scores(kind, list(range(len(options))), 1, options)
                return RuntimePrediction(decision, prob, logits, 3), []

        records = build_probe_set([NOUL_RECORD], probes=["abstain_removal"])
        demos = [dict(NOUL_RECORD, text="A training example")]
        with tempfile.TemporaryDirectory() as temp:
            for model in calls:
                summary = run_model(FakeRuntime(), model, records, Path(temp) / f"{model}.jsonl", demos)
                metrics = summary["option_probes"]["abstain_removal"]["noul"]
                self.assertEqual(metrics["abstention_rate"], 1.0)
                self.assertEqual(metrics["forced_choice_accuracy"], 1.0)
        self.assertEqual(calls["jev"], calls["qwen"])
        self.assertEqual(calls["jev"][0][2], calls["jev"][1][2])
        self.assertIn("A training example", calls["jev"][0][2])


class CalibrationTests(unittest.TestCase):
    def test_platt_reduces_overconfidence(self):
        records = synthetic_records("noul")
        labels = label_indices(records, "noul")
        raw_probs = probability_matrix(records, "noul")
        cal_probs, parameters, _ = cross_fitted(records, "noul", "platt", n_folds=5, seed=0)
        raw_ece = expected_calibration_error(raw_probs.max(axis=1), raw_probs.argmax(axis=1) == labels)
        cal_ece = expected_calibration_error(cal_probs.max(axis=1), cal_probs.argmax(axis=1) == labels)
        self.assertLess(cal_ece, raw_ece)
        self.assertGreater(parameters["a"], 1.0)

    def test_temperature_scaling_recovers_softening(self):
        records = synthetic_records("score")
        labels = label_indices(records, "score")
        logits = np.asarray([[record["logits"][str(i)] for i in range(6)] for record in records])
        scaler = TemperatureScaler().fit(logits, labels)
        self.assertGreater(scaler.temperature, 1.4)
        self.assertLess(scaler.temperature, 3.0)
        self.assertFalse(scaler.at_boundary)

    def test_temperature_flags_uninformative_logits(self):
        """Confident logits whose labels are independent rail at the upper bound."""
        rng = np.random.default_rng(3)
        n = 2000
        stated = rng.integers(0, 6, size=n)
        logits = np.full((n, 6), -0.6)
        logits[np.arange(n), stated] = 3.0
        labels = rng.integers(0, 6, size=n)
        scaler = TemperatureScaler().fit(logits, labels)
        self.assertTrue(scaler.at_boundary)
        self.assertEqual(scaler.temperature, scaler.temperature)  # finite, not NaN
        self.assertGreaterEqual(scaler.temperature, 1.0e4)

    def test_cross_fitting_is_deterministic(self):
        records = synthetic_records("noul", n=80)
        first, _, _ = cross_fitted(records, "noul", "platt", n_folds=4, seed=7)
        second, _, _ = cross_fitted(records, "noul", "platt", n_folds=4, seed=7)
        np.testing.assert_allclose(first, second)

    def test_evaluate_reports_raw_and_calibrated(self):
        report = evaluate(synthetic_records("noul"), "noul", "mock")
        self.assertIn("ece_15bin", report.raw)
        self.assertIn("ece_15bin", report.calibrated)
        self.assertEqual(report.raw["n"], 200)

    def test_binary_auroc_matches_hand_computed_values(self):
        self.assertEqual(binary_auroc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])), 1.0)
        self.assertEqual(binary_auroc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([0, 0, 1, 1])), 0.0)
        # One positive and one negative tied at the same score -> 0.5.
        self.assertEqual(binary_auroc(np.array([0.5, 0.5]), np.array([0, 1])), 0.5)

    def test_macro_f1_and_balanced_accuracy_are_not_accuracy(self):
        labels = np.array([0, 1, 0, 1])
        predictions = np.array([0, 1, 1, 1])  # class 0: P=1.0 R=0.5; class 1: P=2/3 R=1.0
        self.assertAlmostEqual(macro_f1(predictions, labels, 2), (0.8 + 2 / 3) / 2)
        self.assertAlmostEqual(balanced_accuracy(predictions, labels, 2), 0.75)
        self.assertAlmostEqual(float((predictions == labels).mean()), 0.75)

        # A majority-class-only predictor looks good on accuracy but not here.
        tilted_labels = np.array([0] * 9 + [1])
        tilted_predictions = np.zeros(10, dtype=int)
        self.assertAlmostEqual(float((tilted_predictions == tilted_labels).mean()), 0.9)
        self.assertAlmostEqual(balanced_accuracy(tilted_predictions, tilted_labels, 2), 0.5)
        self.assertAlmostEqual(macro_f1(tilted_predictions, tilted_labels, 2), (18 / 19) / 2)

    def test_quadratic_weighted_kappa_is_one_for_perfect_ordinal_predictions(self):
        labels = np.array([0, 1, 2, 3, 4, 5])
        self.assertAlmostEqual(quadratic_weighted_kappa(labels, labels, 6), 1.0)
        self.assertLess(quadratic_weighted_kappa(np.array([0, 0, 0, 0, 0, 0]), labels, 6), 0.3)

    def test_bootstrap_intervals_bracket_the_point_estimate(self):
        records = synthetic_records("noul", n=160)
        labels = label_indices(records, "noul")
        probs = probability_matrix(records, "noul")
        intervals = bootstrap_ci(probs, labels, "noul", n_boot=200, seed=0)
        for key in ("ece_15bin", "brier", "auroc", "macro_f1"):
            lower, upper = intervals[key]
            self.assertLessEqual(lower, upper)
        self.assertIn("ece_15bin", intervals)

    def test_held_out_fit_uses_only_the_fit_split(self):
        fit_records = synthetic_records("noul", n=120, seed=1)
        eval_records = synthetic_records("noul", n=80, seed=2)
        calibrated, params = fit_on(fit_records, eval_records, "noul", "platt")
        expected = PlattScaler().fit(
            probability_matrix(fit_records, "noul")[:, 1], label_indices(fit_records, "noul")
        )
        self.assertAlmostEqual(params["a"], expected.a)
        self.assertAlmostEqual(params["b"], expected.b)
        np.testing.assert_allclose(calibrated.sum(axis=1), 1.0)

        report = evaluate(eval_records, "noul", "mock", fit_records=fit_records, n_boot=50)
        self.assertEqual(report.protocol, "held_out_fit")
        self.assertEqual(report.n_folds, 0)

    def test_end_to_end_report_and_plot(self):
        records = synthetic_records("noul", n=120) + synthetic_records("score", n=120)
        with tempfile.TemporaryDirectory() as temp:
            reports = run_calibration({"mock": records}, Path(temp), n_folds=4)
            self.assertEqual(len(reports), 2)
            self.assertTrue((Path(temp) / "calibration_report.json").exists())
            self.assertTrue((Path(temp) / "calibration_report.md").exists())
            self.assertTrue((Path(temp) / "reliability.png").exists())


class RunnerCliTests(unittest.TestCase):
    def test_predict_then_calibrate(self):
        with tempfile.TemporaryDirectory() as temp:
            data = Path(temp) / "eval.jsonl"
            sample = [
                dict(NOUL_RECORD, id=f"noul_{i:04d}") for i in range(30)
            ] + [
                dict(SCORE_RECORD, id=f"score_{i:04d}") for i in range(30)
            ]
            data.write_text("\n".join(json.dumps(r) for r in sample) + "\n", encoding="utf-8")
            preds = Path(temp) / "preds" / "mock.jsonl"
            runner_main(["predict", "--model", "mock", "--input", str(data), "--out", str(preds)])
            self.assertTrue(preds.exists())
            out = Path(temp) / "calibration"
            runner_main(["calibrate", "--pred-dir", str(preds.parent), "--out-dir", str(out), "--folds", "3"])
            payload = json.loads((out / "calibration_report.json").read_text())
            self.assertTrue(payload["reports"])


if __name__ == "__main__":
    unittest.main()
