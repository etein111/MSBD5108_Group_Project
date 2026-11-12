"""Offline tests for Block 3 — no GPU and no running service required."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from block3.adapters import (
    MockAdapter,
    Prediction,
    augment_with_shots,
    load_demonstrations,
    normalize_decision,
)
from block3.calibration import (
    PlattScaler,
    TemperatureScaler,
    cross_fitted,
    evaluate,
    expected_calibration_error,
    label_indices,
    probability_matrix,
    run_calibration,
)
from block3.probes import PROBES, apply_probe, build_probe_set, write_probe_sets
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
    def test_harness_projection(self):
        prediction = Prediction("yes", {"no": 0.3, "yes": 0.7}, {"no": -1.0, "yes": 1.0}, 250.0)
        decision, confidence, logits, latency_seconds = prediction.as_harness_tuple()
        self.assertEqual(decision, "yes")
        self.assertAlmostEqual(confidence, 0.7)
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
            self.assertIsInstance(record["probability"], float)
            self.assertTrue(0.0 <= record["probability"] <= 1.0)
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

    def test_runner_produces_calibratable_records(self):
        results = run_adapter([NOUL_RECORD, SCORE_RECORD], MockAdapter())
        self.assertEqual([record["status"] for record in results], ["success", "success"])
        self.assertAlmostEqual(sum(results[0]["prob"].values()), 1.0)
        self.assertIn("ground_truth", results[0])


class ProbeTests(unittest.TestCase):
    def test_each_probe_builds_one_pair(self):
        for name in PROBES:
            record = NOUL_RECORD if name != "score_only" else SCORE_RECORD
            original, perturbed = apply_probe(record, name)
            self.assertEqual(original["id"] == perturbed["id"], True)
            if name == "prompt_injection":
                self.assertIn("Ignore all previous instructions", perturbed["text"])
                self.assertEqual(original["text"], record["text"])
            elif name == "abstain_removal":
                self.assertIn("unsure", original["prompt"])
                self.assertNotIn("unsure", perturbed["prompt"])
            elif name == "option_renaming":
                self.assertIn("unsafe", perturbed["prompt"])
            elif name == "option_order_permutation":
                self.assertIn("yes, no", perturbed["prompt"])

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
