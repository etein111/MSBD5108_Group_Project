"""Check the evaluation environment without downloads, GPU, or repository outputs."""

import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from evaluation.data_pipeline import create_jevbench_sample
from evaluation.metrics import JevBenchMetrics


class EvaluationEnvironmentTests(unittest.TestCase):
    def test_data_metrics_and_plot_without_downloads(self):
        records = []
        for kind, labels in (("noul", ("no", "yes")), ("score", (0, 5))):
            for index, label in enumerate(labels):
                sample = create_jevbench_sample(f"{kind}-{index}", kind, "fixture", label)
                records.append({
                    "type": sample["type"], "ground_truth": sample["label"],
                    "prediction": label, "probability": 0.9, "status": "success",
                })
        evaluator = JevBenchMetrics(records)
        metrics = evaluator.compute_all_metrics()
        self.assertEqual(metrics["noul"]["exact_match"], 1.0)
        self.assertAlmostEqual(metrics["noul"]["brier_score"], 0.01)
        self.assertEqual(metrics["score"]["mae"], 0.0)
        self.assertEqual(metrics["score"]["qwk"], 1.0)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "metrics.png"
            evaluator.plot_comprehensive_dashboard(str(output))
            self.assertGreater(output.stat().st_size, 0)
