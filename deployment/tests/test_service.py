import math
import unittest

from fastapi.testclient import TestClient

from deployment.protocol import (
    Prediction,
    format_scores,
    render_prompt,
    validate_token_count,
)
from deployment.server import create_app


class FakeRuntime:
    def health(self):
        return {"status": "ready"}

    def predict(self, model, kind, text, prompt, options=None):
        validate_token_count(len(text), 10)
        scores = [-2, 0] if kind == "noul" else [-6, 0, 6, 1, 2, 3]
        if kind == "choice":
            scores = list(range(len(options)))
        decision, prob, logits = format_scores(kind, scores, 2 if model == "jev" else 1, options)
        return Prediction(decision, prob, logits, 12.3), [1, 2, 3]


class ServiceTests(unittest.TestCase):
    def test_prediction_accepts_both_latency_keywords(self):
        for kwargs in ({"latency": 0.0}, {"latency_ms": 0.0}, {"latency": 0.0, "latency_ms": 0.0}):
            prediction = Prediction("yes", {"yes": 1.0}, {"yes": 0.0}, **kwargs)
            self.assertEqual(prediction.latency_ms, 0.0)
            self.assertEqual(prediction.latency, 0.0)
        with self.assertRaisesRegex(TypeError, "must agree"):
            Prediction("yes", {}, {}, latency=1.0, latency_ms=2.0)
        with self.assertRaisesRegex(TypeError, "requires latency_ms"):
            Prediction("yes", {}, {})

    def test_native_template_preserves_text(self):
        self.assertEqual(
            render_prompt("noul", " raw\ninput ", "Is it spam?"),
            "[kind] noul\n[state]  raw\ninput \n[question] Is it spam?\n"
            "[options]\nfalse\ntrue\n[decision]:",
        )

    def test_binary_raw_logits_and_temperature(self):
        decision, prob, logits = format_scores("noul", [1000.0, 1002.0], 2)
        self.assertEqual(decision, "yes")
        self.assertEqual(logits, {"no": 1000.0, "yes": 1002.0})
        self.assertAlmostEqual(prob["yes"], 1 / (1 + math.exp(-1)))
        self.assertAlmostEqual(sum(prob.values()), 1)

    def test_score_result_is_an_int(self):
        decision, prob, _ = format_scores("score", [0, 0, 0, 0, 4, 0], 1)
        self.assertEqual(decision, 4)
        self.assertEqual(set(prob), set("012345"))
        self.assertAlmostEqual(prob["4"], math.exp(4) / (math.exp(4) + 5))

    def test_same_response_schema_for_both_paths(self):
        with TestClient(create_app(FakeRuntime, audit_inputs=True)) as client:
            self.assertEqual(client.get("/health").json(), {"status": "ready"})
            fingerprints = []
            for model in ("jev", "qwen"):
                for kind, expected in (("noul", "yes"), ("score", 2)):
                    response = client.post(
                        f"/v1/{model}/predict",
                        json={"type": kind, "text": "message", "prompt": "Judge this."},
                    )
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(set(response.json()), {"decision", "prob", "logits", "latency", "latency_ms"})
                    self.assertEqual(response.json()["latency_ms"], 12.3)
                    self.assertEqual(response.json()["latency"], response.json()["latency_ms"])
                    self.assertEqual(response.json()["decision"], expected)
                    fingerprints.append(response.headers["X-Input-Token-SHA256"])
            self.assertEqual(len(set(fingerprints)), 1)

    def test_bad_input_and_token_limit(self):
        with TestClient(create_app(FakeRuntime)) as client:
            endpoint = "/v1/qwen/predict"
            payload = {"type": "noul", "text": "a" * 10, "prompt": "Question?"}
            at_limit = client.post(endpoint, json=payload)
            self.assertEqual(at_limit.status_code, 200)
            self.assertNotIn("X-Input-Token-SHA256", at_limit.headers)
            response = client.post(endpoint, json={**payload, "text": "a" * 11})
            self.assertEqual(response.status_code, 413)
            self.assertIn("limit is 10", response.json()["detail"])
            for invalid in ({"type": "choice"}, {"prompt": " "}, {"options": ["a", "b"]}):
                self.assertEqual(client.post(endpoint, json={**payload, **invalid}).status_code, 422)

    def test_choice_slots_follow_option_order_and_return_stable_ids(self):
        options = [{"id": "reject", "text": "Negative"}, {"id": "accept", "text": "Positive"}]
        self.assertIn("\n[options]\nA) Negative\nB) Positive\n", render_prompt("choice", "Text", "Question", options))
        payload = {"type": "choice", "text": "message", "prompt": "Judge this."}
        with TestClient(create_app(FakeRuntime)) as client:
            for model in ("jev", "qwen"):
                for ordered in (options, options[::-1], options + [{"id": "unsure", "text": "Unsure"}]):
                    response = client.post(f"/v1/{model}/predict", json={**payload, "options": ordered})
                    self.assertEqual(response.status_code, 200, response.text)
                    result = response.json()
                    self.assertEqual(result["decision"], ordered[-1]["id"])
                    self.assertEqual(result["logits"], {o["id"]: i for i, o in enumerate(ordered)})
                    self.assertAlmostEqual(sum(result["prob"].values()), 1)

    def test_choice_rejects_invalid_options(self):
        valid = [{"id": str(i), "text": str(i)} for i in range(16)]
        payload = {"type": "choice", "text": "message", "prompt": "Judge this."}
        with TestClient(create_app(FakeRuntime)) as client:
            endpoint = "/v1/jev/predict"
            self.assertEqual(client.post(endpoint, json={**payload, "options": valid}).status_code, 200)
            for options in (valid[:1], valid + [{"id": "extra", "text": "Extra"}],
                            [valid[0], valid[0]], [valid[0], {"id": "x", "text": "one\ntwo"}],
                            [valid[0], {"id": "x", "text": " "}], [valid[0], {"id": "x"}]):
                self.assertEqual(client.post(endpoint, json={**payload, "options": options}).status_code, 422)
            self.assertEqual(client.post(endpoint, json={**payload, "type": "noul", "options": valid}).status_code, 422)
