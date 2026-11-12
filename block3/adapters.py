"""Block 3 — model adapters (T6, T7).

Bridge the local deployment service (``deployment/server.py``) to the evaluation
harness. Both adapters POST to ``/v1/{model}/predict`` with ``{type, text, prompt}``
and speak two interfaces:

* ``predict(prompt, text, task_type, options) -> (decision, confidence, logits, latency_seconds)``
  — the ``evaluation.harness.ModelAdapter`` contract, so an adapter can be passed
  straight into ``JevBenchHarness``.
* ``predict_full(...) -> Prediction`` — the richer form carrying the full label
  distribution and latency in milliseconds, used by the calibration pipeline (T9).

Schema alignment rule (the one place the two teammate modules disagree):

======================  ============================  =============================
field                   deployment (``Prediction``)   harness (``ModelAdapter``)
======================  ============================  =============================
decision                ``"no"/"yes"`` or ``int``      same
probability             ``prob: dict[label, float]``   scalar ``P(decision)``
logits                  ``logits: dict``               same
latency                 milliseconds                   seconds
======================  ============================  =============================

``Prediction.as_harness_tuple`` performs exactly that projection, so the full
distribution is never lost when it matters.
"""

from __future__ import annotations

import hashlib
import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

OUTPUT_LABELS: Dict[str, tuple] = {
    "noul": ("no", "yes"),
    "score": tuple(str(level) for level in range(6)),
}


def normalize_decision(task_type: str, decision: Any) -> Any:
    """Coerce the service's decision into the label space used by the datasets."""
    if task_type == "score":
        value = int(decision)
        if not 0 <= value <= 5:
            raise ValueError(f"score decision out of range: {decision!r}")
        return value
    value = str(decision).strip().lower()
    if value not in OUTPUT_LABELS["noul"]:
        raise ValueError(f"noul decision must be 'no' or 'yes', got {decision!r}")
    return value


@dataclass
class Prediction:
    decision: Any
    prob: Dict[str, float]
    logits: Dict[str, float]
    latency: float  # milliseconds, as reported by the deployment service
    model: str = ""

    @classmethod
    def from_payload(cls, obj: Mapping[str, Any], task_type: str, model: str = "") -> "Prediction":
        return cls(
            decision=normalize_decision(task_type, obj["decision"]),
            prob={label: float(value) for label, value in obj["prob"].items()},
            logits={label: float(value) for label, value in obj["logits"].items()},
            latency=float(obj["latency"]),
            model=model,
        )

    @property
    def confidence(self) -> float:
        """Probability assigned to the chosen decision."""
        return float(self.prob[str(self.decision)])

    def as_harness_tuple(self) -> tuple:
        """``(decision, P(decision), logits, latency_seconds)``."""
        return self.decision, self.confidence, dict(self.logits), self.latency / 1000.0


class HttpAdapter:
    """Thin HTTP client over the deployment service."""

    model = ""

    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: float = 300.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def predict_full(
        self,
        prompt: str,
        text: str,
        task_type: str,
        options: Optional[Sequence[Any]] = None,
    ) -> Prediction:
        # The service derives the native option set from ``type``. ``options`` is
        # accepted only so this method is drop-in compatible with the harness;
        # renaming/reordering is exercised through the prompt instead (T10).
        payload = {"type": task_type, "text": text, "prompt": prompt}
        request = urllib.request.Request(
            f"{self.base_url}/v1/{self.model}/predict",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"[{self.model}] HTTP {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            raise ConnectionError(
                f"[{self.model}] cannot reach {self.base_url} — start it with "
                f"`python -m deployment.server` ({error.reason})"
            ) from error
        return Prediction.from_payload(body, task_type, self.model)

    def predict(
        self,
        prompt: str,
        text: str,
        task_type: str,
        options: Optional[Sequence[Any]] = None,
    ) -> tuple:
        return self.predict_full(prompt, text, task_type, options).as_harness_tuple()


class JevAdapter(HttpAdapter):
    """T6: JEV-9B adapter — decision head plus the model's native probabilities."""

    model = "jev"


class QwenAdapter(HttpAdapter):
    """T7: Qwen3.5-9B adapter — few-shot constrained decoding, logit probabilities.

    Constrained decoding lives in the service (the readout is restricted to the
    native option tokens). The few-shot half lives here: demonstrations from the
    *train* split are prepended to the question, never to JEV's prompt, so the
    only difference between the two models is the treatment the protocol assigns
    to the general-purpose baseline.
    """

    model = "qwen"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        timeout: float = 300.0,
        demonstrations: Optional[Sequence[Mapping[str, Any]]] = None,
    ):
        super().__init__(base_url, timeout)
        self.demonstrations = list(demonstrations or [])

    def predict_full(
        self,
        prompt: str,
        text: str,
        task_type: str,
        options: Optional[Sequence[Any]] = None,
    ) -> Prediction:
        return super().predict_full(augment_with_shots(prompt, task_type, self.demonstrations), text, task_type, options)


def native_label(task_type: str, label: Any) -> str:
    """Map dataset labels onto the tokens the service constrains decoding to."""
    if task_type == "noul":
        return "true" if str(label).strip().lower() == "yes" else "false"
    return str(int(label))


def augment_with_shots(
    prompt: str, task_type: str, demonstrations: Sequence[Mapping[str, Any]]
) -> str:
    """Prepend same-type demonstrations to the question; empty list is a no-op."""
    demos = [demo for demo in demonstrations if demo.get("type") == task_type]
    if not demos:
        return prompt
    lines = ["Examples (text -> answer):"]
    for demo in demos:
        lines.append(f"- {json.dumps(str(demo['text']))} -> {native_label(task_type, demo['label'])}")
    lines.extend(["", "Now answer the actual question."])
    lines.append(prompt)
    return "\n".join(lines)


def load_demonstrations(path: Path, per_type: int = 4) -> list:
    """Take the first ``per_type`` train-split records of each decision type."""
    with Path(path).open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    chosen: Dict[str, list] = {}
    for record in records:
        bucket = chosen.setdefault(record["type"], [])
        if len(bucket) < per_type:
            bucket.append({"type": record["type"], "text": record["text"], "label": record["label"]})
    return [demo for bucket in chosen.values() for demo in bucket]


class MockAdapter:
    """Deterministic offline adapter for tests and dry runs (no GPU, no server).

    Emits the same :class:`Prediction` schema as the HTTP adapters, so the
    harness, calibration and probe code can be exercised end to end.
    """

    model = "mock"

    def __init__(self, model: str = "mock", seed: int = 0):
        self.model = model
        self.seed = seed

    def predict_full(
        self,
        prompt: str,
        text: str,
        task_type: str,
        options: Optional[Sequence[Any]] = None,
    ) -> Prediction:
        digest = hashlib.sha256(
            f"{self.model}|{self.seed}|{task_type}|{text}".encode("utf-8")
        ).digest()
        labels = OUTPUT_LABELS[task_type]
        if task_type == "noul":
            decision: Any = "yes" if digest[0] & 1 else "no"
        else:
            decision = digest[0] % 6
        strength = 2.0 + (digest[1] / 255.0) * 3.0
        logits = {
            label: strength if str(decision) == label else -(digest[2 + i] / 255.0)
            for i, label in enumerate(labels)
        }
        return Prediction(
            decision=decision,
            prob=_softmax(logits),
            logits=logits,
            latency=50.0,
            model=self.model,
        )

    def predict(
        self,
        prompt: str,
        text: str,
        task_type: str,
        options: Optional[Sequence[Any]] = None,
    ) -> tuple:
        return self.predict_full(prompt, text, task_type, options).as_harness_tuple()


def _softmax(logits: Mapping[str, float]) -> Dict[str, float]:
    values = list(logits.values())
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = math.fsum(weights)
    return {label: weight / total for label, weight in zip(logits, weights)}
