import math
from dataclasses import dataclass
from typing import Literal, Sequence

DecisionType = Literal["noul", "score"]
ModelName = Literal["jev", "qwen"]

NATIVE_OPTIONS: dict[DecisionType, tuple[str, ...]] = {
    "noul": ("false", "true"),
    "score": tuple(str(level) for level in range(6)),
}
OUTPUT_LABELS: dict[DecisionType, tuple[str, ...]] = {
    "noul": ("no", "yes"),
    "score": NATIVE_OPTIONS["score"],
}


class InputTooLong(ValueError):
    def __init__(self, count: int, limit: int):
        super().__init__(f"Rendered prompt uses {count} tokens; limit is {limit}.")


class InferenceOutOfMemory(RuntimeError):
    pass


@dataclass
class Prediction:
    decision: str | int
    prob: dict[str, float]
    logits: dict[str, float]
    latency: float


def render_prompt(kind: DecisionType, text: str, question: str) -> str:
    options = "\n".join(NATIVE_OPTIONS[kind])
    return (
        f"[kind] {kind}\n[state] {text}\n[question] {question}\n"
        f"[options]\n{options}\n[decision]:"
    )


def validate_token_count(count: int, limit: int) -> None:
    if count > limit:
        raise InputTooLong(count, limit)


def format_scores(
    kind: DecisionType, scores: Sequence[float], temperature: float
) -> tuple[str | int, dict[str, float], dict[str, float]]:
    labels = OUTPUT_LABELS[kind]
    if len(scores) != len(labels) or not all(math.isfinite(score) for score in scores):
        raise ValueError("Expected one finite logit per option.")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Expected a finite, positive temperature.")

    winner = max(range(len(labels)), key=scores.__getitem__)
    weights = [math.exp((score - scores[winner]) / temperature) for score in scores]
    total = math.fsum(weights)
    probabilities = dict(zip(labels, (weight / total for weight in weights)))
    decision = int(labels[winner]) if kind == "score" else labels[winner]
    return decision, probabilities, dict(zip(labels, scores))
