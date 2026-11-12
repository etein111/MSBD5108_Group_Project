import math
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

DecisionType = Literal["noul", "score", "choice"]
ModelName = Literal["jev", "qwen"]
ChoiceOptions = Sequence[Mapping[str, str]]
CHOICE_LETTERS = tuple("ABCDEFGHIJKLMNOP")

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


@dataclass(frozen=True, init=False)
class Prediction:
    decision: str | int
    prob: dict[str, float]
    logits: dict[str, float]
    latency_ms: float

    def __init__(
        self, decision: str | int, prob: dict[str, float], logits: dict[str, float],
        latency_ms: float | None = None, *, latency: float | None = None,
    ):
        if latency_ms is None:
            latency_ms = latency
        elif latency is not None and latency != latency_ms:
            raise TypeError("latency and latency_ms must agree")
        if latency_ms is None:
            raise TypeError("Prediction requires latency_ms (or legacy latency)")
        object.__setattr__(self, "decision", decision)
        object.__setattr__(self, "prob", prob)
        object.__setattr__(self, "logits", logits)
        object.__setattr__(self, "latency_ms", latency_ms)

    @property
    def latency(self) -> float:
        """Legacy milliseconds alias for in-process callers."""
        return self.latency_ms


def decision_labels(kind: DecisionType, options: ChoiceOptions | None = None) -> tuple[str, ...]:
    if kind != "choice":
        if options is not None:
            raise ValueError("Custom options are only supported for choice.")
        return OUTPUT_LABELS[kind]
    if options is None or not 2 <= len(options) <= len(CHOICE_LETTERS):
        raise ValueError("Choice requires 2 to 16 options.")
    labels = []
    for option in options:
        if not isinstance(option, Mapping) or option.keys() != {"id", "text"}:
            raise ValueError("Each choice option requires exactly id and text.")
        for value in option.values():
            if not isinstance(value, str) or not value.strip() or value.splitlines() != [value]:
                raise ValueError("Choice option id and text must be nonblank single-line strings.")
        if option["id"] != option["id"].strip():
            raise ValueError("Choice option IDs cannot have surrounding whitespace.")
        labels.append(option["id"])
    if len(set(labels)) != len(labels):
        raise ValueError("Choice option IDs must be unique.")
    return tuple(labels)


def render_prompt(
    kind: DecisionType, text: str, question: str, options: ChoiceOptions | None = None
) -> str:
    decision_labels(kind, options)
    if options is None:
        option_lines = NATIVE_OPTIONS[kind]
    else:
        option_lines = [f"{letter}) {option['text']}" for letter, option in zip(CHOICE_LETTERS, options)]
    rendered_options = "\n".join(option_lines)
    return (
        f"[kind] {kind}\n[state] {text}\n[question] {question}\n"
        f"[options]\n{rendered_options}\n[decision]:"
    )


def validate_token_count(count: int, limit: int) -> None:
    if count > limit:
        raise InputTooLong(count, limit)


def format_scores(
    kind: DecisionType, scores: Sequence[float], temperature: float, options: ChoiceOptions | None = None
) -> tuple[str | int, dict[str, float], dict[str, float]]:
    labels = decision_labels(kind, options)
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
