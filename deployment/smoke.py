"""Run real HTTP acceptance checks and print a JSON report."""

import argparse
import json
import math
import urllib.error
import urllib.request
from pathlib import Path

from .protocol import decision_labels, format_scores, render_prompt
from .runtime import DEFAULT_CACHE_DIR, REVISION, resolve_snapshot

MODELS = ("jev", "qwen")
LOCAL_HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))
EXAMPLES = (
    {
        "type": "noul",
        "text": "Congratulations! You won a $500 gift card. Click this link to claim it now.",
        "prompt": "Is this message spam?",
    },
    {
        "type": "score",
        "text": "Your idea is stupid, and you're a complete idiot.",
        "prompt": "Rate the toxicity from 0 (none) to 5 (severe abuse or threats).",
    },
)
CHOICE_OPTIONS = [{"id": "no", "text": "No"}, {"id": "yes", "text": "Yes"}]
CHOICE_EXAMPLES = tuple(
    {**EXAMPLES[0], "type": "choice", "options": options}
    for options in (
        CHOICE_OPTIONS,
        CHOICE_OPTIONS[::-1],
        CHOICE_OPTIONS + [{"id": "unsure", "text": "Unsure"}],
    )
)


def request_json(port, path, body=None):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data, {"Content-Type": "application/json"}
    )
    with LOCAL_HTTP.open(request, timeout=180) as response:
        return json.load(response), response.headers


def near_limit_payload(cache_dir, limit):
    from transformers import AutoTokenizer

    snapshot = resolve_snapshot(cache_dir, local_only=True)
    tokenizer = AutoTokenizer.from_pretrained(snapshot)
    sample = dict(EXAMPLES[0])
    low, high = 1, limit * 2
    while low <= high:
        words = (low + high) // 2
        sample["text"] = (" text" * words).lstrip()
        count = len(tokenizer.encode(
            render_prompt(sample["type"], sample["text"], sample["prompt"]),
            add_special_tokens=False, verbose=False,
        ))
        if count == limit:
            return sample
        if count < limit:
            low = words + 1
        else:
            high = words - 1
    raise AssertionError(f"Could not construct the {limit}-token fixture")


def check_pair(port, payload, temperatures):
    kind = payload["type"]
    options = payload.get("options")
    labels = decision_labels(kind, options)
    results = {}
    fingerprints = set()
    counts = set()
    for model in MODELS:
        response, headers = request_json(port, f"/v1/{model}/predict", payload)
        fingerprints.add(headers["X-Input-Token-SHA256"])
        counts.add(int(headers["X-Input-Token-Count"]))
        assert set(response["prob"]) == set(response["logits"]) == set(labels), response
        assert math.isclose(math.fsum(response["prob"].values()), 1.0), response
        assert math.isfinite(response["latency_ms"]) and response["latency_ms"] > 0, response
        assert response["latency"] == response["latency_ms"], response
        temperature = temperatures[kind] if model == "jev" else 1.0
        decision, prob, _ = format_scores(
            kind, [response["logits"][label] for label in labels], temperature, options
        )
        assert type(response["decision"]) is type(decision) and response["decision"] == decision
        for label in labels:
            assert math.isclose(response["prob"][label], prob[label], abs_tol=1e-12), response
        results[model] = response
    assert len(fingerprints) == len(counts) == 1, "Input tokens differ by model"
    return {"input_tokens": counts.pop(), "input_sha256": fingerprints.pop(), "models": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    args = parser.parse_args()
    health, _ = request_json(args.port, "/health")
    assert health["status"] == "ready" and health["revision"] == REVISION, health
    # This fixture pins the temperatures of the deployed release independently of the server.
    temperatures = {
        "noul": 1.0022178236691426, "score": 1.012184473350984, "choice": 0.9840233476014968,
    }
    cases = EXAMPLES + CHOICE_EXAMPLES
    for example in cases:
        check_pair(args.port, example, temperatures)
    examples = [
        {"request": example, **check_pair(args.port, example, temperatures)} for example in cases
    ]

    limit = health["max_input_tokens"]
    payload = near_limit_payload(args.cache_dir, limit)
    boundary = check_pair(args.port, payload, temperatures)
    assert boundary["input_tokens"] == limit, boundary
    payload["text"] += " text"
    for model in MODELS:
        try:
            request_json(args.port, f"/v1/{model}/predict", payload)
        except urllib.error.HTTPError as error:
            assert error.code == 413, (error.code, error.read())
        else:
            raise AssertionError(f"{model} accepted an over-limit request")
    health, _ = request_json(args.port, "/health")
    print(json.dumps({
        "health": health, "examples": examples, "boundary": boundary,
        "over_limit_status": 413,
    }, indent=2))


if __name__ == "__main__":
    main()
