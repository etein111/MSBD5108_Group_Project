import argparse
import hashlib
import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, ConfigDict, field_validator

from .protocol import DecisionType, InferenceOutOfMemory, InputTooLong, ModelName, Prediction
from .runtime import DEFAULT_CACHE_DIR, DEFAULT_MAX_INPUT_TOKENS, Runtime


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    type: DecisionType
    text: str
    prompt: str

    @field_validator("text", "prompt")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Must contain non-whitespace text.")
        return value


def create_app(load_runtime, *, audit_inputs: bool = False) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runtime = load_runtime()
        yield

    app = FastAPI(title="MSBD5108 local model service", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return app.state.runtime.health()

    @app.post("/v1/{model}/predict")
    def predict(model: ModelName, body: PredictRequest, response: Response) -> Prediction:
        try:
            prediction, token_ids = app.state.runtime.predict(
                model, body.type, body.text, body.prompt
            )
        except InputTooLong as error:
            raise HTTPException(status_code=413, detail=str(error)) from error
        except InferenceOutOfMemory as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        if audit_inputs:
            encoded = json.dumps(token_ids, separators=(",", ":")).encode("ascii")
            response.headers["X-Input-Token-SHA256"] = hashlib.sha256(encoded).hexdigest()
            response.headers["X-Input-Token-Count"] = str(len(token_ids))
        return prediction

    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Serve both benchmark models on localhost.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--max-input-tokens", type=int, default=DEFAULT_MAX_INPUT_TOKENS)
    parser.add_argument("--audit-inputs", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.max_input_tokens <= DEFAULT_MAX_INPUT_TOKENS:
        parser.error(f"--max-input-tokens must be between 1 and {DEFAULT_MAX_INPUT_TOKENS}")
    app = create_app(
        lambda: Runtime(args.cache_dir, args.max_input_tokens),
        audit_inputs=args.audit_inputs,
    )
    uvicorn.run(app, host="127.0.0.1", port=args.port, workers=1)


if __name__ == "__main__":
    main()
