import json
import os
import platform
import subprocess
import time
from importlib.metadata import version
from pathlib import Path
from threading import Lock

from .protocol import (
    CHOICE_LETTERS,
    NATIVE_OPTIONS,
    ChoiceOptions,
    DecisionType,
    InferenceOutOfMemory,
    ModelName,
    Prediction,
    format_scores,
    render_prompt,
    validate_token_count,
)

REPO = "autotrust/JEV-9B"
REVISION = "7ac669b8fb476ba926491e3d1c39f9330d68dbed"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / ".cache" / "huggingface"
DEFAULT_MAX_INPUT_TOKENS = 4096
TEXT_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "model.safetensors.index.json",
    *(f"model-{shard:05}-of-00005.safetensors" for shard in range(1, 6)),
    "adapter/adapter_config.json",
    "adapter/adapter_model.safetensors",
    "head.safetensors",
    "judge_config.json",
    "calibration.json",
)


def resolve_snapshot(cache_dir: Path, *, local_only: bool = False) -> Path:
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    options = dict(revision=REVISION, allow_patterns=TEXT_FILES, cache_dir=cache_dir / "hub")
    try:
        cached = Path(snapshot_download(REPO, **options, local_files_only=True))
        if all((cached / filename).is_file() for filename in TEXT_FILES):
            return cached
    except LocalEntryNotFoundError:
        if local_only:
            raise
    if local_only:
        raise FileNotFoundError(f"Incomplete JEV snapshot in {cache_dir}")
    return Path(snapshot_download(REPO, **options, max_workers=3))


class Runtime:
    def __init__(self, cache_dir: Path, max_input_tokens: int):
        os.environ.setdefault("HF_HOME", str(cache_dir.resolve()))
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "90")

        import torch
        from peft import PeftModel
        from safetensors.torch import load_file
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("A CUDA GPU with BF16 support is required.")
        self.torch = torch
        self.device = torch.device("cuda:0")
        self.max_input_tokens = max_input_tokens
        self.lock = Lock()
        self.driver_version = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).splitlines()[0].strip()
        torch.cuda.reset_peak_memory_stats(self.device)
        snapshot = resolve_snapshot(cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(snapshot)
        self.base = AutoModelForCausalLM.from_pretrained(
            snapshot,
            device_map={"": 0},
            dtype=torch.bfloat16,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
        )
        self.model = PeftModel.from_pretrained(
            self.base, snapshot / "adapter", is_trainable=False
        ).eval()
        head = load_file(snapshot / "head.safetensors")
        self.head_weight = head["proj.weight"].to(self.device, dtype=torch.float32)
        self.head_bias = head["proj.bias"].to(self.device, dtype=torch.float32)
        config = json.loads((snapshot / "judge_config.json").read_text(encoding="utf-8"))
        verbalizers = {**NATIVE_OPTIONS, "choice": CHOICE_LETTERS}
        self.slots = {kind: slice(*config["slots"]["ranges"][kind]) for kind in verbalizers}
        self.verbalizer_ids = {}
        for kind, options in verbalizers.items():
            token_ids = config["verbalizer_ids"][self.slots[kind]]
            if [self.tokenizer.encode(option, add_special_tokens=False) for option in options] != [
                [token_id] for token_id in token_ids
            ]:
                raise RuntimeError(f"Unexpected JEV tokenizer or verbalizer IDs for {kind}.")
            self.verbalizer_ids[kind] = torch.tensor(token_ids, device=self.device)
        calibration = json.loads((snapshot / "calibration.json").read_text(encoding="utf-8"))
        self.temperatures = calibration["per_kind"]
        self.packages = {
            name: version(name)
            for name in (
                "torch", "transformers", "peft", "bitsandbytes", "accelerate",
                "huggingface-hub", "safetensors", "fastapi", "uvicorn",
            )
        }

    def predict(
        self, model: ModelName, kind: DecisionType, text: str, prompt: str,
        options: ChoiceOptions | None = None,
    ) -> tuple[Prediction, list[int]]:
        with self.lock, self.torch.inference_mode():
            started = time.perf_counter()
            token_ids = self.tokenizer.encode(
                render_prompt(kind, text, prompt, options), add_special_tokens=False,
                truncation=False, verbose=False,
            )
            validate_token_count(len(token_ids), self.max_input_tokens)
            try:
                input_ids = self.torch.tensor([token_ids], device=self.device)
                if model == "jev":
                    hidden = self.base.model(input_ids=input_ids, use_cache=False)
                    last = hidden.last_hidden_state[0, -1].float()
                    slots = self.slots[kind]
                    if options is not None:
                        slots = slice(slots.start, slots.start + len(options))
                    scores = self.head_weight[slots] @ last + self.head_bias[slots]
                    temperature = self.temperatures[kind]
                else:
                    with self.model.disable_adapter():
                        output = self.base(
                            input_ids=input_ids, use_cache=False, logits_to_keep=1
                        )
                    ids = self.verbalizer_ids[kind]
                    if options is not None:
                        ids = ids[:len(options)]
                    scores = output.logits[0, -1].index_select(0, ids)
                    temperature = 1.0
                # Copying to CPU synchronizes the GPU work before we stop the timer.
                decision, prob, logits = format_scores(kind, scores.float().tolist(), temperature, options)
            except self.torch.cuda.OutOfMemoryError:
                raise InferenceOutOfMemory(
                    "GPU ran out of memory. Free VRAM or lower the shared input limit."
                ) from None
            latency_ms = (time.perf_counter() - started) * 1000
            return Prediction(decision, prob, logits, latency_ms), token_ids

    def health(self) -> dict:
        cuda = self.torch.cuda
        return {
            "status": "ready",
            "models": {"jev": REPO, "qwen": "Qwen/Qwen3.5-9B (shared text backbone)"},
            "revision": REVISION,
            "quantization": "NF4, double quantization, BF16 compute",
            "gpu": cuda.get_device_name(self.device),
            "driver_version": self.driver_version,
            "cuda": self.torch.version.cuda,
            "python": platform.python_version(),
            "packages": self.packages,
            "max_input_tokens": self.max_input_tokens,
            "peak_allocated_mib": round(cuda.max_memory_allocated(self.device) / 2**20, 2),
            "peak_reserved_mib": round(cuda.max_memory_reserved(self.device) / 2**20, 2),
        }
