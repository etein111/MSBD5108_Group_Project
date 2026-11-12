"""Block 3 — model adapters (T6/T7), calibration (T9) and adversarial probes (T10)."""

from .adapters import HttpAdapter, JevAdapter, MockAdapter, Prediction, QwenAdapter
from .calibration import (
    CalibrationReport,
    PlattScaler,
    TemperatureScaler,
    evaluate,
    run_calibration,
)
from .probes import PROBES, apply_probe, build_probe_set
from .runner import run_adapter

__all__ = [
    "HttpAdapter", "JevAdapter", "MockAdapter", "Prediction", "QwenAdapter",
    "CalibrationReport", "PlattScaler", "TemperatureScaler", "evaluate", "run_calibration",
    "PROBES", "apply_probe", "build_probe_set",
    "run_adapter",
]
