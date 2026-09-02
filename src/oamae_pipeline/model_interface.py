from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


@dataclass
class PredictionBundle:
    probability: np.ndarray
    diagnostics: dict[str, np.ndarray]


class ModelAdapter:
    def __init__(self, callable_path: str, checkpoint: Path, config: dict[str, Any]):
        module_name, function_name = callable_path.rsplit(":", 1)
        module = importlib.import_module(module_name)
        self.function: Callable[..., Any] = getattr(module, function_name)
        self.checkpoint = checkpoint
        self.config = config

    def predict(self, batch: dict[str, np.ndarray]) -> PredictionBundle:
        result = self.function(batch=batch, checkpoint=str(self.checkpoint), config=self.config)
        if not isinstance(result, dict) or "probability" not in result:
            raise TypeError("Adapter must return a mapping with probability")
        probability = np.asarray(result["probability"], dtype=np.float32)
        diagnostics = {key: np.asarray(value) for key, value in result.items() if key != "probability"}
        if not np.isfinite(probability).all() or probability.min() < 0 or probability.max() > 1:
            raise ValueError("Probability output must be finite and within 0 to 1")
        return PredictionBundle(probability=probability, diagnostics=diagnostics)
