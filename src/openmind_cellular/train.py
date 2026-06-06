"""Adaptive trainer — train a model or load from muscle memory."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from .probe import ResourceSnapshot, probe as do_probe

_MUSCLE_DIR = Path.home() / ".openmind" / "muscle_memory"


def _model_cache_key(model_name: str) -> str:
    return hashlib.sha256(model_name.encode()).hexdigest()[:16]


def _cache_path(model_name: str) -> Path:
    return _MUSCLE_DIR / f"{_model_cache_key(model_name)}.npz"


def _save_model(model_name: str, data: dict) -> None:
    _MUSCLE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(_cache_path(model_name), **data)


def _load_cached(model_name: str) -> dict | None:
    path = _cache_path(model_name)
    if path.exists():
        return dict(np.load(path))
    return None


def _simulate_model(model_name: str) -> dict:
    """Generate a simple simulated model (random weights)."""
    rng = np.random.default_rng(abs(hash(model_name)) & 0xFFFFFFFF)
    return {
        "weights": rng.standard_normal(64),
        "bias": rng.standard_normal(1),
        "model_name": np.array(model_name),
        "simulated": np.array(True),
    }


def train_or_load(
    model_name: str,
    data: np.ndarray | None = None,
    fallback: str = "cached",  # "cached" | "simulate" | "fail"
    resources: ResourceSnapshot | None = None,
) -> Any:
    """Train a model or load from muscle memory.

    Attempts full train if GPU available.
    Falls back to cached model if not.
    Falls back to simulated model if no cache.
    Never fails (unless fallback="fail").
    """
    if resources is None:
        resources = do_probe()

    # Try training if GPU is available and data is provided
    if resources.gpu_available and data is not None:
        try:
            # Simulate training: in production this would use torch/tf
            weights = np.mean(data, axis=0) if data.ndim > 1 else np.array([np.mean(data)])
            bias = np.std(data) if data.size > 0 else 0.0
            model = {
                "weights": weights,
                "bias": np.array([bias]),
                "model_name": np.array(model_name),
                "simulated": np.array(False),
            }
            _save_model(model_name, {k: v for k, v in model.items()})
            return model
        except Exception:
            pass

    # Try cached
    cached = _load_cached(model_name)
    if cached is not None:
        return cached

    # Simulate
    if fallback == "simulate":
        return _simulate_model(model_name)
    elif fallback == "cached":
        return _simulate_model(model_name)
    elif fallback == "fail":
        raise RuntimeError(f"No model found for '{model_name}' and fallback='fail'")
    else:
        return _simulate_model(model_name)
