"""Adaptive inferencer — run inference adapting to available resources."""

from __future__ import annotations

from typing import Any

import numpy as np

from .probe import ResourceSnapshot, probe as do_probe


def _infer_gpu(model: Any, data: np.ndarray) -> np.ndarray:
    """Run inference on GPU (simulated)."""
    weights = model.get("weights", np.ones(1))
    if isinstance(weights, np.ndarray) and weights.ndim >= 1:
        return np.dot(data, weights) if data.ndim > 1 else data * np.mean(weights)
    return data


def _infer_api(model: Any, data: np.ndarray) -> np.ndarray:
    """Run inference via cloud API (simulated)."""
    weights = model.get("weights", np.ones(1))
    bias = model.get("bias", np.zeros(1))
    if isinstance(weights, np.ndarray):
        result = np.dot(data, weights) if data.ndim > 1 else data * np.mean(weights)
        if isinstance(bias, np.ndarray) and bias.size > 0:
            result = result + bias[0]
        return result
    return data


def _infer_cached(model: Any, data: np.ndarray) -> np.ndarray:
    """Use muscle memory predictions."""
    weights = model.get("weights", np.ones(1))
    if isinstance(weights, np.ndarray) and weights.size > 0:
        # Simple linear prediction from cached weights
        if data.ndim > 1:
            w = weights[:data.shape[-1]] if weights.size >= data.shape[-1] else np.resize(weights, data.shape[-1])
            return np.dot(data, w)
        return data * weights[0]
    return np.zeros(data.shape[0])


def infer_adaptive(
    model: Any,
    data: np.ndarray,
    strategy: str = "adaptive",  # "adaptive" | "gpu" | "api" | "cached"
    resources: ResourceSnapshot | None = None,
) -> np.ndarray:
    """Run inference adapting to available resources.

    adaptive: try GPU → try API → use cached predictions
    gpu: force local GPU (error if unavailable)
    api: force cloud API (error if no key)
    cached: always use muscle memory predictions
    """
    if resources is None:
        resources = do_probe()

    if strategy == "gpu":
        if not resources.gpu_available:
            raise RuntimeError("GPU not available but strategy='gpu'")
        return _infer_gpu(model, data)

    if strategy == "api":
        if not any(resources.api_keys.values()):
            raise RuntimeError("No API key available but strategy='api'")
        return _infer_api(model, data)

    if strategy == "cached":
        return _infer_cached(model, data)

    # adaptive: try GPU → API → cached
    if resources.gpu_available:
        return _infer_gpu(model, data)

    if any(resources.api_keys.values()) and resources.network:
        return _infer_api(model, data)

    return _infer_cached(model, data)
