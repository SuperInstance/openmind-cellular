"""Metabolic pathways — auto-select processing strategy based on resources."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .probe import ResourceSnapshot


class MetabolicPath(str, Enum):
    """Five metabolic pathways for computation."""

    FULL_TRAIN = "full_train"          # GPU + API + RAM + Time
    TRANSFER = "transfer"              # GPU + pretrained model
    CLOUD_INFERENCE = "cloud"          # No GPU, API key valid
    MUSCLE_MEMORY = "muscle_memory"    # No GPU, no API, use cache
    HARDWARE_LOOP = "hardware_loop"    # ESP32/sensor online


def select_path(task: str, resources: ResourceSnapshot) -> MetabolicPath:
    """Select the best metabolic pathway for a task given current resources.

    The tripartite synchronizer decides based on:
    1. What hardware is online (GPU, ESP32)
    2. What services are reachable (API keys, network)
    3. What the task needs (training vs inference vs sensing)
    """
    task_lower = task.lower()

    # Hardware loop: if ESP32 is online and task is sensor-related
    if resources.esp32_online and any(
        kw in task_lower for kw in ("sensor", "sense", "read", "measure", "monitor", "hardware")
    ):
        return MetabolicPath.HARDWARE_LOOP

    # Full train: GPU available + network + sufficient RAM
    has_api = any(resources.api_keys.values())
    if resources.gpu_available and resources.ram_available_gb >= 4.0:
        if "train" in task_lower or "learn" in task_lower or "fit" in task_lower:
            return MetabolicPath.FULL_TRAIN
        return MetabolicPath.TRANSFER

    # Cloud inference: no GPU but API key available
    if has_api and resources.network:
        return MetabolicPath.CLOUD_INFERENCE

    # Muscle memory: no GPU, no API — use cache
    return MetabolicPath.MUSCLE_MEMORY
