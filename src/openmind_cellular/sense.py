"""Adaptive sensor — read real data or generate realistic simulation."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .probe import ResourceSnapshot, probe as do_probe

_MUSCLE_DIR = Path.home() / ".openmind" / "muscle_memory"


def _parse_duration(duration: str) -> float:
    """Parse duration string like '1h', '30m', '2d' to seconds."""
    unit = duration[-1].lower()
    value = float(duration[:-1])
    multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    return value * multipliers.get(unit, 1)


def _parse_rate(rate: str) -> float:
    """Parse rate string like '1Hz', '10Hz' to Hz."""
    return float(rate.replace("Hz", "").strip())


def _load_distribution(source: str) -> dict:
    """Load cached distribution parameters for a source."""
    path = _MUSCLE_DIR / f"dist_{source}.json"
    if path.exists():
        import json
        return json.loads(path.read_text())
    return {"mean": 22.0, "std": 3.0, "min": -10.0, "max": 50.0}


def _generate_simulated(source: str, n_samples: int) -> np.ndarray:
    """Generate realistic simulated data from cached distributions."""
    dist = _load_distribution(source)
    rng = np.random.default_rng(hash(source) & 0xFFFFFFFF)
    raw = rng.normal(dist["mean"], dist["std"], n_samples)
    return np.clip(raw, dist.get("min", -1e9), dist.get("max", 1e9))


def _read_esp32(port: str, n_samples: int) -> np.ndarray:
    """Read data from ESP32 over serial."""
    try:
        import serial  # type: ignore
        ser = serial.Serial(port, 115200, timeout=2)
        readings = []
        for _ in range(min(n_samples, 100)):
            line = ser.readline().decode().strip()
            try:
                readings.append(float(line))
            except ValueError:
                continue
        ser.close()
        if readings:
            return np.array(readings)
    except Exception:
        pass
    return _generate_simulated("esp32_fallback", n_samples)


def sense_or_simulate(
    source: str,
    duration: str = "1h",
    rate: str = "1Hz",
    esp32_port: str | None = None,
    resources: ResourceSnapshot | None = None,
) -> np.ndarray:
    """Read sensor data or generate realistic simulation.

    If ESP32 is online: real sensor data.
    If not: simulated data from cached distributions.
    """
    if resources is None:
        resources = do_probe()

    seconds = _parse_duration(duration)
    hz = _parse_rate(rate)
    n_samples = int(seconds * hz)

    # Determine port
    port = esp32_port
    if port is None and resources.esp32_online:
        port = resources.esp32_online[0]

    # Try real data
    if port is not None and port in resources.esp32_online:
        return _read_esp32(port, n_samples)

    # Fallback to simulation
    return _generate_simulated(source, n_samples)
