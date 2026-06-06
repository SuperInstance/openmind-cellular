"""Data ebb/flow — the simulation ↔ reality bridge."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

_MUSCLE_DIR = Path.home() / ".openmind" / "muscle_memory"


class DataTide:
    """Manages the ebb and flow between simulated and real data."""

    def __init__(self) -> None:
        self._simulated: dict[str, dict[str, Any]] = {}
        self._real: dict[str, dict[str, Any]] = {}
        self._calibrated: dict[str, dict[str, float]] = {}

    def add_simulated(self, source: str, data: np.ndarray, params: dict) -> None:
        """Add simulated data with its generation parameters."""
        self._simulated[source] = {"data": data, "params": params}

    def add_real(self, source: str, data: np.ndarray) -> None:
        """Add real sensor data."""
        self._real[source] = {"data": data}

    def calibrate(self) -> dict[str, dict[str, float]]:
        """Compare simulated vs real data, update parameters.

        Returns calibration metrics per source.
        """
        results: dict[str, dict[str, float]] = {}
        for source in self._simulated:
            if source in self._real:
                sim = np.asarray(self._simulated[source]["data"]).flatten()
                real = np.asarray(self._real[source]["data"]).flatten()
                n = min(len(sim), len(real))
                if n == 0:
                    continue
                sim, real = sim[:n], real[:n]

                # Compute calibration metrics
                diff = sim - real
                rmse = float(np.sqrt(np.mean(diff ** 2)))
                mae = float(np.mean(np.abs(diff)))
                sim_std = float(np.std(sim)) if np.std(sim) > 0 else 1.0
                nrmse = rmse / sim_std

                # Update params to better match reality
                old_params = self._simulated[source]["params"]
                real_mean = float(np.mean(real))
                real_std = float(np.std(real)) if np.std(real) > 0.1 else old_params.get("std", 1.0)

                new_params = {
                    "mean": real_mean,
                    "std": real_std,
                    "min": old_params.get("min", real_mean - 4 * real_std),
                    "max": old_params.get("max", real_mean + 4 * real_std),
                }
                self._simulated[source]["params"] = new_params
                self._calibrated[source] = new_params

                # Persist
                self._save_distribution(source, new_params)

                results[source] = {
                    "rmse": rmse,
                    "mae": mae,
                    "nrmse": nrmse,
                    "samples_compared": float(n),
                }
        return results

    def generate(self, source: str, n: int) -> np.ndarray:
        """Generate calibrated simulated data for a source."""
        if source in self._simulated:
            params = self._simulated[source]["params"]
        else:
            params = self._load_distribution(source)

        rng = np.random.default_rng()
        raw = rng.normal(params.get("mean", 0), params.get("std", 1), n)
        return np.clip(raw, params.get("min", -1e9), params.get("max", 1e9))

    def quality_report(self) -> dict[str, dict[str, Any]]:
        """Report how close simulation is to reality per source."""
        report: dict[str, dict[str, Any]] = {}
        for source in self._simulated:
            entry: dict[str, Any] = {
                "has_simulated": True,
                "has_real": source in self._real,
                "calibrated": source in self._calibrated,
            }
            if source in self._calibrated:
                entry["calibration_params"] = self._calibrated[source]
            if source in self._simulated:
                sim = np.asarray(self._simulated[source]["data"]).flatten()
                entry["sim_samples"] = len(sim)
                entry["sim_mean"] = float(np.mean(sim))
            if source in self._real:
                real = np.asarray(self._real[source]["data"]).flatten()
                entry["real_samples"] = len(real)
                entry["real_mean"] = float(np.mean(real))
            report[source] = entry
        return report

    @staticmethod
    def _save_distribution(source: str, params: dict) -> None:
        _MUSCLE_DIR.mkdir(parents=True, exist_ok=True)
        path = _MUSCLE_DIR / f"dist_{source}.json"
        path.write_text(json.dumps(params))

    @staticmethod
    def _load_distribution(source: str) -> dict:
        path = _MUSCLE_DIR / f"dist_{source}.json"
        if path.exists():
            return json.loads(path.read_text())
        return {"mean": 22.0, "std": 3.0, "min": -10.0, "max": 50.0}
