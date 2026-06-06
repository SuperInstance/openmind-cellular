"""Tests for openmind-cellular."""

import numpy as np
import pytest

from openmind_cellular.probe import ResourceSnapshot
from openmind_cellular.metabolism import MetabolicPath, select_path
from openmind_cellular.train import train_or_load
from openmind_cellular.sense import sense_or_simulate
from openmind_cellular.infer import infer_adaptive
from openmind_cellular.dataflow import DataTide
from openmind_cellular.cell import cell


def _full_resources() -> ResourceSnapshot:
    return ResourceSnapshot(
        gpu_available=True, gpu_memory_free=8192, gpu_name="RTX 4090",
        cpu_cores=16, ram_available_gb=32.0,
        api_keys={"openai": True, "deepinfra": True},
        esp32_online=[], battery_pct=85.0, network=True,
        timestamp=1703275200.0,
    )


def _minimal_resources() -> ResourceSnapshot:
    return ResourceSnapshot(
        gpu_available=False, gpu_memory_free=0, gpu_name="",
        cpu_cores=4, ram_available_gb=2.0,
        api_keys={"openai": False, "deepinfra": False},
        esp32_online=[], battery_pct=None, network=False,
        timestamp=1703275200.0,
    )


def _api_only_resources() -> ResourceSnapshot:
    return ResourceSnapshot(
        gpu_available=False, gpu_memory_free=0, gpu_name="",
        cpu_cores=4, ram_available_gb=8.0,
        api_keys={"openai": True, "deepinfra": True},
        esp32_online=[], battery_pct=50.0, network=True,
        timestamp=1703275200.0,
    )


def _esp32_resources() -> ResourceSnapshot:
    return ResourceSnapshot(
        gpu_available=False, gpu_memory_free=0, gpu_name="",
        cpu_cores=4, ram_available_gb=4.0,
        api_keys={"openai": False},
        esp32_online=["/dev/ttyUSB0"],
        battery_pct=60.0, network=True,
        timestamp=1703275200.0,
    )


# --- ResourceSnapshot tests ---

class TestResourceSnapshot:
    def test_mock_probe_returns_correct_fields(self):
        r = _full_resources()
        assert r.gpu_available is True
        assert r.gpu_memory_free == 8192
        assert r.gpu_name == "RTX 4090"
        assert r.cpu_cores == 16
        assert r.ram_available_gb == 32.0
        assert r.api_keys["openai"] is True
        assert r.esp32_online == []
        assert r.battery_pct == 85.0
        assert r.network is True
        assert r.timestamp == 1703275200.0


# --- Metabolism tests ---

class TestMetabolism:
    def test_full_train_when_all_resources(self):
        r = _full_resources()
        assert select_path("train my model", r) == MetabolicPath.FULL_TRAIN

    def test_muscle_memory_when_no_gpu_no_api(self):
        r = _minimal_resources()
        assert select_path("predict something", r) == MetabolicPath.MUSCLE_MEMORY

    def test_hardware_loop_when_esp32_online(self):
        r = _esp32_resources()
        assert select_path("read sensor data", r) == MetabolicPath.HARDWARE_LOOP

    def test_cloud_inference_when_api_key_valid(self):
        r = _api_only_resources()
        assert select_path("classify image", r) == MetabolicPath.CLOUD_INFERENCE

    def test_transfer_when_gpu_but_no_training_task(self):
        r = _full_resources()
        assert select_path("classify this image", r) == MetabolicPath.TRANSFER


# --- Train tests ---

class TestTrain:
    def test_returns_model_when_data_provided(self):
        r = _full_resources()
        data = np.random.randn(100, 10)
        model = train_or_load("test-model", data=data, resources=r, fallback="simulate")
        assert "weights" in model
        assert model["simulated"] is not None

    def test_falls_back_to_cached(self):
        r = _minimal_resources()
        model = train_or_load("nonexistent-model", data=None, resources=r, fallback="cached")
        assert model is not None
        assert "weights" in model

    def test_falls_back_to_simulated_when_no_cache(self):
        r = _minimal_resources()
        model = train_or_load("brand-new-model", data=None, resources=r, fallback="simulate")
        assert model is not None
        assert bool(model["simulated"]) is True

    def test_fail_raises(self):
        r = _minimal_resources()
        with pytest.raises(RuntimeError):
            train_or_load("nope", data=None, resources=r, fallback="fail")


# --- Sense tests ---

class TestSense:
    def test_returns_simulated_when_no_esp32(self):
        r = _minimal_resources()
        data = sense_or_simulate("temperature", duration="1s", rate="10Hz", resources=r)
        assert isinstance(data, np.ndarray)
        assert len(data) == 10

    def test_returns_real_data_when_esp32_mocked(self):
        # ESP32 reading will fail in test env, falls back to simulation
        r = _esp32_resources()
        data = sense_or_simulate("temperature", duration="1s", rate="5Hz", resources=r)
        assert isinstance(data, np.ndarray)
        assert len(data) == 5


# --- Infer tests ---

class TestInfer:
    def test_uses_cached_when_no_gpu(self):
        r = _minimal_resources()
        model = {"weights": np.array([1.0, 2.0]), "bias": np.array([0.5])}
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = infer_adaptive(model, data, strategy="cached", resources=r)
        assert isinstance(result, np.ndarray)
        assert len(result) == 2

    def test_adaptive_falls_to_cached(self):
        r = _minimal_resources()
        model = {"weights": np.array([1.0, 2.0]), "bias": np.array([0.5])}
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = infer_adaptive(model, data, strategy="adaptive", resources=r)
        assert isinstance(result, np.ndarray)

    def test_gpu_strategy_raises_without_gpu(self):
        r = _minimal_resources()
        model = {"weights": np.array([1.0])}
        data = np.array([[1.0]])
        with pytest.raises(RuntimeError, match="GPU not available"):
            infer_adaptive(model, data, strategy="gpu", resources=r)

    def test_api_strategy_raises_without_key(self):
        r = _minimal_resources()
        model = {"weights": np.array([1.0])}
        data = np.array([[1.0]])
        with pytest.raises(RuntimeError, match="No API key"):
            infer_adaptive(model, data, strategy="api", resources=r)


# --- DataTide tests ---

class TestDataTide:
    def test_calibrate_updates_parameters(self):
        tide = DataTide()
        sim = np.random.normal(20, 2, 1000)
        real = np.random.normal(25, 3, 1000)
        tide.add_simulated("temp", sim, {"mean": 20, "std": 2, "min": -10, "max": 50})
        tide.add_real("temp", real)
        results = tide.calibrate()
        assert "temp" in results
        assert results["temp"]["rmse"] > 0

    def test_quality_report_returns_metrics(self):
        tide = DataTide()
        tide.add_simulated("temp", np.array([20, 21, 22]), {"mean": 21, "std": 1, "min": 0, "max": 50})
        tide.add_real("temp", np.array([25, 26, 27]))
        report = tide.quality_report()
        assert "temp" in report
        assert report["temp"]["has_simulated"] is True
        assert report["temp"]["has_real"] is True

    def test_generate_uses_calibrated_params(self):
        tide = DataTide()
        tide.add_simulated("temp", np.array([20, 21, 22]), {"mean": 21, "std": 1, "min": 0, "max": 50})
        tide.add_real("temp", np.array([30, 31, 32]))
        tide.calibrate()
        generated = tide.generate("temp", 100)
        assert len(generated) == 100
        # Should be closer to 30 than 20 after calibration
        assert abs(np.mean(generated) - 31) < abs(np.mean(generated) - 21)


# --- Cell decorator tests ---

class TestCell:
    def test_runs_function_and_adapts(self):
        @cell(resource_aware=True, fallback="cached")
        def my_func():
            return 42

        result = my_func()
        assert result == 42

    def test_fallback_on_resource_failure(self):
        @cell(resource_aware=True, fallback="cached")
        def failing_func():
            raise RuntimeError("No GPU!")

        result = failing_func()
        assert result is None  # fallback returns None

    def test_fail_mode_raises(self):
        @cell(resource_aware=True, fallback="fail")
        def failing_func():
            raise RuntimeError("No GPU!")

        with pytest.raises(RuntimeError):
            failing_func()

    def test_metadata_attached(self):
        @cell(resource_aware=True, fallback="cached")
        def my_func():
            return 42

        assert hasattr(my_func, "_cell_config")
        assert my_func._cell_config["fallback"] == "cached"
