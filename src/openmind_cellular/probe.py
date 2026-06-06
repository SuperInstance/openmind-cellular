"""Resource probe — detect what's available right now."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field


def _check_gpu() -> tuple[bool, int, str]:
    """Check GPU availability via nvidia-smi."""
    if not shutil.which("nvidia-smi"):
        return False, 0, ""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,name", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return False, 0, ""
        line = result.stdout.strip().split("\n")[0]
        parts = line.split(",")
        free_mb = int(parts[0].strip())
        name = parts[1].strip()
        return True, free_mb, name
    except Exception:
        return False, 0, ""


def _check_api_keys() -> dict[str, bool]:
    """Check which API keys are available in environment."""
    keys = {}
    for name, env_var in [
        ("openai", "OPENAI_API_KEY"),
        ("deepinfra", "DEEPINFRA_API_KEY"),
        ("anthropic", "ANTHROPIC_API_KEY"),
        ("huggingface", "HF_TOKEN"),
    ]:
        val = os.environ.get(env_var, "")
        keys[name] = bool(val and val.startswith(("sk-", "hf_", "key_")) or len(val) > 8)
    return keys


def _check_esp32() -> list[str]:
    """Check for ESP32 devices on serial ports."""
    ports: list[str] = []
    try:
        import serial.tools.list_ports  # type: ignore
        for p in serial.tools.list_ports.comports():
            if p.vid is not None:
                ports.append(p.device)
    except ImportError:
        pass
    # Also check /dev/ttyUSB* and /dev/ttyACM* on Linux
    import glob
    for pattern in ("/dev/ttyUSB*", "/dev/ttyACM*"):
        for path in glob.glob(pattern):
            if path not in ports:
                ports.append(path)
    return ports


def _check_battery() -> float | None:
    """Check battery percentage (laptop)."""
    # Try UPower on Linux
    try:
        result = subprocess.run(
            ["upower", "-e"], capture_output=True, text=True, timeout=3,
        )
        for line in result.stdout.strip().split("\n"):
            if "BAT" in line:
                info = subprocess.run(
                    ["upower", "-i", line], capture_output=True, text=True, timeout=3,
                )
                for iline in info.stdout.split("\n"):
                    if "percentage" in iline:
                        return float(iline.split(":")[1].strip().replace("%", ""))
        return None
    except Exception:
        return None


def _check_network() -> bool:
    """Check if internet is available."""
    try:
        import socket
        sock = socket.create_connection(("8.8.8.8", 53), timeout=2)
        sock.close()
        return True
    except Exception:
        return False


@dataclass
class ResourceSnapshot:
    """Snapshot of currently available resources."""

    gpu_available: bool = False
    gpu_memory_free: int = 0  # MB
    gpu_name: str = ""
    cpu_cores: int = 0
    ram_available_gb: float = 0.0
    api_keys: dict[str, bool] = field(default_factory=dict)
    esp32_online: list[str] = field(default_factory=list)
    battery_pct: float | None = None
    network: bool = False
    timestamp: float = 0.0


def probe() -> ResourceSnapshot:
    """Take a resource snapshot. Completes in <1 second."""
    gpu_ok, gpu_mem, gpu_name = _check_gpu()
    api_keys = _check_api_keys()
    esp32 = _check_esp32()

    # RAM available
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    ram_gb = int(line.split()[1]) / (1024 * 1024)
                    break
            else:
                ram_gb = 0.0
    except Exception:
        ram_gb = 0.0

    return ResourceSnapshot(
        gpu_available=gpu_ok,
        gpu_memory_free=gpu_mem,
        gpu_name=gpu_name,
        cpu_cores=os.cpu_count() or 0,
        ram_available_gb=round(ram_gb, 2),
        api_keys=api_keys,
        esp32_online=esp32,
        battery_pct=_check_battery(),
        network=_check_network(),
        timestamp=time.time(),
    )
