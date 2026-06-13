# OpenMind Cellular

**OpenMind Cellular** is a Python resource-adaptive computation layer for Jupyter notebooks that dynamically selects processing strategies based on real-time hardware availability — GPU, cloud APIs, cached models, or hardware sensors.

## Why It Matters

Modern ML notebooks crash when expected hardware is missing. A training pipeline developed on an RTX 4090 fails on a laptop without CUDA. OpenMind Cellular solves this by treating computation like biological cellular metabolism: each "cell" (function) adapts its processing pathway to available resources, just as muscle cells switch between aerobic and anaerobic respiration based on oxygen availability. The same pipeline runs on a gaming PC (full GPU training), a laptop (cloud API inference), or offline (cached predictions from muscle memory) — always producing correct results.

## How It Works

### Resource Probing

The `probe()` function inspects available computation resources at call time:

```
GPU available?     → CUDA device count, VRAM
Cloud API reachable? → network latency check
Cached model exists? → filesystem lookup
Hardware sensors?   → serial/I2C device scan
```

Probe cost: **O(1)** — a fixed set of system queries with timeouts.

### Adaptive Dispatch

Each `@cell(resource_aware=True)` decorator wraps the function in a strategy selector:

```
strategy = select_strategy(probe())
  if GPU:          return gpu_strategy
  elif cloud_api:  return cloud_strategy
  elif cache:      return cached_strategy
  else:            return fallback_strategy
```

Strategy selection is **O(1)** — a priority-ordered cascade.

### Sense-or-Simulate

The `sense_or_simulate()` function provides transparent data sourcing:

- **Real sensor**: ESP32 bridge via `openmind-esp32-bridge` (cost: network round-trip)
- **Simulation**: Cached or generated data matching sensor characteristics

### Train-or-Load

`train_or_load()` implements the cache-or-compute tradeoff:

```
if cache_valid(model_id):
    return load(cache_path)     # O(1) — disk read
else:
    model = train(data)          # O(N·E) — N samples, E epochs
    save(cache_path, model)
    return model
```

### Metabolic Pathways

The `metabolism` module models energy/compute budget allocation, analogous to ATP management in biological cells:

```
budget = probe_compute_budget()
if budget > threshold_high:  full_training()
elif budget > threshold_low: quantized_training()
else:                        inference_only()
```

## Quick Start

```python
from openmind_cellular import cell, probe, sense_or_simulate, train_or_load, infer_adaptive

@cell(resource_aware=True, fallback="cached")
def my_pipeline():
    resources = probe()
    data = sense_or_simulate("temperature", duration="1h")
    model = train_or_load("my-model", data=data)
    return infer_adaptive(model, data)

results = my_pipeline()  # Adapts to any environment
```

## API

| Function | Description |
|----------|-------------|
| `@cell(resource_aware, fallback)` | Decorator making a function resource-adaptive |
| `probe()` | Inspect GPU, cloud, cache, and sensor availability |
| `sense_or_simulate(name, duration)` | Real sensor data or realistic simulation |
| `train_or_load(model_id, data)` | Train fresh or load from cache |
| `infer_adaptive(model, data)` | GPU/API/cached inference selection |

Modules: `cell`, `probe`, `metabolism`, `dataflow`, `train`

## Architecture Notes

OpenMind Cellular is the computation layer of the OpenMind nervous system in SuperInstance. In γ + η = C, it represents γ (growth — maximizing computation quality given available resources) modulated by η (avoidance — gracefully degrading when resources are scarce). The ESP32 bridge integration connects biological-inspired computation to real-world sensors.

See [ARCHITECTURE.md](https://github.com/SuperInstance/SuperInstance/blob/main/ARCHITECTURE.md) for the OpenMind architecture.

## References

1. Foster, I. et al. (2017). "Cloud Computing for Scientific Research." *Communications of the ACM*.
2. Kluyver, T. et al. (2016). "Jupyter Notebooks — a publishing format for reproducible computational workflows." *Positioning and Power in Academic Publishing*.
3. Dean, J. et al. (2012). "Large Scale Distributed Deep Networks." *NeurIPS*.

## License

MIT
