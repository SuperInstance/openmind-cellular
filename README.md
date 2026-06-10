# openmind-cellular

**Resource-adaptive Jupyter computation layer.**

Each computation cell adapts its processing strategy based on what's available *right now* — GPU, cloud APIs, cached models, or hardware sensors. No more crashes because the notebook expected a GPU that isn't there.

## The Idea: Cellular Computation

Biological cells adapt to their environment. A muscle cell uses different metabolic pathways depending on oxygen availability — aerobic when oxygen is plentiful, anaerobic when it's not. Neither pathway is "better"; they're both essential, and the cell switches between them.

`openmind-cellular` brings this to Jupyter notebooks. Instead of writing `if torch.cuda.is_available(): ...` everywhere, you write your computation once and the system adapts:

```python
from openmind_cellular import cell, probe, sense_or_simulate, train_or_load, infer_adaptive

@cell(resource_aware=True, fallback="cached")
def my_pipeline():
    resources = probe()  # What's available right now?
    data = sense_or_simulate("temperature", duration="1h")  # Real sensor or realistic sim
    model = train_or_load("my-model", data=data)  # Train or load from cache
    return infer_adaptive(model, data)  # GPU, API, or cached inference

results = my_pipeline()  # Works on your gaming PC, your laptop, or offline
```

The same pipeline runs differently (but correctly) on:
- **Gaming PC with RTX 4090**: Full training on GPU with real data
- **Laptop on WiFi**: Cloud API inference with simulated data
- **Offline Raspberry Pi**: Cached model predictions from muscle memory
- **Workshop bench**: Real ESP32 sensor data feeding a hardware loop

## Five Metabolic Pathways

| Pathway | Trigger | What Happens |
|---------|---------|-------------|
| **Full Train** | GPU + RAM + time | Train model from scratch on local hardware |
| **Transfer** | GPU available | Fine-tune pretrained model |
| **Cloud Inference** | API key + network | Send data to cloud API for inference |
| **Muscle Memory** | Nothing else | Use cached models and predictions |
| **Hardware Loop** | ESP32/sensor online | Read real sensor data, local processing |

The system automatically selects the best pathway. You can override with explicit strategies, but the adaptive default usually does the right thing.

## Muscle Memory

When models are trained, they're cached in `~/.openmind/muscle_memory/`. When you run offline or without a GPU, the system loads these cached models instead of crashing. This is "muscle memory" — the computation remembers what it learned and can still function without full resources.

The `DataTide` class manages the ebb and flow between simulated and real data. When real data arrives, it calibrates the simulation parameters so future simulated data is more realistic.

## Installation

```bash
pip install openmind-cellular

# Optional: GPU support
pip install openmind-cellular[gpu]

# Optional: Cloud API support
pip install openmind-cellular[api]
```

## API

- `probe()` — Take a resource snapshot (< 1 second)
- `select_path(task, resources)` — Choose metabolic pathway
- `train_or_load(model_name, data)` — Train or load from cache
- `sense_or_simulate(source, duration)` — Real sensors or simulation
- `infer_adaptive(model, data)` — Adaptive inference
- `DataTide` — Simulation ↔ reality bridge
- `@cell(resource_aware=True)` — Decorator for resource-aware cells

## License

MIT
