"""openmind-cellular: Resource-adaptive Jupyter computation layer."""

from .probe import ResourceSnapshot, probe
from .metabolism import MetabolicPath, select_path
from .train import train_or_load
from .sense import sense_or_simulate
from .infer import infer_adaptive
from .dataflow import DataTide
from .cell import cell

__all__ = [
    "ResourceSnapshot",
    "probe",
    "MetabolicPath",
    "select_path",
    "train_or_load",
    "sense_or_simulate",
    "infer_adaptive",
    "DataTide",
    "cell",
]
__version__ = "0.1.0"
