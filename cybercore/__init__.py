"""CyberCore: plug and play wrappers for tabular machine learning."""

from .benchmark import Benchmark, BenchmarkResult
from .data import DataSource, infer_task
from .datasets import list_examples, load_example
from .models import Model, available_models, create_model, register_model
from .tuning import Categorical, Float, Int, TuneResult

__version__ = "0.1.0"

__all__ = [
    "Benchmark",
    "BenchmarkResult",
    "Categorical",
    "DataSource",
    "Float",
    "Int",
    "Model",
    "TuneResult",
    "available_models",
    "create_model",
    "infer_task",
    "list_examples",
    "load_example",
    "register_model",
]
