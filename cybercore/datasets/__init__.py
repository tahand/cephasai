"""Small bundled datasets for examples and the showcase."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources

from ..data import DataSource

@dataclass(frozen=True)
class _Example:
    file: str
    target: str
    description: str
    split: dict  # split() arguments used by the source course script
    features: tuple[str, ...] | None = None  # None: every non-target column


_EXAMPLES = {
    "salary": _Example(
        "salary.csv", "Salary", "Salary vs years of experience (30 rows, 1 feature).",
        {"test_size": 1 / 3, "random_state": 0},
    ),
    "startups": _Example(
        "startups.csv", "Profit", "Startup profit from spend and state (50 rows, 4 features, 1 categorical).",
        {"test_size": 0.2, "random_state": 0},
    ),
    "positions": _Example(
        "positions.csv", "Salary",
        "Salary by position level (10 rows, 1 feature). Too small to split: models train on every row.",
        {"test_size": 0, "random_state": 0},
        features=("Level",),  # Position is a text label for Level
    ),
}


def list_examples() -> dict[str, str]:
    return {name: ex.description for name, ex in _EXAMPLES.items()}


def example_split(name: str) -> dict:
    """The ``split()`` arguments that reproduce the example's course results."""
    return dict(_EXAMPLES[name].split)


def example_features(name: str) -> list[str] | None:
    features = _EXAMPLES[name].features
    return None if features is None else list(features)


def example_target(name: str) -> str:
    return _EXAMPLES[name].target


def example_path(name: str):
    return resources.files(__package__) / _EXAMPLES[name].file


def load_example(name: str) -> DataSource:
    """Load a bundled dataset as a DataSource, e.g. ``load_example("salary")``."""
    if name not in _EXAMPLES:
        raise KeyError(f"Unknown example {name!r}; available: {sorted(_EXAMPLES)}.")
    ex = _EXAMPLES[name]
    with resources.as_file(resources.files(__package__) / ex.file) as path:
        return DataSource(path, target_column=ex.target, features=example_features(name))
