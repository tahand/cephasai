"""Hyperparameter search over a backend-neutral parameter space.

A search space is a dict mapping parameter names to :class:`Int`, :class:`Float`
or :class:`Categorical` (a plain list is treated as ``Categorical``). The same
space drives grid search, randomised search, or Optuna's TPE sampler.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, is_classifier
from sklearn.model_selection import KFold, ParameterGrid, ParameterSampler, StratifiedKFold, cross_val_score

Method = Literal["auto", "grid", "random", "optuna"]


@dataclass(frozen=True)
class Int:
    low: int
    high: int
    log: bool = False


@dataclass(frozen=True)
class Float:
    low: float
    high: float
    log: bool = False


@dataclass(frozen=True, init=False)
class Categorical:
    choices: tuple

    def __init__(self, choices):
        object.__setattr__(self, "choices", tuple(choices))


Space = dict[str, Any]

# Integer ranges with at most this many values are searched exhaustively by grid search.
_FULL_INT_GRID = 10


@dataclass
class TuneResult:
    best_params: dict[str, Any]
    best_score: float
    scoring: str
    method: str
    n_trials: int
    seconds: float
    trials: pd.DataFrame = field(repr=False)


def _normalise(space: Space) -> dict[str, Int | Float | Categorical]:
    return {k: Categorical(v) if isinstance(v, (list, tuple)) else v for k, v in space.items()}


def _as_grid(dim, points: int) -> list:
    if isinstance(dim, Categorical):
        return list(dim.choices)
    if isinstance(dim, Int) and dim.high - dim.low < max(points, _FULL_INT_GRID):
        return list(range(dim.low, dim.high + 1))  # small ranges: try every value
    spacing = np.geomspace if dim.log else np.linspace
    values = spacing(dim.low, dim.high, points)
    if isinstance(dim, Int):
        return sorted({int(round(v)) for v in values})
    return [float(v) for v in values]


def _as_distribution(dim):
    if isinstance(dim, Categorical):
        return list(dim.choices)
    if isinstance(dim, Int):
        return stats.randint(dim.low, dim.high + 1) if not dim.log else _LogInt(dim.low, dim.high)
    if dim.log:
        return stats.loguniform(dim.low, dim.high)
    return stats.uniform(dim.low, dim.high - dim.low)


class _LogInt:
    """Log-uniform integer sampler with the ``rvs`` interface scikit-learn expects."""

    def __init__(self, low: int, high: int):
        self._dist = stats.loguniform(low, high + 1)

    def rvs(self, random_state=None, size=None):
        return np.floor(self._dist.rvs(random_state=random_state, size=size)).astype(int)


def _grid_size(space, points: int) -> float:
    return math.prod(len(_as_grid(d, points)) for d in space.values())


def tune(
    build: Callable[..., BaseEstimator],
    X,
    y,
    space: Space,
    base_params: dict[str, Any] | None = None,
    method: Method = "auto",
    n_trials: int = 20,
    cv: int = 5,
    scoring: str = "r2",
    grid_points: int = 5,
    random_state: int | None = 42,
) -> tuple[BaseEstimator, TuneResult]:
    """Search ``space`` with cross-validation and return the best estimator, refitted on all of ``X, y``.

    ``build(**params)`` makes an unfitted estimator; it can be an estimator class
    such as ``Ridge`` or a ``Model.build_estimator`` method, so the space is keyed
    by the caller's own parameter names. ``base_params`` are fixed values that
    each candidate overrides.

    ``method="auto"`` uses an exhaustive grid when it has no more than ``n_trials``
    combinations, and random search otherwise. Continuous dimensions are
    discretised to ``grid_points`` values for grid search.
    """
    space = _normalise(space)
    if not space:
        raise ValueError("The search space is empty.")
    base_params = dict(base_params or {})
    # Shuffle the folds: rows can arrive sorted (e.g. an unsplit file ordered by its feature),
    # and unshuffled folds would then test only extrapolation.
    n_splits = max(2, min(cv, len(y)))
    if is_classifier(build(**base_params)):
        folds = StratifiedKFold(n_splits, shuffle=True, random_state=random_state)
    else:
        folds = KFold(n_splits, shuffle=True, random_state=random_state)

    def score(params: dict[str, Any]) -> float:
        return float(cross_val_score(build(**{**base_params, **params}), X, y, cv=folds, scoring=scoring).mean())

    if method == "auto":
        method = "grid" if _grid_size(space, grid_points) <= n_trials else "random"

    start = time.perf_counter()
    if method == "optuna":
        rows = _search_optuna(score, space, n_trials, random_state)
    elif method == "grid":
        rows = [{**p, "score": score(p)} for p in ParameterGrid({k: _as_grid(d, grid_points) for k, d in space.items()})]
    elif method == "random":
        sampler = ParameterSampler({k: _as_distribution(d) for k, d in space.items()}, n_iter=n_trials, random_state=random_state)
        rows = [{**p, "score": score(p)} for p in _unique(sampler)]
    else:
        raise ValueError(f"Unknown tuning method {method!r}.")

    trials = _trials_frame(rows)
    best_row = max(rows, key=lambda r: -np.inf if np.isnan(r["score"]) else r["score"])
    best_params = {k: _plain(v) for k, v in best_row.items() if k != "score"}
    best = build(**{**base_params, **best_params}).fit(X, y)
    seconds = time.perf_counter() - start
    return best, TuneResult(best_params, best_row["score"], scoring, method, len(trials), seconds, trials)


def _trials_frame(rows: list[dict]) -> pd.DataFrame:
    """One row per trial. Columns holding ``None`` (e.g. "no depth limit") keep it rather than becoming NaN."""
    columns = {}
    for key in rows[0]:
        values = [r[key] for r in rows]
        columns[key] = pd.Series(values, dtype=object if any(v is None for v in values) else None)
    return pd.DataFrame(columns)


def _unique(candidates):
    """Drop repeated draws, which random search makes when a space is mostly categorical."""
    seen = []
    for params in candidates:
        if params not in seen:
            seen.append(params)
            yield params


def _plain(value):
    return value.item() if isinstance(value, np.generic) else value


def _search_optuna(score, space, n_trials, random_state) -> list[dict]:
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("method='optuna' needs Optuna: pip install 'cybercore[tuning]'") from exc

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def suggest(trial, name, dim):
        if isinstance(dim, Categorical):
            return trial.suggest_categorical(name, list(dim.choices))
        if isinstance(dim, Int):
            return trial.suggest_int(name, dim.low, dim.high, log=dim.log)
        return trial.suggest_float(name, dim.low, dim.high, log=dim.log)

    def objective(trial):
        return score({name: suggest(trial, name, dim) for name, dim in space.items()})

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials)
    return [{**t.params, "score": t.value} for t in study.trials]


__all__ = ["Int", "Float", "Categorical", "TuneResult", "tune"]
