"""Compare several models on one DataSource split."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .data import DataSource
from .metrics import HIGHER_IS_BETTER, PRIMARY_METRIC
from .models import Model, create_model


@dataclass
class BenchmarkEntry:
    model: Model
    metrics: dict[str, float]  # on the test rows, or the training rows when there is no test set
    train_metrics: dict[str, float]
    fit_seconds: float
    predict_seconds: float
    y_pred: np.ndarray = field(repr=False)
    error: str | None = None


@dataclass
class BenchmarkResult:
    task: str
    entries: list[BenchmarkEntry]
    scored_on: str = "test"  # "train" when the data was split with test_size=0

    @property
    def primary_metric(self) -> str:
        return PRIMARY_METRIC[self.task]

    @property
    def table(self) -> pd.DataFrame:
        """One row per model, best first by the primary metric."""
        rows = []
        for e in self.entries:
            row: dict[str, Any] = {"model": e.model.label or e.model.name, "fitted_as": getattr(e.model, "variant", None)}
            row.update(e.metrics)
            if self.scored_on == "test":
                row.update({f"train_{k}": v for k, v in e.train_metrics.items()})
            row.update(fit_seconds=e.fit_seconds, predict_seconds=e.predict_seconds, error=e.error)
            rows.append(row)
        table = pd.DataFrame(rows)
        if self.primary_metric in table:
            table = table.sort_values(self.primary_metric, ascending=not HIGHER_IS_BETTER[self.primary_metric], na_position="last")
        return table.reset_index(drop=True)

    def best(self, metric: str | None = None) -> BenchmarkEntry:
        metric = metric or self.primary_metric
        scored = [e for e in self.entries if e.error is None and metric in e.metrics]
        if not scored:
            raise ValueError(f"No successful model reported {metric!r}.")
        pick = max if HIGHER_IS_BETTER.get(metric, True) else min
        return pick(scored, key=lambda e: e.metrics[metric])


class Benchmark:
    """Fit and score each model on the same cached split of ``data``.

    Args:
        data: The dataset. It is split with default settings if not already split.
        models: Registered model names or ``Model`` instances. Defaults to every
            registered model that matches the data's task.
        tune: Run each model's hyperparameter search instead of a plain fit.
        tune_kwargs: Extra arguments for :meth:`Model.tune`.
    """

    def __init__(self, data: DataSource, models: Sequence[str | Model] | None = None, tune: bool = False, **tune_kwargs):
        from .models import available_models

        self.data = data
        if models is None:
            models = list(available_models(data.task))
        self.models = [create_model(m) if isinstance(m, str) else m for m in models]
        self.tune = tune
        self.tune_kwargs = tune_kwargs

    def run(self) -> BenchmarkResult:
        X_train, X_test, y_train, y_test = self.data.ensure_split()
        has_test = self.data.has_test_set
        entries = []
        for model in self.models:
            try:
                if self.tune and model.param_space:
                    model.tune(self.data, **self.tune_kwargs)
                else:
                    model.fit(self.data)
                X_eval = X_test if has_test else X_train
                start = time.perf_counter()
                y_pred = model.predict(X_eval)
                predict_seconds = time.perf_counter() - start
                train_metrics = model.evaluate(X_test=X_train, y_test=y_train)
                entries.append(BenchmarkEntry(
                    model=model,
                    metrics=model.evaluate(self.data) if has_test else train_metrics,
                    train_metrics=train_metrics,
                    fit_seconds=model.fit_seconds_ or 0.0,
                    predict_seconds=predict_seconds,
                    y_pred=y_pred,
                ))
            except Exception as exc:  # one failing model shouldn't sink the whole comparison
                entries.append(BenchmarkEntry(model, {}, {}, 0.0, 0.0, np.array([]), error=f"{type(exc).__name__}: {exc}"))
        return BenchmarkResult(self.data.task, entries, scored_on="test" if has_test else "train")
