"""Model base class and registry.

Adding a model is one small subclass::

    @register_model
    class MyModel(Model):
        name = "my_model"
        label = "My Model"
        task = "regression"
        param_space = {"alpha": Float(1e-3, 10, log=True)}

        def build_estimator(self, **params):
            return SomeSklearnEstimator(**params)
"""

from __future__ import annotations

import time
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from ..data import DataSource, Task
from ..metrics import CV_SCORING, classification_metrics, regression_metrics
from ..tuning import Method, Space, TuneResult, tune

_REGISTRY: dict[str, type["Model"]] = {}


def register_model(cls: type["Model"]) -> type["Model"]:
    """Class decorator that makes a model available by name."""
    if not getattr(cls, "name", None):
        raise TypeError(f"{cls.__name__} needs a non-empty `name` to be registered.")
    if cls.name in _REGISTRY:
        raise ValueError(f"A model named {cls.name!r} is already registered.")
    _REGISTRY[cls.name] = cls
    return cls


def available_models(task: Task | None = None) -> dict[str, type["Model"]]:
    """Registered models, optionally filtered to those that support ``task``."""
    return {n: c for n, c in _REGISTRY.items() if task is None or c.task == task}


def create_model(name: str, **params) -> "Model":
    try:
        return _REGISTRY[name](**params)
    except KeyError:
        raise KeyError(f"Unknown model {name!r}; available: {sorted(_REGISTRY)}.") from None


class Model:
    """A named, tunable wrapper around a scikit-learn style estimator.

    Subclasses set the class attributes and implement :meth:`build_estimator`.
    Constructor keyword arguments override ``default_params``.
    """

    name: ClassVar[str] = ""
    label: ClassVar[str] = ""
    task: ClassVar[Task] = "regression"
    description: ClassVar[str] = ""
    default_params: ClassVar[dict[str, Any]] = {}
    param_space: ClassVar[Space] = {}

    def __init__(self, **params):
        self.params: dict[str, Any] = {**self.default_params, **params}
        self.estimator_: BaseEstimator | None = None
        self.feature_names_: list[str] | None = None
        self.fit_seconds_: float | None = None
        self.tune_result_: TuneResult | None = None
        self._data: DataSource | None = None

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{type(self).__name__}({args})"

    def build_estimator(self, **params) -> BaseEstimator:
        raise NotImplementedError

    @property
    def variant(self) -> str:
        """A plain description of what was fitted, e.g. ``"Multiple linear regression (4 features)"``.

        Subclasses override :meth:`_variant_name` to name the form they took.
        """
        return f"{self._variant_name()} ({self._feature_count_text()})" if self.is_fitted else self.label or self.name

    def _variant_name(self) -> str:
        return self.label or self.name

    def _n_input_features(self) -> int:
        """Features before encoding, when fitted from a DataSource; otherwise the columns fitted on."""
        if self._data is not None:
            return len(self._data.features)
        return self._require_fitted().n_features_in_

    def _feature_count_text(self) -> str:
        n = self._n_input_features()
        text = f"{n} feature" + ("" if n == 1 else "s")
        columns = len(self.feature_names_) if self.feature_names_ else n
        if columns != n:
            text += f", {columns} columns after encoding"
        return text

    @property
    def is_fitted(self) -> bool:
        return self.estimator_ is not None

    def _check_task(self, data: DataSource) -> None:
        if data.task != self.task:
            raise ValueError(f"{self.label or self.name} is a {self.task} model but the data is a {data.task} task.")

    def _training_data(self, data, y):
        if isinstance(data, DataSource):
            self._check_task(data)
            self._data = data
            X_train, _, y_train, _ = data.ensure_split()
            return X_train, y_train
        if y is None:
            raise ValueError("Pass a DataSource, or both X and y.")
        self._data = None
        return data, y

    # ------------------------------------------------------------------ training

    def fit(self, data: DataSource | pd.DataFrame | np.ndarray, y=None) -> "Model":
        """Fit on a DataSource's training split, or on explicit ``X, y``."""
        X, y = self._training_data(data, y)
        estimator = self.build_estimator(**self.params)
        start = time.perf_counter()
        estimator.fit(X, y)
        self.fit_seconds_ = time.perf_counter() - start
        self.estimator_ = estimator
        self.feature_names_ = list(X.columns) if isinstance(X, pd.DataFrame) else None
        return self

    def tune(
        self,
        data: DataSource | pd.DataFrame | np.ndarray,
        y=None,
        space: Space | None = None,
        method: Method = "auto",
        n_trials: int = 20,
        cv: int = 5,
        scoring: str | None = None,
        random_state: int | None = 42,
    ) -> TuneResult:
        """Cross-validate over ``space`` (default: ``param_space``) on the training data.

        The model keeps the best parameters and is refitted on the full training set.
        """
        X, y = self._training_data(data, y)
        space = self.param_space if space is None else space
        best, result = tune(
            self.build_estimator, X, y, space, base_params=self.params, method=method, n_trials=n_trials,
            cv=cv, scoring=scoring or CV_SCORING[self.task], random_state=random_state,
        )
        self.fit_seconds_ = result.seconds
        self.params.update(result.best_params)
        self.estimator_ = best
        self.feature_names_ = list(X.columns) if isinstance(X, pd.DataFrame) else None
        self.tune_result_ = result
        return result

    # ------------------------------------------------------------------ inference

    def _require_fitted(self) -> BaseEstimator:
        if self.estimator_ is None:
            raise RuntimeError(f"{self!r} is not fitted yet; call fit() or tune() first.")
        return self.estimator_

    def predict(self, X) -> np.ndarray:
        """Predict from model-ready features (use ``DataSource.transform`` for raw rows)."""
        return self._require_fitted().predict(X)

    @property
    def feature_importances(self) -> dict[str, float] | None:
        """How much each column drove the predictions (sums to 1), for models that report it; else ``None``."""
        estimator = self._require_fitted()
        importances = getattr(estimator, "feature_importances_", None)
        if importances is None:
            return None
        names = self.feature_names_ or [f"x{i}" for i in range(len(importances))]
        return {n: float(v) for n, v in zip(names, importances)}

    def predict_proba(self, X) -> np.ndarray | None:
        estimator = self._require_fitted()
        return estimator.predict_proba(X) if hasattr(estimator, "predict_proba") else None

    def n_predictors(self, X) -> int:
        """Number of terms the model fits, for adjusted R². Models that expand features override this."""
        return int(np.shape(X)[1])

    def evaluate(self, data: DataSource | None = None, X_test=None, y_test=None) -> dict[str, float]:
        """Score on the test split of ``data`` (or the training DataSource), or on explicit ``X_test, y_test``."""
        if X_test is None or y_test is None:
            source = data or self._data
            if source is None:
                raise ValueError("Pass a DataSource or explicit X_test and y_test.")
            _, X_test, _, y_test = source.ensure_split()
            if not source.has_test_set:
                raise ValueError(
                    "The data has no test rows (it was split with test_size=0). "
                    "Score the training rows instead: evaluate(X_test=data.X_train, y_test=data.y_train)."
                )

        y_pred = self.predict(X_test)
        if self.task == "regression":
            return regression_metrics(y_test, y_pred, n_features=self.n_predictors(X_test))
        return classification_metrics(y_test, y_pred, self.predict_proba(X_test))
