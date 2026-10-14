"""Unified tabular data wrapper.

``DataSource`` owns loading, cleaning and train/test splitting so that every
model benchmarked against the same data sees the same split. Preprocessing is
*configured* by :meth:`DataSource.preprocess` but only *fitted* on the training
rows inside :meth:`DataSource.split`, which keeps test-set statistics (medians,
category levels, scaling parameters) out of training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Sequence, Union

import numpy as np
import pandas as pd
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

Task = Literal["regression", "classification"]
DataInput = Union[str, Path, pd.DataFrame, pl.DataFrame]

# An integer target with at most this many distinct values is treated as class labels.
_MAX_CLASSES_FOR_INT_TARGET = 20

_READERS = {
    ".csv": pl.read_csv,
    ".parquet": pl.read_parquet,
    ".json": pl.read_json,
    ".ndjson": pl.read_ndjson,
}


def _load(data: DataInput) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, pl.DataFrame):
        return data.to_pandas()
    path = Path(data)
    reader = _READERS.get(path.suffix.lower())
    if reader is None:
        supported = ", ".join(sorted(_READERS))
        raise ValueError(f"Unsupported file type {path.suffix!r}; expected one of {supported}.")
    if not path.exists():
        raise FileNotFoundError(path)
    # Polars parses large files much faster than pandas; hand back pandas for scikit-learn.
    return reader(path).to_pandas()


def _feature_kind(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return "categorical"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    return "categorical"


def infer_task(target: pd.Series) -> Task:
    """Guess whether a target column calls for regression or classification."""
    values = target.dropna()
    if _feature_kind(values) != "numeric":
        return "classification"
    if pd.api.types.is_float_dtype(values) and not np.all(np.mod(values, 1) == 0):
        return "regression"
    if values.nunique() <= _MAX_CLASSES_FOR_INT_TARGET and values.nunique() < len(values) / 2:
        return "classification"
    return "regression"


class DataSource:
    """A dataset with a target column, ready for cleaning and splitting.

    Args:
        data: A ``.csv``/``.parquet``/``.json`` path, or a pandas/polars DataFrame.
        target_column: Name of the column to predict.
        features: Columns to use as features. Defaults to every other column.
        task: ``"regression"`` or ``"classification"``. Inferred from the target if omitted.
    """

    def __init__(
        self,
        data: DataInput,
        target_column: str,
        features: Sequence[str] | None = None,
        task: Task | None = None,
    ):
        frame = _load(data)
        if target_column not in frame.columns:
            raise KeyError(f"Target column {target_column!r} not found; columns are {list(frame.columns)}.")
        if features is None:
            features = [c for c in frame.columns if c != target_column]
        missing = [c for c in features if c not in frame.columns]
        if missing:
            raise KeyError(f"Feature columns not found: {missing}.")
        if target_column in features:
            raise ValueError("The target column cannot also be a feature.")

        self.frame: pd.DataFrame = frame[[*features, target_column]]
        self.target_column = target_column
        self.task: Task = task or infer_task(self.frame[target_column])
        self.feature_types: dict[str, str] = {c: _feature_kind(self.frame[c]) for c in features}

        self._preprocess_config = dict(impute_missing=True, encode_categorical=True, scale_numeric=False)
        self.preprocessor: ColumnTransformer | None = None
        self.label_encoder: LabelEncoder | None = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.X_train_raw = self.X_test_raw = None
        self._split_args: dict[str, Any] | None = None

    def __repr__(self) -> str:
        rows, cols = self.frame.shape
        state = "split" if self.is_split else "unsplit"
        return f"DataSource({rows} rows, {cols - 1} features, target={self.target_column!r}, task={self.task}, {state})"

    @classmethod
    def from_polars(cls, frame: pl.DataFrame, target_column: str, **kwargs) -> "DataSource":
        return cls(frame, target_column, **kwargs)

    def to_polars(self) -> pl.DataFrame:
        return pl.from_pandas(self.frame)

    @property
    def features(self) -> list[str]:
        return list(self.feature_types)

    @property
    def is_split(self) -> bool:
        return self.X_train is not None

    @property
    def classes(self) -> list | None:
        """Original class labels, for classification targets that were label-encoded."""
        return None if self.label_encoder is None else list(self.label_encoder.classes_)

    # ------------------------------------------------------------------ cleaning

    def preprocess(self, impute_missing: bool = True, encode_categorical: bool = True, scale_numeric: bool = False) -> "DataSource":
        """Configure the cleaning pipeline applied to features.

        Numeric columns are median-imputed (and optionally standardised); categorical
        columns are mode-imputed and one-hot encoded. Datetime columns are dropped.
        Rows with a missing target are always removed.

        The pipeline is fitted on the training split only. If the data is already
        split, the existing split is re-processed with the new settings.
        Returns ``self`` so calls can be chained: ``ds.preprocess().split()``.
        """
        self._preprocess_config = dict(
            impute_missing=impute_missing,
            encode_categorical=encode_categorical,
            scale_numeric=scale_numeric,
        )
        if self.is_split:
            self.split(**self._split_args)
        return self

    def _build_preprocessor(self) -> ColumnTransformer:
        cfg = self._preprocess_config
        numeric = [c for c, kind in self.feature_types.items() if kind == "numeric"]
        categorical = [c for c, kind in self.feature_types.items() if kind == "categorical"]

        numeric_steps = []
        if cfg["impute_missing"]:
            numeric_steps.append(("impute", SimpleImputer(strategy="median")))
        if cfg["scale_numeric"]:
            numeric_steps.append(("scale", StandardScaler()))

        categorical_steps = []
        if cfg["impute_missing"]:
            categorical_steps.append(("impute", SimpleImputer(strategy="most_frequent")))
        if cfg["encode_categorical"]:
            categorical_steps.append(("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)))

        transformers = []
        if numeric:
            transformers.append(("numeric", Pipeline(numeric_steps) if numeric_steps else "passthrough", numeric))
        if categorical:
            transformers.append(("categorical", Pipeline(categorical_steps) if categorical_steps else "passthrough", categorical))
        if not transformers:
            raise ValueError("No usable feature columns (only datetime columns were found).")

        return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False).set_output(transform="pandas")

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted cleaning pipeline to new raw feature rows."""
        if self.preprocessor is None:
            raise RuntimeError("Call split() before transform(); the pipeline is fitted on the training split.")
        return self.preprocessor.transform(X[self.features])

    # ------------------------------------------------------------------ splitting

    def split(self, test_size: float = 0.2, random_state: int | None = 42, stratify: bool = True) -> tuple:
        """Split into train/test sets, fit preprocessing on train, and cache the result.

        ``stratify`` only applies to classification targets where every class has
        at least two rows; it is ignored otherwise.

        ``test_size=0`` trains on every row and leaves the test set empty, for
        datasets too small to hold rows back. Scores are then training scores.

        Returns:
            ``(X_train, X_test, y_train, y_test)`` as preprocessed DataFrames/arrays.
        """
        self._split_args = dict(test_size=test_size, random_state=random_state, stratify=stratify)

        frame = self.frame.dropna(subset=[self.target_column])
        X = frame[self.features]
        y = frame[self.target_column]

        if self.task == "classification":
            self.label_encoder = LabelEncoder()
            y = pd.Series(self.label_encoder.fit_transform(y), index=y.index, name=y.name)
        else:
            self.label_encoder = None
            y = y.astype(float)

        stratify_on = None
        if stratify and self.task == "classification" and y.value_counts().min() >= 2:
            stratify_on = y

        if test_size == 0:
            X_train, X_test, y_train, y_test = X, X.iloc[:0], y, y.iloc[:0]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=stratify_on
            )

        self.preprocessor = self._build_preprocessor()
        self.X_train = self.preprocessor.fit_transform(X_train, y_train)
        # Imputers refuse zero-row input, so an empty test set takes the training columns.
        self.X_test = self.preprocessor.transform(X_test) if len(X_test) else self.X_train.iloc[:0]
        self.X_train_raw, self.X_test_raw = X_train, X_test
        self.y_train, self.y_test = y_train.to_numpy(), y_test.to_numpy()
        return self.X_train, self.X_test, self.y_train, self.y_test

    @property
    def has_test_set(self) -> bool:
        return self.X_test is not None and len(self.X_test) > 0

    def ensure_split(self) -> tuple:
        """Return the cached split, creating one with default settings if needed."""
        if not self.is_split:
            self.split()
        return self.X_train, self.X_test, self.y_train, self.y_test

    # ------------------------------------------------------------------ reporting

    def summary(self) -> dict:
        """Describe the dataset: shape, feature types, missingness, correlations and target."""
        frame = self.frame
        numeric_cols = [c for c in frame.columns if _feature_kind(frame[c]) == "numeric"]
        target = frame[self.target_column]

        if self.task == "regression":
            target_info = {k: float(v) for k, v in target.describe().items()}
        else:
            target_info = {str(k): int(v) for k, v in target.value_counts().items()}

        return {
            "shape": {"rows": int(frame.shape[0]), "columns": int(frame.shape[1])},
            "task": self.task,
            "target": self.target_column,
            "feature_types": dict(self.feature_types),
            "missing": {c: int(n) for c, n in frame.isna().sum().items()},
            "correlations": frame[numeric_cols].corr().round(4).to_dict() if len(numeric_cols) > 1 else {},
            "target_distribution": target_info,
        }
