"""Support vector models."""

from __future__ import annotations

from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from ..tuning import Categorical, Float
from .base import Model, register_model


@register_model
class SVRModel(Model):
    """Support vector regression.

    SVR is distance-based and its ``epsilon`` margin is measured in target units, so
    both the features and the target are standardised inside the model. Predictions
    are converted back to the target's original units.
    """

    name = "svr"
    label = "Support Vector Regression"
    task = "regression"
    description = "Fits a smooth curve that keeps most points within a margin, using a kernel to bend it."
    default_params = {"kernel": "rbf", "C": 1.0, "epsilon": 0.1, "gamma": "scale"}
    param_space = {
        "kernel": Categorical(["rbf", "linear", "poly", "sigmoid"]),
        "C": Float(0.01, 1000, log=True),
        "epsilon": Float(0.001, 1.0, log=True),
        "gamma": Categorical(["scale", "auto"]),
    }

    def build_estimator(self, **params):
        return TransformedTargetRegressor(
            regressor=Pipeline([("scale", StandardScaler()), ("svr", SVR(**params))]),
            transformer=StandardScaler(),
        )

    def _variant_name(self) -> str:
        return f"Support vector regression, {self.params['kernel']} kernel"
