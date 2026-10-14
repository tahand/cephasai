"""Ensemble models."""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor

from ..tuning import Categorical, Int
from .base import Model, register_model


@register_model
class RandomForestRegressionModel(Model):
    """A random forest: many decision trees, each grown on a bootstrap sample of the rows, averaged.

    Averaging smooths out any one tree's quirks, so the forest's steps are finer than a
    single tree's and it overfits less, though it can still fit small data closely.
    """

    name = "random_forest_regression"
    label = "Random Forest Regression"
    task = "regression"
    description = "Averages the predictions of many decision trees, each trained on a random sample of the rows."
    # random_state fixes the bootstrap samples, so results are repeatable.
    default_params = {
        "n_estimators": 10, "max_depth": None, "min_samples_leaf": 1, "max_features": 1.0, "random_state": 0,
    }
    param_space = {
        "n_estimators": Int(5, 300, log=True),
        "max_depth": Categorical([None, 2, 3, 4, 5, 6, 8, 10, 15]),  # None: no limit
        "min_samples_leaf": Int(1, 10),
        "max_features": Categorical([1.0, "sqrt", 0.5]),  # share of columns each split may consider
    }

    def build_estimator(self, **params):
        return RandomForestRegressor(**params)

    def _variant_name(self) -> str:
        forest = self._require_fitted()
        depths = [tree.get_depth() for tree in forest.estimators_]
        return f"Random forest regression, {len(depths)} trees, depth up to {max(depths)}"
