"""Tree models."""

from __future__ import annotations

from sklearn.tree import DecisionTreeRegressor

from ..tuning import Categorical, Int
from .base import Model, register_model


@register_model
class DecisionTreeRegressionModel(Model):
    """A decision tree: splits the feature space into regions and predicts each region's mean.

    Without a depth limit it keeps splitting until every leaf is pure, which on small
    data means memorising the training rows.
    """

    name = "decision_tree_regression"
    label = "Decision Tree Regression"
    task = "regression"
    description = "Predicts the average of the training rows that fall in the same region of the features."
    # random_state fixes how ties between equally good splits are broken.
    default_params = {
        "max_depth": None, "min_samples_leaf": 1, "min_samples_split": 2, "criterion": "squared_error", "random_state": 0,
    }
    param_space = {
        "max_depth": Categorical([None, 1, 2, 3, 4, 5, 6, 8, 10, 15]),  # None: no limit
        "min_samples_leaf": Int(1, 10),
        "min_samples_split": Int(2, 10),
        "criterion": Categorical(["squared_error", "absolute_error"]),
    }

    def build_estimator(self, **params):
        return DecisionTreeRegressor(**params)

    def _variant_name(self) -> str:
        tree = self._require_fitted()
        return f"Decision tree regression, depth {tree.get_depth()}, {tree.get_n_leaves()} leaves"
