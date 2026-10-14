"""Model wrappers. Importing this package registers every built-in model."""

from .base import Model, available_models, create_model, register_model
from .linear import LinearRegressionModel, PolynomialRegressionModel
from .svm import SVRModel
from .tree import DecisionTreeRegressionModel
from .ensemble import RandomForestRegressionModel

__all__ = ["Model", "available_models", "create_model", "register_model", "LinearRegressionModel", "PolynomialRegressionModel", "SVRModel", "DecisionTreeRegressionModel", "RandomForestRegressionModel"]
