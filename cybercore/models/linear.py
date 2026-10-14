"""Linear models."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from ..tuning import Categorical, Int
from .base import Model, register_model


class _LinearTerms:
    """Coefficient, intercept and equation helpers for models that end in ``LinearRegression``."""

    def _terms(self) -> tuple[list[str], np.ndarray, float]:
        estimator = self._require_fitted()
        names = self.feature_names_ or [f"x{i}" for i in range(len(estimator.coef_))]
        return list(names), estimator.coef_, float(estimator.intercept_)

    @property
    def coefficients(self) -> dict[str, float]:
        names, coefs, _ = self._terms()
        return {n: float(c) for n, c in zip(names, coefs)}

    @property
    def intercept(self) -> float:
        return self._terms()[2]

    def equation(self, target: str = "y", precision: int = 2) -> str:
        """The fitted model as text, e.g. ``Salary = 26,780.10 + 9,312.58 × YearsExperience``."""
        terms = [f"{self.intercept:,.{precision}f}"]
        for name, coef in self.coefficients.items():
            sign = "-" if coef < 0 else "+"
            terms.append(f"{sign} {abs(coef):,.{precision}f} × {name}")
        return f"{target} = " + " ".join(terms)


@register_model
class LinearRegressionModel(_LinearTerms, Model):
    """Ordinary least squares: ``y = b0 + b1*x1 + ... + bn*xn``.

    With one feature this is simple linear regression; with several it is
    multiple linear regression.
    """

    name = "linear_regression"
    label = "Linear Regression"
    task = "regression"
    description = "Ordinary least squares fit of a straight line (or hyperplane) to the features."
    default_params = {"fit_intercept": True}
    param_space = {
        "fit_intercept": Categorical([True, False]),
        "positive": Categorical([False, True]),
    }

    def build_estimator(self, **params):
        return LinearRegression(**params)

    def _variant_name(self) -> str:
        return "Simple linear regression" if self._n_input_features() == 1 else "Multiple linear regression"


@register_model
class PolynomialRegressionModel(_LinearTerms, Model):
    """Linear regression on polynomial terms: ``y = b0 + b1*x + b2*x² + ... + bd*x^d``.

    With several features, interaction terms (``x1*x2``) are included too.
    """

    name = "polynomial_regression"
    label = "Polynomial Regression"
    task = "regression"
    description = "Fits a curve by adding powers of the features (x², x³, …) to a linear regression."
    default_params = {"degree": 4}
    param_space = {"degree": Int(1, 6)}

    def build_estimator(self, degree: int = 4):
        # include_bias=False: LinearRegression's intercept already plays the constant term.
        return Pipeline([
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("linear", LinearRegression()),
        ])

    def _terms(self) -> tuple[list[str], np.ndarray, float]:
        pipeline = self._require_fitted()
        poly, linear = pipeline.named_steps["poly"], pipeline.named_steps["linear"]
        inputs = self.feature_names_ or [f"x{i}" for i in range(poly.n_features_in_)]
        names = [
            " × ".join(name if p == 1 else f"{name}^{p}" for name, p in zip(inputs, powers) if p)
            for powers in poly.powers_
        ]
        return names, linear.coef_, float(linear.intercept_)

    def _variant_name(self) -> str:
        return f"Polynomial regression, degree {self.params['degree']}"

    def n_predictors(self, X) -> int:
        return self._require_fitted().named_steps["poly"].n_output_features_
