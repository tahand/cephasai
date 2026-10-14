import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from cybercore import Benchmark, Categorical, DataSource, Float, Int, available_models, create_model, infer_task, load_example
from cybercore.datasets import example_path, example_split
from cybercore.metrics import classification_metrics
from cybercore.tuning import tune


@pytest.fixture
def salary():
    return load_example("salary")


@pytest.fixture
def mixed_frame():
    rng = np.random.default_rng(0)
    n = 60
    frame = pd.DataFrame({
        "age": rng.integers(20, 60, n).astype(float),
        "city": rng.choice(["Leeds", "York", "Hull"], n),
        "income": rng.normal(30_000, 5_000, n),
    })
    frame.loc[[1, 5, 9], "age"] = np.nan
    frame.loc[[2, 6], "city"] = None
    frame["spend"] = frame["income"] * 0.1 + rng.normal(0, 100, n)
    return frame


# ---------------------------------------------------------------- DataSource


def test_loads_csv_path_and_dataframes(salary):
    path = example_path("salary")
    from_path = DataSource(str(path), "Salary")
    from_pandas = DataSource(pd.read_csv(path), "Salary")
    from_polars = DataSource(pl.read_csv(path), "Salary")
    for ds in (salary, from_path, from_pandas, from_polars):
        assert ds.frame.shape == (30, 2)
        assert ds.features == ["YearsExperience"]
        assert ds.task == "regression"


def test_loads_parquet_and_json(tmp_path, salary):
    salary.to_polars().write_parquet(tmp_path / "s.parquet")
    salary.to_polars().write_json(tmp_path / "s.json")
    assert DataSource(tmp_path / "s.parquet", "Salary").frame.shape == (30, 2)
    assert DataSource(tmp_path / "s.json", "Salary").frame.shape == (30, 2)


def test_rejects_bad_inputs(tmp_path, salary):
    with pytest.raises(KeyError):
        DataSource(salary.frame, "Nope")
    with pytest.raises(ValueError):
        DataSource(tmp_path / "data.xlsx", "Salary")


def test_split_matches_plain_sklearn(salary):
    X_train, X_test, y_train, y_test = salary.split(test_size=1 / 3, random_state=0)
    raw = pd.read_csv(example_path("salary"))
    ref = train_test_split(raw[["YearsExperience"]], raw["Salary"], test_size=1 / 3, random_state=0)
    assert list(X_train.index) == list(ref[0].index)
    np.testing.assert_allclose(X_test["YearsExperience"], ref[1]["YearsExperience"])
    np.testing.assert_allclose(y_train, ref[2])
    assert len(X_test) == 10


def test_preprocessing_is_fitted_on_train_only(mixed_frame):
    ds = DataSource(mixed_frame, "spend").preprocess(scale_numeric=True)
    X_train, X_test, _, _ = ds.split(test_size=0.25, random_state=1)
    assert not X_train.isna().any().any() and not X_test.isna().any().any()
    assert {"city_Hull", "city_Leeds", "city_York"} <= set(X_train.columns)
    # Scaling statistics come from the training rows, so only train is exactly standardised.
    assert abs(X_train["income"].mean()) < 1e-9
    assert abs(X_test["income"].mean()) > 1e-6


def test_preprocess_after_split_reprocesses_same_rows(mixed_frame):
    ds = DataSource(mixed_frame, "spend")
    X_train, *_ = ds.split(random_state=3)
    ds.preprocess(scale_numeric=True)
    assert list(ds.X_train.index) == list(X_train.index)
    assert abs(ds.X_train["income"].mean()) < 1e-9


def test_classification_split_is_stratified_and_label_encoded():
    frame = pd.DataFrame({"x": range(40), "label": ["a"] * 30 + ["b"] * 10})
    ds = DataSource(frame, "label")
    assert ds.task == "classification"
    _, _, y_train, y_test = ds.split(test_size=0.2)
    assert ds.classes == ["a", "b"]
    assert sorted(np.bincount(y_test)) == [2, 6]


def test_infer_task():
    assert infer_task(pd.Series([1.5, 2.25, 3.0])) == "regression"
    assert infer_task(pd.Series(["yes", "no", "yes"])) == "classification"
    assert infer_task(pd.Series([0, 1, 0, 1, 1, 0])) == "classification"
    assert infer_task(pd.Series(range(100))) == "regression"


def test_summary(mixed_frame):
    summary = DataSource(mixed_frame, "spend").summary()
    assert summary["shape"] == {"rows": 60, "columns": 4}
    assert summary["feature_types"] == {"age": "numeric", "city": "categorical", "income": "numeric"}
    assert summary["missing"]["age"] == 3 and summary["missing"]["city"] == 2
    assert summary["correlations"]["income"]["spend"] > 0.9
    assert summary["target_distribution"]["count"] == 60


# ---------------------------------------------------------------- models


def test_linear_regression_matches_course_script(salary):
    salary.split(test_size=1 / 3, random_state=0)
    model = create_model("linear_regression").fit(salary)

    ref = LinearRegression().fit(salary.X_train.to_numpy(), salary.y_train)
    assert model.intercept == pytest.approx(ref.intercept_)
    assert model.coefficients["YearsExperience"] == pytest.approx(ref.coef_[0])
    np.testing.assert_allclose(model.predict(salary.X_test), ref.predict(salary.X_test.to_numpy()))

    metrics = model.evaluate()
    assert metrics["r2"] > 0.95
    assert set(metrics) == {"r2", "adjusted_r2", "rmse", "mae"}
    assert model.equation("Salary").startswith("Salary = 26,816.19 + 9,345.94 × YearsExperience")


def test_multiple_linear_regression_matches_course_script():
    startups = load_example("startups")
    assert startups.feature_types["State"] == "categorical"
    assert example_split("startups") == {"test_size": 0.2, "random_state": 0}
    X_train, X_test, _, y_test = startups.split(**example_split("startups"))
    assert {"State_California", "State_Florida", "State_New York"} <= set(X_train.columns)

    model = create_model("linear_regression").fit(startups)
    # Predictions printed by Section 5's multiple_linear_regression.py.
    course = [103015.20, 132582.28, 132447.74, 71976.10, 178537.48, 116161.24, 67851.69, 98791.73, 113969.44, 167921.07]
    np.testing.assert_allclose(model.predict(X_test), course, atol=0.01)
    assert model.evaluate()["r2"] == pytest.approx(0.934707, abs=1e-6)
    assert len(model.coefficients) == 6


def test_polynomial_regression_matches_course_script():
    positions = load_example("positions")
    assert positions.features == ["Level"]
    positions.split(**example_split("positions"))
    level_6_5 = positions.transform(pd.DataFrame({"Level": [6.5]}))

    poly = create_model("polynomial_regression").fit(positions)
    linear = create_model("linear_regression").fit(positions)
    # Values from Section 6's polynomial_regression.py: predict([[6.5]]) for each model.
    assert poly.predict(level_6_5)[0] == pytest.approx(158862.45265, abs=1e-3)
    assert linear.predict(level_6_5)[0] == pytest.approx(330378.78788, abs=1e-3)
    assert list(poly.coefficients) == ["Level", "Level^2", "Level^3", "Level^4"]

    result = Benchmark(positions, ["linear_regression", "polynomial_regression"]).run()
    assert result.scored_on == "train"
    assert result.best().model.name == "polynomial_regression"
    table = result.table.set_index("model")
    assert table.loc["Polynomial Regression", "r2"] == pytest.approx(0.997392, abs=1e-6)
    assert table.loc["Linear Regression", "r2"] == pytest.approx(0.669041, abs=1e-6)
    assert not any(c.startswith("train_") for c in table.columns)


def test_svr_matches_course_script():
    positions = load_example("positions")
    positions.split(**example_split("positions"))
    model = create_model("svr").fit(positions)

    # Section 7's support_vector_regression.py: sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))
    level_6_5 = positions.transform(pd.DataFrame({"Level": [6.5]}))
    assert model.predict(level_6_5)[0] == pytest.approx(170370.0204065, abs=1e-3)
    course_fit = [73474.16, 70786.95, 88213.01, 108254.99, 122574.52, 145503.11, 209410.23, 330606.89, 471671.96, 557821.76]
    np.testing.assert_allclose(model.predict(positions.X_train), course_fit, atol=0.01)
    assert model.evaluate(X_test=positions.X_train, y_test=positions.y_train)["r2"] == pytest.approx(0.751600, abs=1e-6)
    assert model.variant == "Support vector regression, rbf kernel (1 feature)"


def test_svr_scales_internally_whatever_the_data_settings(salary):
    # The DataSource's own scaling setting must not change SVR's predictions.
    raw = create_model("svr").fit(salary).predict(salary.transform(salary.frame[["YearsExperience"]]))
    salary.preprocess(scale_numeric=True)
    scaled = create_model("svr").fit(salary).predict(salary.transform(salary.frame[["YearsExperience"]]))
    np.testing.assert_allclose(raw, scaled, rtol=1e-6)
    assert raw.mean() > 10_000  # predictions come back in salary units


def test_svr_tunes_mixed_space():
    positions = load_example("positions")
    positions.split(test_size=0)
    model = create_model("svr")
    result = model.tune(positions, method="random", n_trials=10, cv=3)
    assert result.method == "random" and len(result.trials) == 10
    assert set(result.best_params) == {"kernel", "C", "epsilon", "gamma"}
    assert model.is_fitted and model.params["kernel"] == result.best_params["kernel"]


def test_decision_tree_matches_course_script():
    positions = load_example("positions")
    positions.split(**example_split("positions"))
    model = create_model("decision_tree_regression").fit(positions)

    # Section 8's decision_tree_regression.py: regressor.predict([[6.5]])
    assert model.predict(positions.transform(pd.DataFrame({"Level": [6.5]})))[0] == 150000.0
    # An unlimited tree memorises 10 rows: one leaf each, a perfect training score.
    assert model.evaluate(X_test=positions.X_train, y_test=positions.y_train)["r2"] == 1.0
    assert model.variant == "Decision tree regression, depth 6, 10 leaves (1 feature)"
    assert model.feature_importances == {"Level": 1.0}

    shallow = create_model("decision_tree_regression", max_depth=2).fit(positions)
    assert shallow.variant.startswith("Decision tree regression, depth 2, 4 leaves")


def test_tree_tuning_handles_unlimited_depth_and_importances():
    startups = load_example("startups")
    model = create_model("decision_tree_regression")
    result = model.tune(startups, space={"max_depth": Categorical([None, 2, 4])}, method="grid", cv=3)
    assert list(result.trials["max_depth"]) == [None, 2, 4]
    importances = model.feature_importances
    assert set(importances) == set(startups.X_train.columns)
    assert sum(importances.values()) == pytest.approx(1.0)
    assert max(importances, key=importances.get) == "R&D Spend"
    assert create_model("linear_regression").fit(startups).feature_importances is None


def test_split_without_test_set():
    positions = load_example("positions")
    X_train, X_test, y_train, y_test = positions.split(test_size=0)
    assert len(X_train) == 10 and len(X_test) == 0 and len(y_test) == 0
    assert list(X_test.columns) == list(X_train.columns)
    assert not positions.has_test_set
    model = create_model("linear_regression").fit(positions)
    with pytest.raises(ValueError, match="no test rows"):
        model.evaluate()
    assert model.evaluate(X_test=X_train, y_test=y_train)["r2"] > 0.6


def test_polynomial_term_names_and_adjusted_r2():
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"R&D Spend": rng.normal(size=40), "Admin": rng.normal(size=40)})
    frame["y"] = frame["R&D Spend"] ** 2 + frame["Admin"] + rng.normal(0, 0.1, 40)
    data = DataSource(frame, "y")
    data.split(test_size=0)
    model = create_model("polynomial_regression", degree=2).fit(data)
    assert list(model.coefficients) == ["R&D Spend", "Admin", "R&D Spend^2", "R&D Spend × Admin", "Admin^2"]
    assert model.coefficients["R&D Spend^2"] == pytest.approx(1.0, abs=0.05)
    assert "× R&D Spend × Admin" in model.equation()
    # Adjusted R² counts the 5 expanded terms, not the 2 input columns.
    m = model.evaluate(X_test=data.X_train, y_test=data.y_train)
    assert m["adjusted_r2"] == pytest.approx(1 - (1 - m["r2"]) * 39 / (40 - 5 - 1))


def test_variant_describes_the_fitted_form(salary):
    model = create_model("linear_regression")
    assert model.variant == "Linear Regression"  # unfitted
    assert model.fit(salary).variant == "Simple linear regression (1 feature)"

    startups = load_example("startups")
    assert create_model("linear_regression").fit(startups).variant == (
        "Multiple linear regression (4 features, 6 columns after encoding)"
    )
    X_train, _, y_train, _ = startups.ensure_split()
    assert create_model("linear_regression").fit(X_train.to_numpy(), y_train).variant == (
        "Multiple linear regression (6 features)"
    )

    positions = load_example("positions")
    positions.split(test_size=0)
    assert create_model("polynomial_regression").fit(positions).variant == "Polynomial regression, degree 4 (1 feature)"
    table = Benchmark(startups, ["linear_regression"]).run().table
    assert table.loc[0, "fitted_as"].startswith("Multiple linear regression")


def test_model_accepts_explicit_arrays_and_test_sets(salary):
    X_train, X_test, y_train, y_test = salary.split()
    model = create_model("linear_regression").fit(X_train.to_numpy(), y_train)
    assert model.coefficients.keys() == {"x0"}
    assert model.evaluate(X_test=X_test.to_numpy(), y_test=y_test)["r2"] > 0.9
    with pytest.raises(ValueError):
        model.fit(X_train)


def test_model_rejects_wrong_task():
    ds = DataSource(pd.DataFrame({"x": range(10), "label": ["a", "b"] * 5}), "label")
    with pytest.raises(ValueError, match="regression model"):
        create_model("linear_regression").fit(ds)


def test_unfitted_model_and_unknown_name():
    with pytest.raises(RuntimeError):
        create_model("linear_regression").predict([[1.0]])
    with pytest.raises(KeyError):
        create_model("not_a_model")


def test_registry():
    assert {"linear_regression", "polynomial_regression", "svr", "decision_tree_regression"} <= set(available_models("regression"))
    assert "linear_regression" not in available_models("classification")


# ---------------------------------------------------------------- tuning


def test_model_tune_picks_grid_for_small_space(salary):
    model = create_model("linear_regression")
    result = model.tune(salary)
    assert result.method == "grid" and result.n_trials == 4
    assert model.params["fit_intercept"] is True  # a salary line needs its intercept
    assert model.is_fitted and model.evaluate()["r2"] > 0.9


@pytest.mark.parametrize("method", ["grid", "random", "optuna"])
def test_tune_methods_over_mixed_space(method):
    from sklearn.linear_model import Ridge

    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 3))
    y = X @ [1.0, -2.0, 0.5] + rng.normal(0, 0.1, 80)
    space = {"alpha": Float(1e-3, 10, log=True), "max_iter": Int(100, 1000), "fit_intercept": [True, False]}
    best, result = tune(Ridge, X, y, space, method=method, n_trials=8, cv=3)
    assert result.best_score > 0.9
    assert set(result.best_params) == set(space)
    assert len(result.trials) == result.n_trials
    assert best.predict(X).shape == (80,)


def test_tune_searches_model_params_not_estimator_params(salary):
    # degree belongs to the model, not to the sklearn pipeline it builds.
    model = create_model("polynomial_regression")
    result = model.tune(salary)
    assert result.method == "grid" and sorted(result.trials["degree"]) == [1, 2, 3, 4, 5, 6]
    assert model.params == {"degree": result.best_params["degree"]}
    model.fit(salary)  # params remain valid for a plain refit


def test_categorical_space_value():
    assert Categorical([1, 2]).choices == (1, 2)


# ---------------------------------------------------------------- metrics & benchmark


def test_classification_metrics():
    y = np.array([0, 1, 1, 0])
    proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.4, 0.6], [0.45, 0.55]])
    m = classification_metrics(y, proba.argmax(1), proba)
    assert m["accuracy"] == 0.75 and m["roc_auc"] == 1.0


def test_benchmark(salary):
    salary.split(test_size=1 / 3, random_state=0)

    class Broken:
        name = label = "broken"
        param_space = {}

        def fit(self, data):
            raise RuntimeError("boom")

    result = Benchmark(salary, ["linear_regression", Broken()]).run()
    table = result.table
    assert list(table["model"]) == ["Linear Regression", "broken"]
    assert table.loc[0, "r2"] > 0.95 and table.loc[0, "train_r2"] > 0.9
    assert table.loc[1, "error"] == "RuntimeError: boom"
    assert result.best().model.name == "linear_regression"
    assert result.best("rmse").model.name == "linear_regression"


def test_benchmark_defaults_to_registered_models(salary):
    result = Benchmark(salary, tune=True).run()
    assert [e.model.name for e in result.entries] == list(available_models("regression"))
    assert result.entries[0].model.tune_result_ is not None


def test_random_forest_matches_course_script():
    positions = load_example("positions")
    positions.split(**example_split("positions"))
    model = create_model("random_forest_regression").fit(positions)

    # Section 9's random_forest_regression.py: RandomForestRegressor(n_estimators=10, random_state=0).predict([[6.5]])
    assert model.predict(positions.transform(pd.DataFrame({"Level": [6.5]})))[0] == pytest.approx(167000.0)
    # Bootstrap samples leave rows out of each tree, so unlike a single tree the forest doesn't memorise all 10.
    assert model.evaluate(X_test=positions.X_train, y_test=positions.y_train)["r2"] < 1.0
    assert model.variant.startswith("Random forest regression, 10 trees")
    assert model.feature_importances == {"Level": 1.0}


def test_random_forest_tuning_and_importances():
    startups = load_example("startups")
    model = create_model("random_forest_regression")
    result = model.tune(startups, space={"max_features": Categorical([1.0, "sqrt"]), "n_estimators": Categorical([10, 50])}, method="grid", cv=3)
    assert len(result.trials) == 4
    importances = model.feature_importances
    assert sum(importances.values()) == pytest.approx(1.0)
    assert max(importances, key=importances.get) == "R&D Spend"
