# cephasai

Plug and play AI model wrappers for tabular machine learning.

- **`cybercore`** is a Python library that loads a dataset, cleans and splits it, then trains, tunes, scores and compares models with a few lines of code.
- **The showcase** (`showcase/app.py`) is a Streamlit web app built on `cybercore`. You use it to pick a dataset and models, compare their scores, view charts, and make predictions without writing code.

**Contents:** [Requirements](#requirements) · [First-time setup](#first-time-setup) · [Running the showcase](#running-the-showcase) · [Using the showcase](#using-the-showcase) · [Using the library](#using-the-library) · [Example datasets](#example-datasets) · [Registered models](#registered-models) · [Running the tests](#running-the-tests) · [Project layout](#project-layout) · [How it works](#how-it-works) · [Adding a model](#adding-a-model) · [Troubleshooting](#troubleshooting)

---

## Requirements

- **Python 3.10 or newer.** It's developed and tested on Python 3.14. Check your version with `python --version`.
- **Git**, to clone the repository.
- Windows, macOS or Linux. The commands below are for **Windows PowerShell**. On macOS or Linux, replace `.venv\Scripts\` with `.venv/bin/`.

Every other dependency (pandas, Polars, scikit-learn, Streamlit and so on) is installed into a private virtual environment by the setup steps.

## First-time setup

Run these once, from a terminal in the folder you cloned into:

```powershell
git clone https://github.com/tahand/cephasai.git
cd cephasai

# 1. Create a virtual environment in .venv, so packages don't touch your system Python
python -m venv .venv

# 2. Install cybercore plus the showcase, tuning and test tools
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e ".[dev]"

# 3. Check everything works (all tests should pass)
.venv\Scripts\python -m pytest
```

`-e` installs `cybercore` in *editable* mode, so changes to the code take effect without reinstalling. The part in brackets chooses optional extras:

| Install command | What you get |
|---|---|
| `pip install -e .` | The library only |
| `pip install -e ".[tuning]"` | The library plus Optuna, for `method="optuna"` tuning |
| `pip install -e ".[showcase]"` | The library plus Streamlit and Altair, for the web app |
| `pip install -e ".[dev]"` | Everything above, plus pytest |

### Optional: activate the environment

Activating the environment lets you type `python`, `pip`, `pytest` and `streamlit` without the `.venv\Scripts\` prefix, for the rest of that terminal session:

```powershell
.venv\Scripts\Activate.ps1
```

Your prompt then starts with `(.venv)`. Type `deactivate` to leave it. If PowerShell refuses with "running scripts is disabled on this system", run this once and try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Running the showcase

From the `cephasai` folder:

```powershell
.venv\Scripts\streamlit run showcase/app.py
```

If you've activated the environment, `streamlit run showcase/app.py` is enough.

The app opens in your browser at **http://localhost:8501**. If it doesn't, open that address yourself. The terminal stays busy while the app runs. Press **Ctrl+C** in the terminal to stop it.

| Situation | What to do |
|---|---|
| You edited `showcase/app.py` | Refresh the browser page, or click **Rerun** at the top right |
| You edited anything in `cybercore/` | Stop the app with Ctrl+C and start it again. The running app keeps the old library loaded |
| Port 8501 is already in use | `.venv\Scripts\streamlit run showcase/app.py --server.port 8502` |
| You don't want a browser to open | Add `--server.headless true` |

In Claude Code, you can type `! .venv\Scripts\streamlit run showcase/app.py` at the prompt to run it from the session.

## Using the showcase

Everything is controlled from the **sidebar** on the left. Results update as soon as you change a setting.

### Sidebar

1. **Data.** Pick a bundled **Example dataset** ([listed below](#example-datasets)), or choose **Upload a file** and drop in a CSV, Parquet or JSON file.
2. **Target column.** The column to predict. Examples preselect the right one; uploads default to the last column.
3. **Features.** The columns used to make the prediction. Examples start with the course's choice, e.g. `positions` uses only `Level`.
4. **Task.** `auto-detect` decides between regression (predicting a number) and classification (predicting a category) from the target column. Override it if the guess is wrong.
5. **Preprocessing.**
   - *Impute missing values* fills gaps with the median (numbers) or most common value (categories).
   - *One-hot encode categoricals* turns text columns such as `State` into 0/1 columns.
   - *Standardise numeric features* rescales numbers to mean 0 and standard deviation 1.
6. **Split.**
   - *Test size* is the share of rows held back to test each model on data it hasn't seen. *None* trains on every row, for datasets too small to split. Scores are then training scores, and the app shows a warning.
   - *Random state* fixes which rows are held back, so results are repeatable.
   - Examples start with the split their course script used, so the numbers match the course.
7. **Models.** Choose which models to compare. Each one has a **settings** panel underneath with its own parameters, e.g. the polynomial *Degree* or the decision tree's *Max depth*, where *No limit* lets the tree grow until it fits every training row. Random Forest Regression adds *N estimators* (how many trees are averaged) and *Max features* (the share of columns each split may consider; `sqrt` means the square root of the column count).
8. **Tune hyperparameters.** Instead of using the settings, search for the best parameters with cross-validation:
   - *Search* chooses the method. *auto* uses a full grid when it's small enough, otherwise random search; *optuna* uses smarter Bayesian search.
   - *Trials* caps how many parameter combinations are tried.
   - *CV folds* sets how many ways the training rows are divided for scoring.

### Tabs

- **Data.** Row and feature counts, missing values, a column summary, a correlation heatmap (red is positive, blue is negative), and a preview of the rows.
- **Benchmark.**
  - A results table ranking every model by R² (regression) or accuracy (classification), with RMSE, MAE, adjusted R² and timings. Its *fitted_as* column says what form each model took on this data, e.g. *Multiple linear regression (4 features)*.
  - A train-vs-test score chart. A big gap between the two means the model has memorised its training rows rather than learned a general pattern (overfitting).
  - For single-feature data, a **Fitted models** chart overlaying every model's curve on the data.
- **Model detail.** Pick one model to see:
  - **Fitted as**: the form the model took on this data. Linear Regression shows as *Simple linear regression (1 feature)* or *Multiple linear regression (N features)* depending on the features you've chosen
  - its scores and, for linear models, its fitted equation
  - **Predict a new value**: type in feature values to get a prediction (e.g. `Level` 6.5 on `positions`)
  - the fitted line or curve against the data
  - predicted-vs-actual and residual charts
  - a table of individual predictions
  - **Feature importance**, for tree and forest models with more than one column: how much each column drove the decisions
  - the tuning trials, if tuning is on

### Using your own data

Choose **Upload a file** in the sidebar. The file needs a header row, one row per example and one column per variable. Text columns are treated as categories and one-hot encoded; date columns are ignored. Then set the target, features and split as above.

## Using the library

The examples below can be run in a Python session started with `.venv\Scripts\python`, or in a script.

### Quick start

```python
import cybercore as cc

data = cc.load_example("salary")
data.split(test_size=1/3, random_state=0)

model = cc.create_model("linear_regression").fit(data)
print(model.equation("Salary"))   # Salary = 26,816.19 + 9,345.94 × YearsExperience
print(model.evaluate())           # {'r2': 0.9749..., 'rmse': ..., 'mae': ..., 'adjusted_r2': ...}
```

### Loading data

```python
data = cc.DataSource("my_data.csv", target_column="Price")                 # .csv, .parquet, .json
data = cc.DataSource(df, target_column="Price")                            # a pandas or Polars DataFrame
data = cc.DataSource(df, target_column="Price", features=["Size", "Age"])  # choose the feature columns
data = cc.DataSource(df, target_column="Churned", task="classification")   # override the detected task

data.summary()   # shape, feature types, missing counts, correlations, target distribution
```

### Cleaning and splitting

```python
data.preprocess(impute_missing=True, encode_categorical=True, scale_numeric=False)
X_train, X_test, y_train, y_test = data.split(test_size=0.2, random_state=42, stratify=True)
```

`preprocess()` only records the settings. The cleaning is learned from the training rows inside `split()` and then applied to the test rows, so nothing about the test rows leaks into training. `stratify` keeps class proportions equal in both halves (classification only). `split(test_size=0)` trains on every row; see [Small datasets](#small-datasets).

### Training, scoring and predicting

```python
model = cc.create_model("polynomial_regression", degree=3)   # parameters override the defaults
model.fit(data)                     # trains on data's training rows
model.evaluate()                    # scores on data's test rows
model.evaluate(X_test=X, y_test=y)  # ...or on any rows you choose
model.coefficients                  # {'Level': ..., 'Level^2': ..., 'Level^3': ...} (linear models)
model.variant                       # 'Polynomial regression, degree 3 (1 feature)'
model.feature_importances           # {'R&D Spend': 0.98, ...} for tree models; None otherwise

# Predicting new rows: pass raw columns through the same cleaning first
import pandas as pd
new_rows = pd.DataFrame({"Level": [6.5]})
model.predict(data.transform(new_rows))
```

You can also skip `DataSource` entirely: `model.fit(X_train, y_train)`.

### Tuning

```python
result = model.tune(data, method="auto", n_trials=20, cv=5)
result.best_params     # e.g. {"degree": 6}
result.best_score      # mean cross-validated R² (regression) or accuracy (classification)
result.trials          # a DataFrame of every combination tried
```

The model keeps the best parameters and is refitted on all training rows. To search a custom space, pass `space`:

```python
from cybercore import Int, Float, Categorical
model.tune(data, space={"degree": Int(1, 8)}, method="grid")
```

### Comparing models

```python
result = cc.Benchmark(data, ["linear_regression", "polynomial_regression"]).run()
result.table           # one row per model, best first
result.best().model    # the winning model, already fitted
cc.Benchmark(data, tune=True, n_trials=30).run()   # every registered model for the task, tuned
```

### Small datasets

The course trains on all 10 rows of the `positions` dataset rather than holding any back:

```python
data = cc.load_example("positions")
data.split(test_size=0)
result = cc.Benchmark(data).run()
result.scored_on       # 'train': scores are training scores
```

Training scores show how well a model *fits* the data, not how well it predicts new data. `model.evaluate()` raises an error on unsplit data, so you score the training rows deliberately: `model.evaluate(X_test=data.X_train, y_test=data.y_train)`.

## Example datasets

Bundled with the library. Use `cc.load_example(name)`, or pick them in the showcase. `cybercore.datasets.example_split(name)` returns the split the course used.

| Name | Rows | Target | Features | Course split | From |
|---|---|---|---|---|---|
| `salary` | 30 | `Salary` | `YearsExperience` | 1/3 test, random state 0 | ML A-Z Part 2 §4 |
| `startups` | 50 | `Profit` | R&D, Administration and Marketing spend; `State` | 20% test, random state 0 | ML A-Z Part 2 §5 |
| `positions` | 10 | `Salary` | `Level` | none (train on all rows) | ML A-Z Part 2 §6, §7, §8 and §9 |

## Registered models

`cc.available_models()` lists these in code.

| Name | Task | Parameters | Source |
|---|---|---|---|
| `linear_regression` | regression | `fit_intercept`, `positive` | Simple (§4) and multiple (§5) linear regression. One model covers both: it's simple with one feature and multiple with several |
| `polynomial_regression` | regression | `degree` (default 4) | Polynomial regression (§6) |
| `svr` | regression | `kernel` (default `rbf`), `C`, `epsilon`, `gamma` | Support vector regression (§7). Scales the features and target itself, so predictions are in the target's own units |
| `decision_tree_regression` | regression | `max_depth` (default no limit), `min_samples_leaf`, `min_samples_split`, `criterion` | Decision tree regression (§8). Unlimited depth memorises small datasets (training R² 1.0); limit the depth to generalise |
| `random_forest_regression` | regression | `n_estimators` (default 10), `max_depth` (default no limit), `min_samples_leaf`, `max_features` | Random forest regression (§9). Averages many trees trained on random samples of the rows, so its steps are finer than one tree's and it overfits less |

## Running the tests

```powershell
.venv\Scripts\python -m pytest            # everything
.venv\Scripts\python -m pytest -k poly    # only tests whose names contain "poly"
```

Each course model has a test that checks `cybercore` reproduces the numbers printed by the course script.

## Project layout

```
cybercore/
  data.py          DataSource: loading, cleaning, splitting, summaries
  models/
    base.py        Model base class and the model registry
    linear.py      Linear and polynomial regression
    svm.py         Support vector regression
    tree.py        Decision tree regression
    ensemble.py    Random forest regression
  tuning.py        Search spaces (Int, Float, Categorical) and grid/random/Optuna search
  metrics.py       Regression and classification metrics
  benchmark.py     Benchmark: fit and rank several models on one split
  datasets/        Bundled example CSVs and load_example()
showcase/app.py    The Streamlit web app
tests/             pytest suite
pyproject.toml     Package metadata and dependencies
```

## How it works

- **`DataSource`** owns the data and the train/test split. The split is cached, so every model in a comparison sees exactly the same rows. Preprocessing is fitted on the training rows only, so the test rows give an honest score.
- **`Model`** wraps a scikit-learn estimator behind a named, registered class. Its parameters belong to the model, which can turn them into whatever estimator it needs. For example, the polynomial model's `degree` configures a pipeline step.
- **Tuning** describes parameter ranges once (`Int`, `Float`, `Categorical`) and runs grid, random or Optuna search over them. It uses shuffled cross-validation on the training rows only.
- **`Benchmark`** fits each model on the shared split and ranks them by test R² (regression) or accuracy (classification). It keeps training scores alongside so overfitting is visible.

## Adding a model

Each model is one registered subclass. It then appears automatically in `available_models()`, `Benchmark` and the showcase, which builds the model's sidebar settings from its `param_space`.

```python
# cybercore/models/neighbors.py
from sklearn.neighbors import KNeighborsRegressor

from ..tuning import Categorical, Int
from .base import Model, register_model


@register_model
class KNNRegressionModel(Model):
    name = "knn_regression"
    label = "K-Nearest Neighbours Regression"
    task = "regression"
    default_params = {"n_neighbors": 5, "weights": "uniform"}
    param_space = {"n_neighbors": Int(1, 20), "weights": Categorical(["uniform", "distance"])}

    def build_estimator(self, **params):
        return KNeighborsRegressor(**params)
```

If the model needs special handling, put it in `build_estimator` so it always happens. For example, `SVRModel` wraps SVR in scalers for the features and target. Override `_variant_name()` to control the "Fitted as" description.

Checklist for a new model:

1. Add the class under `cybercore/models/` and import it in `cybercore/models/__init__.py`.
2. If it comes with a dataset, add the CSV to `cybercore/datasets/` and an entry in `cybercore/datasets/__init__.py`.
3. Add a test that reproduces the source script's numbers.
4. Check it in the showcase: it can be selected, its settings work, and its charts make sense.
5. Add it to the [Registered models](#registered-models) table above.

## Troubleshooting

| Problem | Fix |
|---|---|
| `streamlit` / `pytest` is "not recognized" | Use the full path, `.venv\Scripts\streamlit ...`, or [activate the environment](#optional-activate-the-environment) first |
| "running scripts is disabled on this system" when activating | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, once |
| `ModuleNotFoundError: No module named 'cybercore'` | Run `.venv\Scripts\python -m pip install -e ".[dev]"` from the `cephasai` folder |
| A library change doesn't show in the app | Stop the app with Ctrl+C and start it again |
| "Port 8501 is already in use" | Another copy is still running. Close it, or add `--server.port 8502` |
| `method='optuna' needs Optuna` | `.venv\Scripts\python -m pip install -e ".[tuning]"` |
| Tuning shows "each cross-validation fold scores only about N rows" | The dataset is too small for that many folds. Lower *CV folds*, or treat the tuning scores as rough |
