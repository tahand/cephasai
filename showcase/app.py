"""CyberCore showcase: benchmark registered models against a dataset.

Run with:  streamlit run showcase/app.py
"""

from __future__ import annotations

import io
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import polars as pl
import streamlit as st
from sklearn.metrics import confusion_matrix

import cybercore as cc
from cybercore.datasets import example_features, example_path, example_split, example_target
from cybercore.tuning import Categorical, Float, Int
from cybercore.metrics import HIGHER_IS_BETTER

st.set_page_config(page_title="CyberCore Showcase", layout="wide")

# ---------------------------------------------------------------- chart styling

_PALETTES = {
    "light": dict(
        train="#2a78d6", test="#eb6834", ink="#0b0b0b", ink2="#52514e", muted="#898781",
        grid="#e1e0d9", axis="#c3c2b7", neg="#2a78d6", mid="#f0efec", pos="#e34948",
        seq=["#cde2fb", "#2a78d6", "#0d366b"],
        series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
    ),
    "dark": dict(
        train="#3987e5", test="#d95926", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
        grid="#2c2c2a", axis="#383835", neg="#3987e5", mid="#383835", pos="#e66767",
        seq=["#184f95", "#3987e5", "#cde2fb"],
        series=["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
    ),
}


def _theme_mode() -> str:
    theme = getattr(st.context, "theme", None)
    return "dark" if getattr(theme, "type", None) == "dark" else "light"


C = _PALETTES[_theme_mode()]
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
# Points per fitted curve: fine enough that a tree's flat steps draw with near-vertical jumps.
CURVE_POINTS = 1000


def show(chart: alt.Chart, height: int) -> None:
    chart = (
        chart.properties(height=height)
        .configure(font=FONT, background="transparent")
        .configure_view(stroke=None)
        .configure_axis(
            gridColor=C["grid"], domainColor=C["axis"], tickColor=C["axis"],
            labelColor=C["muted"], titleColor=C["ink2"], labelFontSize=11, titleFontSize=12, titleFontWeight="normal",
        )
        .configure_legend(labelColor=C["ink2"], titleColor=C["ink2"], orient="top", title=None)
        .configure_title(color=C["ink"], anchor="start", fontSize=13, fontWeight=600)
    )
    st.altair_chart(chart, theme=None, width="stretch", height=height)


def _split_color(splits: list[str]) -> alt.Color:
    """Train/test colouring that only lists the splits actually present."""
    palette = {"Train": C["train"], "Test": C["test"]}
    return alt.Color("split:N", scale=alt.Scale(domain=splits, range=[palette[s] for s in splits]))


# ---------------------------------------------------------------- sidebar: data


def _read_upload(upload) -> pd.DataFrame:
    suffix = Path(upload.name).suffix.lower()
    buffer = io.BytesIO(upload.getvalue())
    readers = {".csv": pl.read_csv, ".parquet": pl.read_parquet, ".json": pl.read_json}
    return readers[suffix](buffer).to_pandas()


with st.sidebar:
    st.header("Data")
    source = st.radio("Source", ["Example dataset", "Upload a file"], horizontal=True, label_visibility="collapsed")
    if source == "Example dataset":
        examples = cc.list_examples()
        name = st.selectbox("Dataset", list(examples), format_func=lambda n: n.replace("_", " ").title())
        st.caption(examples[name])
        frame = pl.read_csv(example_path(name)).to_pandas()
        split_defaults = example_split(name)
        default_target, default_features = example_target(name), example_features(name)
    else:
        upload = st.file_uploader("CSV, Parquet or JSON", type=["csv", "parquet", "json"])
        if upload is None:
            st.info("Upload a file to begin.")
            st.stop()
        frame = _read_upload(upload)
        name, split_defaults = upload.name, {"test_size": 0.2, "random_state": 42}
        default_target, default_features = frame.columns[-1], None

    columns = list(frame.columns)
    target = st.selectbox("Target column", columns, index=columns.index(default_target), key=f"target_{name}")
    candidates = [c for c in columns if c != target]
    features = st.multiselect(
        "Features", candidates, default=[c for c in (default_features or candidates) if c in candidates],
        key=f"features_{name}_{target}",
    )
    task_choice = st.selectbox("Task", ["auto-detect", "regression", "classification"])

    st.header("Preprocessing")
    impute = st.checkbox("Impute missing values", value=True)
    encode = st.checkbox("One-hot encode categoricals", value=True)
    scale = st.checkbox("Standardise numeric features", value=False)

    st.header("Split")
    # Keyed by dataset so each example starts from the split its course script used.
    test_size = st.select_slider(
        "Test size", options=[0, 0.1, 0.2, 0.25, 1 / 3, 0.4, 0.5], value=split_defaults["test_size"],
        format_func=lambda v: f"{v:.0%}" if v else "None", key=f"test_size_{name}",
        help="None trains on every row, for datasets too small to hold rows back. Scores are then training scores.",
    )
    seed = st.number_input("Random state", value=split_defaults["random_state"], step=1, key=f"seed_{name}")
    stratify = st.checkbox("Stratify (classification)", value=True)

def _param_controls(model_cls, disabled: bool) -> dict:
    """Sidebar inputs for a model's tunable parameters, generated from its ``param_space``."""
    if not model_cls.param_space:
        return {}
    params = {}
    with st.expander(f"{model_cls.label} settings"):
        for pname, dim in model_cls.param_space.items():
            key = f"param_{model_cls.name}_{pname}"
            label = pname.replace("_", " ").capitalize()
            default = model_cls.default_params.get(pname)
            if isinstance(dim, (list, tuple)):
                dim = Categorical(dim)
            if isinstance(dim, Categorical):
                choices = list(dim.choices)
                index = choices.index(default) if default in choices else 0
                params[pname] = st.selectbox(
                    label, choices, index=index, key=key, disabled=disabled,
                    format_func=lambda v: "No limit" if v is None else str(v),
                )
            elif isinstance(dim, Int):
                value = default if default is not None else dim.low
                params[pname] = int(st.number_input(label, dim.low, dim.high, value, step=1, key=key, disabled=disabled))
            elif isinstance(dim, Float):
                value = float(default if default is not None else dim.low)
                params[pname] = st.number_input(label, float(dim.low), float(dim.high), value, key=key, disabled=disabled, format="%g")
    return params


try:
    data = cc.DataSource(frame, target, features=features, task=None if task_choice == "auto-detect" else task_choice)
    data.preprocess(impute_missing=impute, encode_categorical=encode, scale_numeric=scale)
    data.split(test_size=test_size, random_state=int(seed), stratify=stratify)
except Exception as exc:
    st.error(f"Could not prepare the data: {exc}")
    st.stop()

with st.sidebar:
    st.header("Models")
    registry = cc.available_models(data.task)
    if not registry:
        st.warning(f"No {data.task} models are registered yet.")
    chosen = st.multiselect("Models", list(registry), default=list(registry), format_func=lambda n: registry[n].label)
    do_tune = st.checkbox("Tune hyperparameters", value=False, help="Search each model's parameters with cross-validation instead of using the settings below.")
    if do_tune:
        method = st.selectbox("Search", ["auto", "grid", "random", "optuna"])
        n_trials = st.slider("Trials", 4, 100, 20)
        folds = st.slider("CV folds", 2, 10, 5)
    model_params = {n: _param_controls(registry[n], disabled=do_tune) for n in chosen}

# ---------------------------------------------------------------- run

st.title("CyberCore Showcase")
st.caption(f"{data!r}")

tune_kwargs = dict(method=method, n_trials=n_trials, cv=folds) if do_tune else {}
models = [cc.create_model(n, **model_params[n]) for n in chosen]
result = cc.Benchmark(data, models, tune=do_tune, **tune_kwargs).run() if chosen else None
scored_on = result.scored_on if result else ("test" if data.has_test_set else "train")
if scored_on == "train":
    st.warning(
        "No test split: every score below is measured on the rows the models were trained on. "
        "That shows how well each model fits this data, not how well it predicts new data."
    )
# One fixed colour per registered model, so a model keeps its colour whichever others are selected.
MODEL_COLORS = {registry[n].label: C["series"][i % len(C["series"])] for i, n in enumerate(registry)}

tab_data, tab_bench, tab_model = st.tabs(["Data", "Benchmark", "Model detail"])

# ---------------------------------------------------------------- data tab

with tab_data:
    summary = data.summary()
    cols = st.columns(4)
    cols[0].metric("Rows", f"{summary['shape']['rows']:,}")
    cols[1].metric("Features", len(summary["feature_types"]))
    cols[2].metric("Task", summary["task"].title())
    cols[3].metric("Missing cells", f"{sum(summary['missing'].values()):,}")

    left, right = st.columns([2, 3])
    with left:
        st.subheader("Columns")
        columns = pd.DataFrame({
            "column": list(data.frame.columns),
            "role": ["target" if c == target else "feature" for c in data.frame.columns],
            "type": [summary["feature_types"].get(c, "target") for c in data.frame.columns],
            "missing": [summary["missing"][c] for c in data.frame.columns],
        })
        st.dataframe(columns, hide_index=True, width="stretch")

    with right:
        corr = summary["correlations"]
        if corr:
            st.subheader("Correlations")
            long = pd.DataFrame(corr).rename_axis("a").reset_index().melt("a", var_name="b", value_name="r")
            base = alt.Chart(long).encode(
                x=alt.X("a:N", title=None, sort=None, axis=alt.Axis(labelAngle=0, orient="top")), y=alt.Y("b:N", title=None, sort=None),
            )
            cells = base.mark_rect(stroke=C["grid"], strokeWidth=2).encode(
                color=alt.Color("r:Q", scale=alt.Scale(domain=[-1, 0, 1], range=[C["neg"], C["mid"], C["pos"]], interpolate="lab"), legend=None),
                tooltip=[alt.Tooltip("a:N", title="Column"), alt.Tooltip("b:N", title="vs"), alt.Tooltip("r:Q", format=".3f")],
            )
            labels = base.mark_text(fontSize=11, color=C["ink"]).encode(text=alt.Text("r:Q", format=".2f"))
            show(cells + labels, height=40 + 56 * long["a"].nunique())
            st.caption("Pearson r between numeric columns: red is positive, blue is negative, grey is none.")

    st.subheader("Preview")
    st.dataframe(data.frame.head(50), width="stretch")

# ---------------------------------------------------------------- benchmark tab

with tab_bench:
    if result is None:
        st.info("Pick at least one model in the sidebar.")
    else:
        primary = result.primary_metric
        table = result.table
        st.subheader(f"Results — ranked by {'test' if scored_on == 'test' else 'training'} {primary}")
        failed = table[table["error"].notna()]
        for _, row in failed.iterrows():
            st.error(f"{row['model']}: {row['error']}")
        ok = table[table["error"].isna()].drop(columns="error")
        st.dataframe(ok.style.format(precision=4), hide_index=True, width="stretch")

        if not ok.empty:
            better = "higher" if HIGHER_IS_BETTER[primary] else "lower"
            if scored_on == "test":
                st.caption(f"Train vs test {primary} per model ({better} is better). A large gap suggests overfitting.")
                splits = {f"train_{primary}": "Train", primary: "Test"}
            else:
                st.caption(f"Training {primary} per model ({better} is better).")
                splits = {primary: "Train"}
            long = ok.melt("model", list(splits), var_name="split", value_name="score")
            long["split"] = long["split"].map(splits)
            dots = alt.Chart(long).mark_circle(size=120, opacity=1, stroke="white", strokeWidth=2).encode(
                x=alt.X("score:Q", title=primary, scale=alt.Scale(zero=False)),
                y=alt.Y("model:N", title=None, sort=list(ok["model"])),
                color=_split_color(list(splits.values())),
                tooltip=["model", "split", alt.Tooltip("score:Q", format=".4f")],
            )
            show(dots, height=100 + 44 * len(ok))

        numeric_raw = [c for c, k in data.feature_types.items() if k == "numeric"]
        fitted = [e for e in result.entries if e.error is None]
        if data.task == "regression" and len(data.features) == 1 and numeric_raw and fitted:
            feature = numeric_raw[0]
            st.subheader(f"Fitted models: {target} vs {feature}")
            rows = pd.concat([
                data.X_train_raw.assign(**{target: data.y_train, "split": "Train"}),
                data.X_test_raw.assign(**{target: data.y_test, "split": "Test"}),
            ]) if data.has_test_set else data.X_train_raw.assign(**{target: data.y_train, "split": "Train"})
            grid = pd.DataFrame({feature: np.linspace(rows[feature].min(), rows[feature].max(), CURVE_POINTS)})
            X_grid = data.transform(grid)
            curves = pd.concat([
                grid.assign(fit=e.model.predict(X_grid), model=e.model.label or e.model.name) for e in fitted
            ])
            labels = list(curves["model"].unique())
            lines = alt.Chart(curves).mark_line(strokeWidth=2).encode(
                x=alt.X(f"{feature}:Q", title=feature, scale=alt.Scale(zero=False)),
                y=alt.Y("fit:Q", title=target),
                color=alt.Color("model:N", scale=alt.Scale(domain=labels, range=[MODEL_COLORS.get(m, C["ink2"]) for m in labels])),
                tooltip=["model", alt.Tooltip(f"{feature}:Q", format=",.2f"), alt.Tooltip("fit:Q", title="fitted", format=",.2f")],
            )
            points = alt.Chart(rows).mark_circle(size=70, color=C["ink2"], opacity=0.8).encode(
                x=f"{feature}:Q", y=f"{target}:Q",
                tooltip=["split", alt.Tooltip(f"{feature}:Q", format=",.2f"), alt.Tooltip(f"{target}:Q", format=",.2f")],
            )
            show(points + lines, height=380)
            st.caption("Grey dots: the data. Each line: one model's predictions across the feature's range.")

# ---------------------------------------------------------------- model detail tab


def _scatter(df: pd.DataFrame, x: str, y: str, x_title: str, y_title: str) -> alt.Chart:
    return alt.Chart(df).mark_circle(size=80, opacity=0.9).encode(
        x=alt.X(f"{x}:Q", title=x_title, scale=alt.Scale(zero=False)),
        y=alt.Y(f"{y}:Q", title=y_title, scale=alt.Scale(zero=False)),
        color=_split_color([s for s in ("Train", "Test") if s in set(df["split"])]),
        tooltip=[alt.Tooltip(c, format=",.2f") if c != "split" else c for c in df.columns],
    )


with tab_model:
    entries = [e for e in (result.entries if result else []) if e.error is None]
    if not entries:
        st.info("No fitted models to inspect.")
    else:
        labels = {e.model.label or e.model.name: e for e in entries}
        entry = labels[st.selectbox("Model", list(labels))]
        model = entry.model
        st.markdown(f"**Fitted as:** {model.variant}")

        prefix = "Test" if scored_on == "test" else "Train"
        cols = st.columns(len(entry.metrics))
        for col, (k, v) in zip(cols, entry.metrics.items()):
            train_v = entry.train_metrics.get(k)
            hint = f"Train {k}: {train_v:,.4f}" if scored_on == "test" and train_v is not None else None
            col.metric(f"{prefix} {k}", f"{v:,.4f}", help=hint)

        if hasattr(model, "equation"):
            st.code(model.equation(target), language=None, wrap_lines=True)

        with st.expander("Predict a new value", expanded=True):
            inputs = {}
            cols = st.columns(min(4, len(data.features)))
            for i, f in enumerate(data.features):
                column = data.frame[f]
                with cols[i % len(cols)]:
                    if data.feature_types[f] == "numeric":
                        inputs[f] = st.number_input(f, value=float(column.median()), key=f"predict_{name}_{f}", format="%g")
                    else:
                        inputs[f] = st.selectbox(f, sorted(column.dropna().unique()), key=f"predict_{name}_{f}")
            prediction = model.predict(data.transform(pd.DataFrame([inputs])))[0]
            if data.classes is not None:
                prediction = data.classes[int(prediction)]
                st.metric(f"Predicted {target}", str(prediction))
            else:
                st.metric(f"Predicted {target}", f"{prediction:,.2f}")

        if data.task == "regression":
            train_df = data.X_train_raw.assign(**{target: data.y_train, "predicted": model.predict(data.X_train), "split": "Train"})
            if data.has_test_set:
                test_df = data.X_test_raw.assign(**{target: data.y_test, "predicted": entry.y_pred, "split": "Test"})
                both = pd.concat([train_df, test_df])
            else:
                test_df, both = None, train_df
            numeric_raw = [c for c, k in data.feature_types.items() if k == "numeric"]

            if len(data.features) == 1 and numeric_raw:
                left, right = st.columns(2)
                feature = numeric_raw[0]
                with left:
                    st.subheader(f"{target} vs {feature}")
                    grid = pd.DataFrame({feature: np.linspace(both[feature].min(), both[feature].max(), CURVE_POINTS)})
                    grid["fit"] = model.predict(data.transform(grid))
                    line = alt.Chart(grid).mark_line(color=C["ink2"], strokeWidth=2).encode(
                        x=f"{feature}:Q", y="fit:Q",
                        tooltip=[alt.Tooltip(f"{feature}:Q", format=",.2f"), alt.Tooltip("fit:Q", title="fitted", format=",.2f")],
                    )
                    points = _scatter(both[[feature, target, "split"]], feature, target, feature, target)
                    show(points + line, height=340)
                    st.caption("Grey line: the fitted model. Blue: training rows." + (" Orange: held-out test rows." if data.has_test_set else ""))
                target_panel = right
            else:
                target_panel = st.container()

            with target_panel:
                st.subheader("Predicted vs actual")
                lo, hi = both[[target, "predicted"]].min().min(), both[[target, "predicted"]].max().max()
                ideal = alt.Chart(pd.DataFrame({"v": [lo, hi]})).mark_line(color=C["muted"], strokeDash=[4, 4], strokeWidth=1.5).encode(x="v:Q", y="v:Q")
                points = _scatter(both[[target, "predicted", "split"]], target, "predicted", f"Actual {target}", f"Predicted {target}")
                show(ideal + points, height=340)
                st.caption("Dashed line: perfect prediction.")

            with st.expander("Test-set predictions" if data.has_test_set else "Predictions (training rows)"):
                table = (test_df if data.has_test_set else train_df)[[target, "predicted"]].rename(columns={target: "actual"})
                table["error"] = table["actual"] - table["predicted"]
                st.dataframe(table.style.format("{:,.2f}"), width="stretch")

            st.subheader("Residuals")
            both["residual"] = both[target] - both["predicted"]
            zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=C["axis"], strokeWidth=1.5).encode(y="y:Q")
            show((zero + _scatter(both[["predicted", "residual", "split"]], "predicted", "residual", f"Predicted {target}", "Actual − predicted")), height=260)
        else:
            st.subheader(f"Confusion matrix ({'test' if data.has_test_set else 'training'} rows)")
            y_eval = data.y_test if data.has_test_set else data.y_train
            names = data.classes or sorted(np.unique(y_eval))
            cm = confusion_matrix(y_eval, entry.y_pred, labels=range(len(names)))
            long = pd.DataFrame(cm, index=names, columns=names).rename_axis("actual").reset_index().melt("actual", var_name="predicted", value_name="count")
            base = alt.Chart(long).encode(x=alt.X("predicted:N", sort=None), y=alt.Y("actual:N", sort=None))
            cells = base.mark_rect(stroke=C["grid"], strokeWidth=2).encode(
                color=alt.Color("count:Q", scale=alt.Scale(range=C["seq"]), legend=None), tooltip=["actual", "predicted", "count"],
            )
            show(cells + base.mark_text(color=C["ink"]).encode(text="count:Q"), height=80 + 60 * len(names))

        importances = model.feature_importances
        if importances and len(importances) > 1:
            st.subheader("Feature importance")
            imp = pd.DataFrame({"column": list(importances), "importance": list(importances.values())})
            order = list(imp.sort_values("importance", ascending=False)["column"])
            bars = alt.Chart(imp).mark_bar(color=C["series"][0], cornerRadiusEnd=4, height={"band": 0.7}).encode(
                x=alt.X("importance:Q", title="Share of the model's splitting power", axis=alt.Axis(format="%")),
                y=alt.Y("column:N", title=None, sort=order),
                tooltip=["column", alt.Tooltip("importance:Q", format=".1%")],
            )
            values = bars.mark_text(align="left", dx=4, color=C["ink2"]).encode(text=alt.Text("importance:Q", format=".1%"))
            show(bars + values, height=40 + 36 * len(imp))
            st.caption("How much each column contributed to the model's decisions. The bars sum to 100%.")

        if model.tune_result_ is not None:
            tr = model.tune_result_
            st.subheader("Tuning")
            st.caption(f"{tr.method} search, {tr.n_trials} trials, best CV {tr.scoring} {tr.best_score:.4f} in {tr.seconds:.2f}s")
            rows_per_fold = len(data.y_train) // max(2, min(folds, len(data.y_train)))
            if rows_per_fold < 5:
                st.warning(
                    f"Each cross-validation fold scores only about {rows_per_fold} rows, so these scores are very noisy "
                    "and the chosen parameters may not be the best. Fewer folds or more data will help."
                )
            # Mixed-type parameters (e.g. max_features 1.0 or "sqrt") show as text, labelled like the sidebar.
            trials = tr.trials.copy()
            for col in trials.columns[trials.dtypes == object]:
                trials[col] = trials[col].map(lambda v: "No limit" if v is None else str(v))
            st.dataframe(trials.sort_values("score", ascending=False), hide_index=True, width="stretch")
