import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    r2_score,
    mean_squared_error,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    RandomForestRegressor,
    AdaBoostRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from xgboost import (
    XGBClassifier,
    XGBRegressor,
)
from catboost import (
    CatBoostClassifier,
    CatBoostRegressor,
)
from autogluon.tabular import TabularPredictor

import warnings
warnings.filterwarnings("ignore")

from config import DATASETS, MODELS, EM_DIR, RANDOM_STATE, PLOTS_DIR


def get_column_types(df: pd.DataFrame):
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()
    return categorical_cols, numerical_cols


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols, numerical_cols = get_column_types(X)

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor


def align_real_and_synth(real_df: pd.DataFrame, synth_df: pd.DataFrame, target_col: str):
    common_columns = [col for col in real_df.columns if col in synth_df.columns]

    if target_col not in common_columns:
        raise ValueError(f"Hedef sütun ortak kolonlarda bulunamadı: {target_col}")

    real_df = real_df[common_columns].copy()
    synth_df = synth_df[common_columns].copy()

    return real_df, synth_df


def evaluate_classifier(model_name: str, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    result = {
        "model": model_name,
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "accuracy": accuracy_score(y_test, y_pred),
    }

    labels = sorted(y_test.unique())
    per_class_p = precision_score(y_test, y_pred, average=None, labels=labels, zero_division=0)
    per_class_r = recall_score(y_test, y_pred, average=None, labels=labels, zero_division=0)
    per_class_f1 = f1_score(y_test, y_pred, average=None, labels=labels, zero_division=0)

    for i, label in enumerate(labels):
        safe_label = str(label).replace(">", "gt_").replace("<", "lt_").replace("=", "eq_")
        result[f"precision_{safe_label}"] = per_class_p[i]
        result[f"recall_{safe_label}"] = per_class_r[i]
        result[f"f1_{safe_label}"] = per_class_f1[i]

    return result

def evaluate_autogluon(
    X_train,
    X_test,
    y_train,
    y_test,
    save_dir,
    gan_model_name,
):
    train_df = X_train.copy()
    train_df["target"] = y_train.values

    test_df = X_test.copy()
    test_df["target"] = y_test.values

    predictor = TabularPredictor(
        label="target",
        eval_metric="f1"
    ).fit(
        train_data=train_df,
        presets="medium_quality",
        verbosity=0
    )

    y_pred = predictor.predict(test_df.drop(columns=["target"]))

    leaderboard = predictor.leaderboard(
        test_df,
        silent=True
    )

    leaderboard.to_csv(
        save_dir / f"{gan_model_name}_autogluon_leaderboard.csv",
        index=False
    )

    best_model = leaderboard.iloc[0]["model"]

    metrics = {
        "model": "autogluon",
        "precision": precision_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        ),
        "recall": recall_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        ),
        "f1": f1_score(
            y_test,
            y_pred,
            average="weighted",
            zero_division=0
        ),
        "accuracy": accuracy_score(
            y_test,
            y_pred
        ),
        "autogluon_best_model": best_model,
    }

    return metrics

def evaluate_autogluon_regression(
    X_train,
    X_test,
    y_train,
    y_test,
    save_dir,
    gan_model_name,
):
    train_df = X_train.copy()
    train_df["target"] = y_train.values

    test_df = X_test.copy()
    test_df["target"] = y_test.values

    predictor = TabularPredictor(
        label="target",
        problem_type="regression",
        eval_metric="root_mean_squared_error"
    ).fit(
        train_data=train_df,
        presets="medium_quality",
        verbosity=0
    )

    y_pred = predictor.predict(
        test_df.drop(columns=["target"])
    )

    leaderboard = predictor.leaderboard(
        test_df,
        silent=True
    )

    leaderboard.to_csv(
        save_dir / f"{gan_model_name}_autogluon_regression_leaderboard.csv",
        index=False
    )

    best_model = leaderboard.iloc[0]["model"]

    return {
        "model": "autogluon",
        "rmse": mean_squared_error(y_test, y_pred) ** 0.5,
        "r2": r2_score(y_test, y_pred),
        "autogluon_best_model": best_model,
    }

def evaluate_regressor(model_name: str, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    return {
        "model": model_name,
        "rmse": rmse,
        "r2": r2,
    }

def save_metric_bar_plot(df: pd.DataFrame, metric: str, title: str, save_path: Path, ascending=False):
    if df is None or df.empty or metric not in df.columns:
        return

    plot_df = df.sort_values(metric, ascending=ascending)

    plt.figure(figsize=(8, 5))

    if metric in ["precision", "recall", "f1", "accuracy"]:
        plt.ylim(0, 1)

    plt.bar(plot_df["model"].astype(str), plot_df[metric])
    plt.title(title)
    plt.xlabel("Model")
    plt.ylabel(metric)

    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_combined_metrics_plot(df: pd.DataFrame, title: str, save_path: Path):
    if df is None or df.empty:
        return

    metrics = ["precision", "recall", "f1", "accuracy"]
    plot_df = df.set_index("model")[metrics]

    plot_df.plot(kind="bar", figsize=(10, 6))

    plt.title(title)
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=20)
    plt.legend(title="Metric")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def run_classification_em(dataset_name: str, gan_model_name: str, dataset_cfg: dict):
    target_col = dataset_cfg["target_column"]

    real_path = dataset_cfg["test_path"]
    synth_path = dataset_cfg["synthetic_dir"] / f"{gan_model_name}.csv"

    real_df = pd.read_csv(real_path)
    synth_df = pd.read_csv(synth_path)

    real_df, synth_df = align_real_and_synth(real_df, synth_df, target_col)

    real_df[target_col] = real_df[target_col].astype(str)
    synth_df[target_col] = synth_df[target_col].astype(str)

    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()

    all_labels = pd.concat(
        [
            real_df[target_col],
            synth_df[target_col]
        ]
    )

    le.fit(all_labels)

    real_df[target_col] = le.transform(
        real_df[target_col]
    )

    synth_df[target_col] = le.transform(
        synth_df[target_col]
    )

    X_train = synth_df.drop(columns=[target_col])
    y_train = synth_df[target_col]

    X_test = real_df.drop(columns=[target_col])
    y_test = real_df[target_col]

    preprocessor = build_preprocessor(X_train)

    classifiers = {

        "logistic_regression": LogisticRegression(
            max_iter=5000,
            random_state=RANDOM_STATE,
            solver="lbfgs",
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            eval_metric="mlogloss",
            n_jobs=-1,
        ),

        "catboost": CatBoostClassifier(
            iterations=200,
            learning_rate=0.05,
            depth=6,
            verbose=0,
            random_state=RANDOM_STATE,
        ),

        "adaboost": AdaBoostClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
        ),
    }

    save_dir = EM_DIR / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for clf_name, clf in classifiers.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", clf),
            ]
        )

        metrics = evaluate_classifier(
            model_name=clf_name,
            model=pipeline,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )

        results.append(metrics)

        print(
            f"[CLS] {clf_name} | "
            f"precision={metrics['precision']:.4f} | "
            f"recall={metrics['recall']:.4f} | "
            f"f1={metrics['f1']:.4f} | "
            f"accuracy={metrics['accuracy']:.4f}"
        )

    ag_metrics = evaluate_autogluon(
        X_train,
        X_test,
        y_train,
        y_test,
        save_dir,
        gan_model_name
    )
    results.append(ag_metrics)

    print(
        f"[CLS] autogluon | "
        f"precision={ag_metrics['precision']:.4f} | "
        f"recall={ag_metrics['recall']:.4f} | "
        f"f1={ag_metrics['f1']:.4f} | "
        f"accuracy={ag_metrics['accuracy']:.4f}"
    )

    return pd.DataFrame(results)

def run_regression_em(dataset_name: str, gan_model_name: str, dataset_cfg: dict):
    target_col = dataset_cfg["regression_target"]

    real_path = dataset_cfg["test_path"]
    synth_path = dataset_cfg["synthetic_dir"] / f"{gan_model_name}.csv"

    real_df = pd.read_csv(real_path)
    synth_df = pd.read_csv(synth_path)

    real_df, synth_df = align_real_and_synth(real_df, synth_df, target_col)

    # regresyon hedefi sayısal olmalı
    real_df[target_col] = pd.to_numeric(real_df[target_col], errors="coerce")
    synth_df[target_col] = pd.to_numeric(synth_df[target_col], errors="coerce")

    real_df = real_df.dropna(subset=[target_col]).reset_index(drop=True)
    synth_df = synth_df.dropna(subset=[target_col]).reset_index(drop=True)

    X_train = synth_df.drop(columns=[target_col])
    y_train = synth_df[target_col]

    X_test = real_df.drop(columns=[target_col])
    y_test = real_df[target_col]

    preprocessor = build_preprocessor(X_train)

    regressors = {

        "linear_regression": LinearRegression(),

        "random_forest": RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "xgboost": XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            objective="reg:squarederror",
            n_jobs=-1,
        ),

        "catboost": CatBoostRegressor(
            iterations=200,
            learning_rate=0.05,
            depth=6,
            verbose=0,
            random_state=RANDOM_STATE,
        ),

        "adaboost": AdaBoostRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE,
        ),
    }

    results = []

    for reg_name, reg in regressors.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", reg),
            ]
        )

        metrics = evaluate_regressor(
            model_name=reg_name,
            model=pipeline,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )

        results.append(metrics)

        print(
            f"[REG] {reg_name} | "
            f"rmse={metrics['rmse']:.4f} | "
            f"r2={metrics['r2']:.4f}"
        )

    ag_metrics = evaluate_autogluon_regression(
        X_train,
        X_test,
        y_train,
        y_test,
        EM_DIR / dataset_name,
        gan_model_name
    )

    results.append(ag_metrics)

    print(
        f"[REG] autogluon | "
        f"rmse={ag_metrics['rmse']:.4f} | "
        f"r2={ag_metrics['r2']:.4f}"
    )

    return pd.DataFrame(results)


def evaluate_model_on_dataset(dataset_name: str, gan_model_name: str, dataset_cfg: dict):
    print(f"\n--- {dataset_name.upper()} | {gan_model_name.upper()} ---")

    save_dir = EM_DIR / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)

    plot_dir = PLOTS_DIR/ "em" / dataset_name
    plot_dir.mkdir(parents=True, exist_ok=True)

    cls_df = run_classification_em(dataset_name, gan_model_name, dataset_cfg)
    cls_df.round(3).to_csv(save_dir / f"{gan_model_name}_em_classification.csv", index=False)

    # ayrı grafikler
    for metric in ["precision", "recall", "f1", "accuracy"]:
        save_metric_bar_plot(
            cls_df,
            metric=metric,
            title=f"{dataset_name} - {gan_model_name} EM Classification {metric.upper()}",
            save_path=plot_dir / f"{gan_model_name}_em_cls_{metric}.png"
        )

    # birleşik grafik
    save_combined_metrics_plot(
        cls_df,
        title=f"{dataset_name} - {gan_model_name} EM Classification All Metrics",
        save_path=plot_dir / f"{gan_model_name}_em_cls_all_metrics.png"
    )

    reg_df = run_regression_em(dataset_name, gan_model_name, dataset_cfg)
    reg_df.round(3).to_csv(save_dir / f"{gan_model_name}_em_regression.csv", index=False)

    # RMSE (küçük iyi)
    save_metric_bar_plot(
        reg_df,
        metric="rmse",
        title=f"{dataset_name} - {gan_model_name} EM Regression RMSE",
        save_path=plot_dir / f"{gan_model_name}_em_reg_rmse.png",
        ascending=True
    )

    # R2 (büyük iyi)
    save_metric_bar_plot(
        reg_df,
        metric="r2",
        title=f"{dataset_name} - {gan_model_name} EM Regression R2",
        save_path=plot_dir / f"{gan_model_name}_em_reg_r2.png",
        ascending=False
    )

    best_ag_cls = next(
        (
            r["autogluon_best_model"]
            for _, r in cls_df.iterrows()
            if r["model"] == "autogluon"
        ),
        None
    )

    best_ag_reg = next(
        (
            r["autogluon_best_model"]
            for _, r in reg_df.iterrows()
            if r["model"] == "autogluon"
        ),
        None
    )

    summary = {
        "dataset": dataset_name,
        "synthetic_model": gan_model_name,
        "classification_target": dataset_cfg["target_column"],
        "regression_target": dataset_cfg["regression_target"],
        "train_source": "synthetic",
        "test_source": "real",
        "best_classification_f1_model": cls_df.sort_values("f1", ascending=False).iloc[0]["model"],
        "best_classification_f1_score": float(cls_df["f1"].max()),
        "autogluon_classification_best_model": best_ag_cls,
        "autogluon_regression_best_model": best_ag_reg,
        "best_regression_r2_model": reg_df.sort_values("r2", ascending=False).iloc[0]["model"],
        "best_regression_r2_score": float(reg_df["r2"].max()),
        "best_regression_rmse_model": reg_df.sort_values("rmse", ascending=True).iloc[0]["model"],
        "best_regression_rmse_score": float(reg_df["rmse"].min()),
    }

    with open(save_dir / f"{gan_model_name}_em_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def run_baseline(dataset_name: str, dataset_cfg: dict):
    print(f"\n--- {dataset_name.upper()} | BASELINE (real -> real) ---")

    save_dir = EM_DIR / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)

    plot_dir = PLOTS_DIR / "em" / dataset_name
    plot_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(dataset_cfg["train_path"])
    test_df = pd.read_csv(dataset_cfg["test_path"])

    # --- Classification ---
    cls_target = dataset_cfg["target_column"]
    common_cols = [c for c in train_df.columns if c in test_df.columns]
    train_cls = train_df[common_cols].copy()
    test_cls = test_df[common_cols].copy()
    train_cls[cls_target] = train_cls[cls_target].astype(str)
    test_cls[cls_target] = test_cls[cls_target].astype(str)

    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()

    train_cls[cls_target] = le.fit_transform(
        train_cls[cls_target]
    )

    test_cls[cls_target] = le.transform(
        test_cls[cls_target]
    )

    X_train = train_cls.drop(columns=[cls_target])
    y_train = train_cls[cls_target]
    X_test = test_cls.drop(columns=[cls_target])
    y_test = test_cls[cls_target]

    preprocessor = build_preprocessor(X_train)

    classifiers = {

        "logistic_regression": LogisticRegression(
            max_iter=5000,
            random_state=RANDOM_STATE,
            solver="lbfgs",
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            eval_metric="mlogloss",
            n_jobs=-1,
        ),

        "catboost": CatBoostClassifier(
            iterations=200,
            learning_rate=0.05,
            depth=6,
            verbose=0,
            random_state=RANDOM_STATE,
        ),

        "adaboost": AdaBoostClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
        ),
    }

    cls_results = []
    for clf_name, clf in classifiers.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
        metrics = evaluate_classifier(clf_name, pipeline, X_train, X_test, y_train, y_test)
        cls_results.append(metrics)
        print(f"[CLS] {clf_name} | precision={metrics['precision']:.4f} | recall={metrics['recall']:.4f} | f1={metrics['f1']:.4f} | accuracy={metrics['accuracy']:.4f}")

    ag_metrics = evaluate_autogluon(
        X_train,
        X_test,
        y_train,
        y_test,
        save_dir,
        "baseline"
    )

    cls_results.append(ag_metrics)

    print(
        f"[CLS] autogluon | "
        f"precision={ag_metrics['precision']:.4f} | "
        f"recall={ag_metrics['recall']:.4f} | "
        f"f1={ag_metrics['f1']:.4f} | "
        f"accuracy={ag_metrics['accuracy']:.4f}"
    )

    cls_df = pd.DataFrame(cls_results)
    cls_df.round(3).to_csv(save_dir / "baseline_em_classification.csv", index=False)

    for metric in ["precision", "recall", "f1", "accuracy"]:
        save_metric_bar_plot(cls_df, metric=metric, title=f"{dataset_name} - BASELINE EM Classification {metric.upper()}", save_path=plot_dir / f"baseline_em_cls_{metric}.png")
    save_combined_metrics_plot(cls_df, title=f"{dataset_name} - BASELINE EM Classification All Metrics", save_path=plot_dir / f"baseline_em_cls_all_metrics.png")

    # --- Regression ---
    reg_target = dataset_cfg["regression_target"]
    train_reg = train_df[common_cols].copy()
    test_reg = test_df[common_cols].copy()
    train_reg[reg_target] = pd.to_numeric(train_reg[reg_target], errors="coerce")
    test_reg[reg_target] = pd.to_numeric(test_reg[reg_target], errors="coerce")
    train_reg = train_reg.dropna(subset=[reg_target]).reset_index(drop=True)
    test_reg = test_reg.dropna(subset=[reg_target]).reset_index(drop=True)

    X_train_r = train_reg.drop(columns=[reg_target])
    y_train_r = train_reg[reg_target]
    X_test_r = test_reg.drop(columns=[reg_target])
    y_test_r = test_reg[reg_target]

    preprocessor_r = build_preprocessor(X_train_r)

    regressors = {

        "linear_regression": LinearRegression(),

        "random_forest": RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "xgboost": XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            objective="reg:squarederror",
            n_jobs=-1,
        ),

        "catboost": CatBoostRegressor(
            iterations=200,
            learning_rate=0.05,
            depth=6,
            verbose=0,
            random_state=RANDOM_STATE,
        ),

        "adaboost": AdaBoostRegressor(
            n_estimators=100,
            random_state=RANDOM_STATE,
        ),
    }

    reg_results = []
    for reg_name, reg in regressors.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor_r), ("regressor", reg)])
        metrics = evaluate_regressor(reg_name, pipeline, X_train_r, X_test_r, y_train_r, y_test_r)
        reg_results.append(metrics)
        print(f"[REG] {reg_name} | rmse={metrics['rmse']:.4f} | r2={metrics['r2']:.4f}")

    ag_metrics = evaluate_autogluon_regression(
        X_train_r,
        X_test_r,
        y_train_r,
        y_test_r,
        save_dir,
        "baseline"
    )

    reg_results.append(ag_metrics)

    print(
        f"[REG] autogluon | "
        f"rmse={ag_metrics['rmse']:.4f} | "
        f"r2={ag_metrics['r2']:.4f}"
    )

    reg_df = pd.DataFrame(reg_results)
    reg_df.round(3).to_csv(save_dir / "baseline_em_regression.csv", index=False)

    save_metric_bar_plot(reg_df, metric="rmse", title=f"{dataset_name} - BASELINE EM Regression RMSE", save_path=plot_dir / f"baseline_em_reg_rmse.png", ascending=True)
    save_metric_bar_plot(reg_df, metric="r2", title=f"{dataset_name} - BASELINE EM Regression R2", save_path=plot_dir / f"baseline_em_reg_r2.png", ascending=False)

    best_ag_cls = next(
        (
            r["autogluon_best_model"]
            for _, r in cls_df.iterrows()
            if r["model"] == "autogluon"
        ),
        None
    )

    best_ag_reg = next(
        (
            r["autogluon_best_model"]
            for _, r in reg_df.iterrows()
            if r["model"] == "autogluon"
        ),
        None
    )

    summary = {
        "dataset": dataset_name,
        "synthetic_model": "baseline",
        "classification_target": cls_target,
        "regression_target": reg_target,
        "train_source": "real",
        "test_source": "real",
        "best_classification_f1_model": cls_df.sort_values("f1", ascending=False).iloc[0]["model"],
        "best_classification_f1_score": float(cls_df["f1"].max()),
        "autogluon_classification_best_model": best_ag_cls,
        "autogluon_regression_best_model": best_ag_reg,
        "best_regression_r2_model": reg_df.sort_values("r2", ascending=False).iloc[0]["model"],
        "best_regression_r2_score": float(reg_df["r2"].max()),
        "best_regression_rmse_model": reg_df.sort_values("rmse", ascending=True).iloc[0]["model"],
        "best_regression_rmse_score": float(reg_df["rmse"].min()),
    }

    with open(save_dir / "baseline_em_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main():
    for dataset_name, dataset_cfg in DATASETS.items():
        print(f"\n==================== {dataset_name.upper()} ====================")

        run_baseline(dataset_name, dataset_cfg)

        for gan_model_name in MODELS:
            try:
                evaluate_model_on_dataset(dataset_name, gan_model_name, dataset_cfg)
            except Exception as e:
                print(f"HATA | {dataset_name} | {gan_model_name}: {e}")


if __name__ == "__main__":
    main()