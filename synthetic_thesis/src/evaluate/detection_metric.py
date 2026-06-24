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
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
from autogluon.tabular import TabularPredictor

import warnings
warnings.filterwarnings("ignore")

from config import DATASETS, MODELS, DM_DIR, RANDOM_STATE, PLOTS_DIR


def get_column_types(df: pd.DataFrame):
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()
    return categorical_cols, numerical_cols


def build_detection_dataset(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> pd.DataFrame:
    common_columns = [col for col in real_df.columns if col in synth_df.columns]

    real_part = real_df[common_columns].copy()
    synth_part = synth_df[common_columns].copy()

    real_part["dm_target"] = 0
    synth_part["dm_target"] = 1

    combined_df = pd.concat([real_part, synth_part], axis=0, ignore_index=True)
    combined_df = combined_df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    return combined_df


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


def evaluate_classifier(model_name: str, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "model": model_name,
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_test, y_pred),
    }

    return metrics

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

    leaderboard = predictor.leaderboard(test_df, silent=True)
    print(
        leaderboard[["model", "score_test"]]
        .head(1)
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

def save_metric_bar_plot(df: pd.DataFrame, metric: str, title: str, save_path: Path):
    if df is None or df.empty or metric not in df.columns:
        return

    plot_df = df.sort_values(metric, ascending=False)

    plt.figure(figsize=(8, 5))
    plt.ylim(0, 1)
    plt.bar(plot_df["model"].astype(str), plot_df[metric])
    plt.title(title)
    plt.xlabel("Classifier")
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
    plt.xlabel("Classifier")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=20)
    plt.legend(title="Metric")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def evaluate_model_on_dataset(dataset_name: str, gan_model_name: str, dataset_cfg: dict):
    print(f"\n--- {dataset_name.upper()} | {gan_model_name.upper()} ---")

    real_path = dataset_cfg["test_path"]
    synth_path = dataset_cfg["synthetic_dir"] / f"{gan_model_name}.csv"

    if not real_path.exists():
        raise FileNotFoundError(f"Gerçek veri bulunamadı: {real_path}")

    if not synth_path.exists():
        raise FileNotFoundError(f"Sentetik veri bulunamadı: {synth_path}")

    real_df = pd.read_csv(real_path)
    synth_df = pd.read_csv(synth_path)

    target_col = dataset_cfg["target_column"]

    if target_col in real_df.columns:
        real_df[target_col] = real_df[target_col].astype(str)

    if target_col in synth_df.columns:
        synth_df[target_col] = synth_df[target_col].astype(str)

    dm_df = build_detection_dataset(real_df, synth_df)

    X = dm_df.drop(columns=["dm_target"])
    y = dm_df["dm_target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = build_preprocessor(X)

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
            eval_metric="logloss",
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

    save_dir = DM_DIR / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = PLOTS_DIR / "dm" / dataset_name
    plot_dir.mkdir(parents=True, exist_ok=True)
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
            f"{clf_name} | "
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
    results_df = pd.DataFrame(results)

    print(
        f"[CLS] autogluon | "
        f"precision={ag_metrics['precision']:.4f} | "
        f"recall={ag_metrics['recall']:.4f} | "
        f"f1={ag_metrics['f1']:.4f} | "
        f"accuracy={ag_metrics['accuracy']:.4f}"
    )

    results_df.round(3).to_csv(save_dir / f"{gan_model_name}_dm_results.csv", index=False)

    for metric in ["precision", "recall", "f1", "accuracy"]:
        save_metric_bar_plot(
            results_df,
            metric=metric,
            title=f"{dataset_name} - {gan_model_name} DM {metric.upper()}",
            save_path=plot_dir / f"{gan_model_name}_dm_{metric}.png"
        )

    save_combined_metrics_plot(
        results_df,
        title=f"{dataset_name} - {gan_model_name} DM All Metrics",
        save_path=plot_dir / f"{gan_model_name}_dm_all_metrics.png"
    )

    best_ag = next(
        (
            r["autogluon_best_model"]
            for r in results
            if r["model"] == "autogluon"
        ),
        None
    )

    summary = {
        "dataset": dataset_name,
        "synthetic_model": gan_model_name,
        "rows_real": int(len(real_df)),
        "rows_synthetic": int(len(synth_df)),
        "test_size": 0.20,
        "random_state": RANDOM_STATE,
        "classifiers": list(classifiers.keys()),
        "best_f1_model": results_df.sort_values("f1", ascending=False).iloc[0]["model"],
        "best_f1_score": float(results_df["f1"].max()),
        "autogluon_best_model": best_ag,
    }

    with open(save_dir / f"{gan_model_name}_dm_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main():
    for dataset_name, dataset_cfg in DATASETS.items():
        print(f"\n==================== {dataset_name.upper()} ====================")

        for gan_model_name in MODELS:
            try:
                evaluate_model_on_dataset(dataset_name, gan_model_name, dataset_cfg)
            except Exception as e:
                print(f"HATA | {dataset_name} | {gan_model_name}: {e}")


if __name__ == "__main__":
    main()