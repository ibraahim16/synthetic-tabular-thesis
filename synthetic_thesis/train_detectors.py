import joblib
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parent

SAVE_DIR = ROOT / "models" / "detectors"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# MODELLER
# =====================================================

classifiers = {

    "lr": LogisticRegression(
        max_iter=2000,
        random_state=42
    ),

    "rf": RandomForestClassifier(
        n_estimators=300,
        random_state=42
    ),

    "adaboost": AdaBoostClassifier(
        random_state=42
    ),

    "xgb": XGBClassifier(
        random_state=42,
        eval_metric="logloss"
    ),

    "catboost": CatBoostClassifier(
        verbose=False,
        random_state=42
    )
}

# =====================================================
# SENTETİK MODELLER
# =====================================================

synthetic_models = [

    "smote",
    "borderline_smote",
    "smote_tomek",
    "cgan",
    "ctgan",
    "copulagan",
    "ctabgan_plus",
    "tabsyn",
    "forest_diffusion"
]


# =====================================================
# EĞİTİM
# =====================================================

def train_detectors(dataset_name):

    print(f"\n{'=' * 50}")
    print(dataset_name.upper())
    print(f"{'=' * 50}")

    real = pd.read_csv(
        ROOT / f"data/split/{dataset_name}_train.csv"
    )

    for synth_name in synthetic_models:

        print(f"\nSynthetic Model: {synth_name}")

        synth = pd.read_csv(
            ROOT /
            f"data/synthetic/{dataset_name}/{synth_name}.csv"
        )

        real_copy = real.copy()

        real_copy["target"] = 0
        synth["target"] = 1

        df = pd.concat(
            [real_copy, synth],
            ignore_index=True
        )

        drop_cols = ["target"]

        if dataset_name == "credit":
            drop_cols.append("Class")

        if dataset_name == "adult":
            drop_cols.append("income")

        X = df.drop(
            columns=drop_cols,
            errors="ignore"
        )

        X = pd.get_dummies(X)

        joblib.dump(
            X.columns.tolist(),
            SAVE_DIR / f"{dataset_name}_feature_names.pkl"
        )

        y = df["target"]

        X_train, X_test, y_train, y_test = (
            train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y
            )
        )

        for clf_name, clf in classifiers.items():

            print(
                f"Training: {synth_name} + {clf_name}"
            )

            clf.fit(X_train, y_train)

            filename = (
                f"{dataset_name}_"
                f"{synth_name}_"
                f"{clf_name}.pkl"
            )

            joblib.dump(
                clf,
                SAVE_DIR / filename
            )

            print(f"Saved -> {filename}")


# =====================================================
# ÇALIŞTIR
# =====================================================

train_detectors("adult")
train_detectors("credit")