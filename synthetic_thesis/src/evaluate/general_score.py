from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

STATS_DIR = ROOT / "outputs" / "stats"
DM_DIR = ROOT / "outputs" / "dm"
EM_DIR = ROOT / "outputs" / "em"

MODELS = [
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

DATASETS = ["adult", "credit"]


def normalize(series, higher_is_better=True):

    s = series.astype(float)

    if s.max() == s.min():
        return pd.Series([1] * len(s), index=s.index)

    if higher_is_better:
        return (s - s.min()) / (s.max() - s.min())

    return (s.max() - s) / (s.max() - s.min())


for dataset in DATASETS:

    rows = []

    for model in MODELS:

        # -------------------------
        # STATISTICS
        # -------------------------

        summary_path = (
            STATS_DIR /
            dataset /
            f"{model}_summary.txt"
        )

        metrics = {}

        with open(summary_path, "r") as f:
            for line in f:
                k, v = line.strip().split("=")
                metrics[k] = float(v)

        # -------------------------
        # DETECTION
        # -------------------------

        dm_path = (
            DM_DIR /
            dataset /
            f"{model}_dm_results.csv"
        )

        dm_df = pd.read_csv(dm_path)

        dm_f1 = dm_df["f1"].mean()
        dm_acc = dm_df["accuracy"].mean()

        # detection düşük iyidir
        detection_score = (
            dm_f1 + dm_acc
        ) / 2

        # -------------------------
        # EFFICACY CLS
        # -------------------------

        cls_path = (
            EM_DIR /
            dataset /
            f"{model}_em_classification.csv"
        )

        cls_df = pd.read_csv(cls_path)

        efficacy_cls = cls_df["f1"].mean()

        # -------------------------
        # EFFICACY REG
        # -------------------------

        reg_path = (
            EM_DIR /
            dataset /
            f"{model}_em_regression.csv"
        )

        reg_df = pd.read_csv(reg_path)

        efficacy_r2 = reg_df["r2"].mean()
        efficacy_rmse = reg_df["rmse"].mean()

        rows.append({

            "model": model,

            "ks": metrics["ks_avg"],
            "chi2": metrics["chi2_avg"],
            "corr_mae": metrics["corr_mae"],
            "wasserstein": metrics["wasserstein_avg"],
            "js": metrics["js_avg"],
            "cramer": metrics["cramer_mean_diff"],

            "dm": detection_score,

            "em_cls": efficacy_cls,
            "em_r2": efficacy_r2,
            "em_rmse": efficacy_rmse
        })

    df = pd.DataFrame(rows)

    # =====================
    # NORMALIZATION
    # =====================

    df["ks_n"] = normalize(df["ks"], False)
    df["chi2_n"] = normalize(df["chi2"], False)
    df["corr_n"] = normalize(df["corr_mae"], False)
    df["wass_n"] = normalize(df["wasserstein"], False)
    df["js_n"] = normalize(df["js"], False)
    df["cramer_n"] = normalize(df["cramer"], False)

    # detection düşük iyi
    df["dm_n"] = normalize(df["dm"], False)

    # efficacy yüksek iyi
    df["em_cls_n"] = normalize(df["em_cls"], True)
    df["em_r2_n"] = normalize(df["em_r2"], True)
    df["em_rmse_n"] = normalize(df["em_rmse"], False)

    # =====================
    # CATEGORY SCORES
    # =====================

    df["statistical_score"] = df[
        [
            "ks_n",
            "chi2_n",
            "corr_n",
            "wass_n",
            "js_n",
            "cramer_n"
        ]
    ].mean(axis=1)

    df["efficacy_score"] = df[
        [
            "em_cls_n",
            "em_r2_n",
            "em_rmse_n"
        ]
    ].mean(axis=1)

    # =====================
    # FINAL SCORE
    # =====================

    df["final_score"] = (
            0.4 * df["statistical_score"] +
            0.3 * df["dm_n"] +
            0.3 * df["efficacy_score"]
    )

    df = df.sort_values(
        "final_score",
        ascending=False
    )

    print("\n")
    print("=" * 50)
    print(dataset.upper())
    print("=" * 50)

    print(
        df[
            [
                "model",
                "statistical_score",
                "dm_n",
                "efficacy_score",
                "final_score"
            ]
        ].round(3)
    )

    df.to_csv(
        ROOT /
        "outputs" /
        f"{dataset}_general_ranking.csv",
        index=False
    )