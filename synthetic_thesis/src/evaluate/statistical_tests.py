import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import (
    ks_2samp,
    chi2_contingency,
    wasserstein_distance
)

from scipy.spatial.distance import jensenshannon

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from config import DATASETS, MODELS, STATS_DIR, PLOTS_DIR


def get_column_types(df: pd.DataFrame):

    categorical_cols = df.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()

    numerical_cols = df.select_dtypes(
        include=["number"]
    ).columns.tolist()

    return categorical_cols, numerical_cols


# --------------------------------------------------
# 1) KS TEST - Numerical Columns
# --------------------------------------------------

def ks_test(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    numerical_cols
):

    results = []

    for col in numerical_cols:

        real_series = real_df[col].dropna()
        synth_series = synth_df[col].dropna()

        if len(real_series) == 0 or len(synth_series) == 0:

            results.append({
                "column": col,
                "ks_stat": np.nan,
                "p_value": np.nan
            })

            continue

        stat, p_value = ks_2samp(
            real_series,
            synth_series
        )

        results.append({
            "column": col,
            "ks_stat": stat,
            "p_value": p_value
        })

    return pd.DataFrame(results)


# --------------------------------------------------
# 2) CHI-SQUARE TEST - Categorical Columns
# --------------------------------------------------

def chi_square_test(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    categorical_cols
):

    results = []

    for col in categorical_cols:

        real_counts = real_df[col].astype(str).value_counts()
        synth_counts = synth_df[col].astype(str).value_counts()

        all_categories = real_counts.index.union(
            synth_counts.index
        )

        real_freq = real_counts.reindex(
            all_categories,
            fill_value=0
        )

        synth_freq = synth_counts.reindex(
            all_categories,
            fill_value=0
        )

        contingency_table = np.array([
            real_freq.values,
            synth_freq.values
        ])

        try:

            chi2, p_value, _, _ = chi2_contingency(
                contingency_table
            )

        except ValueError:

            chi2 = np.nan
            p_value = np.nan

        results.append({
            "column": col,
            "chi2": chi2,
            "p_value": p_value
        })

    return pd.DataFrame(results)


# --------------------------------------------------
# 3) PEARSON CORRELATION
# --------------------------------------------------

def correlation_metrics(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    numerical_cols
):

    if len(numerical_cols) == 0:

        empty_df = pd.DataFrame()

        return (
            np.nan,
            np.nan,
            empty_df,
            empty_df,
            empty_df
        )

    real_corr = real_df[numerical_cols].corr()
    synth_corr = synth_df[numerical_cols].corr()

    diff = (real_corr - synth_corr).abs()

    mae = diff.mean().mean()
    mse = (diff ** 2).mean().mean()

    return (
        mae,
        mse,
        real_corr,
        synth_corr,
        diff
    )


# --------------------------------------------------
# 4) WASSERSTEIN DISTANCE
# --------------------------------------------------

def wasserstein_test(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    numerical_cols
):

    results = []

    for col in numerical_cols:

        real_series = real_df[col].dropna()
        synth_series = synth_df[col].dropna()

        if len(real_series) == 0 or len(synth_series) == 0:

            results.append({
                "column": col,
                "wasserstein": np.nan
            })

            continue

        wd = wasserstein_distance(
            real_series,
            synth_series
        )

        results.append({
            "column": col,
            "wasserstein": wd
        })

    return pd.DataFrame(results)


# --------------------------------------------------
# 5) JENSEN-SHANNON DISTANCE
# --------------------------------------------------

def js_distance_test(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    categorical_cols
):

    results = []

    for col in categorical_cols:

        real_counts = (
            real_df[col]
            .astype(str)
            .value_counts(normalize=True)
        )

        synth_counts = (
            synth_df[col]
            .astype(str)
            .value_counts(normalize=True)
        )

        all_categories = real_counts.index.union(
            synth_counts.index
        )

        real_probs = real_counts.reindex(
            all_categories,
            fill_value=0
        ).values

        synth_probs = synth_counts.reindex(
            all_categories,
            fill_value=0
        ).values

        # zero protection
        real_probs = real_probs + 1e-10
        synth_probs = synth_probs + 1e-10

        # normalization
        real_probs = real_probs / real_probs.sum()
        synth_probs = synth_probs / synth_probs.sum()

        js = jensenshannon(
            real_probs,
            synth_probs
        )

        results.append({
            "column": col,
            "js_distance": js
        })

    return pd.DataFrame(results)


# --------------------------------------------------
# 6) CRAMER'S V
# --------------------------------------------------

def cramers_v(confusion_matrix: np.ndarray):

    if confusion_matrix.size == 0:
        return np.nan

    try:

        chi2 = chi2_contingency(
            confusion_matrix
        )[0]

    except ValueError:

        return np.nan

    n = confusion_matrix.sum()

    if n == 0:
        return np.nan

    r, k = confusion_matrix.shape

    denom = min(k - 1, r - 1)

    if denom <= 0:
        return 0.0

    return np.sqrt(
        chi2 / (n * denom)
    )


def cramers_v_matrix(
    df: pd.DataFrame,
    categorical_cols
):

    n = len(categorical_cols)

    if n == 0:
        return pd.DataFrame()

    matrix = pd.DataFrame(
        np.zeros((n, n)),
        index=categorical_cols,
        columns=categorical_cols
    )

    for i, col1 in enumerate(categorical_cols):

        for j, col2 in enumerate(categorical_cols):

            if i <= j:

                confusion = pd.crosstab(
                    df[col1].astype(str),
                    df[col2].astype(str)
                )

                value = cramers_v(
                    confusion.values
                )

                matrix.loc[col1, col2] = value
                matrix.loc[col2, col1] = value

    return matrix


# --------------------------------------------------
# 7) SAVE FUNCTIONS
# --------------------------------------------------

def save_text_metrics(
    path: Path,
    metrics: dict
):

    with open(path, "w", encoding="utf-8") as f:

        for key, value in metrics.items():

            f.write(f"{key}={value}\n")


def save_heatmap(
    matrix: pd.DataFrame,
    title: str,
    save_path: Path,
    cmap: str = "coolwarm",
    center=None
):

    if matrix is None or matrix.empty:
        return

    fig_w = min(
        20,
        max(8, len(matrix.columns) * 0.6)
    )

    fig_h = min(
        16,
        max(6, len(matrix.index) * 0.6)
    )

    plt.figure(figsize=(fig_w, fig_h))

    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f",
        cmap=cmap,
        center=center,
        linewidths=0.5,
        square=True,
        cbar=True,
        annot_kws={"size": 8}
    )

    plt.title(title, fontsize=14)

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def save_bar_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    save_path: Path,
    rotate_xticks: bool = True
):

    if df is None or df.empty:
        return

    if x_col not in df.columns or y_col not in df.columns:
        return

    plot_df = df[[x_col, y_col]].dropna().copy()

    if plot_df.empty:
        return

    plot_df = plot_df.sort_values(
        y_col,
        ascending=False
    )

    plt.figure(figsize=(10, 6))

    plt.bar(
        plot_df[x_col].astype(str),
        plot_df[y_col]
    )

    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)

    if rotate_xticks:

        plt.xticks(
            rotation=45,
            ha="right"
        )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# --------------------------------------------------
# 8) PCA PLOT
# --------------------------------------------------

def save_pca_plot(
    real_df,
    synth_df,
    numerical_cols,
    title,
    save_path,
    csv_path
):

    if len(numerical_cols) == 0:
        return

    real = real_df[numerical_cols].dropna()
    synth = synth_df[numerical_cols].dropna()

    min_len = min(
        len(real),
        len(synth)
    )

    # safety check
    if min_len < 2:
        return

    real = real.sample(
        min_len,
        random_state=42
    )

    synth = synth.sample(
        min_len,
        random_state=42
    )

    scaler = StandardScaler()

    real_scaled = scaler.fit_transform(real)
    synth_scaled = scaler.transform(synth)

    pca = PCA(
        n_components=2,
        random_state=42
    )

    real_pca = pca.fit_transform(real_scaled)
    synth_pca = pca.transform(synth_scaled)

    explained_ratio = pca.explained_variance_ratio_

    pca_df = pd.DataFrame({
        "x": np.concatenate([
            real_pca[:, 0],
            synth_pca[:, 0]
        ]),
        "y": np.concatenate([
            real_pca[:, 1],
            synth_pca[:, 1]
        ]),
        "label": (
            ["Real"] * len(real_pca) +
            ["Synthetic"] * len(synth_pca)
        )
    })

    pca_df.to_csv(
        csv_path,
        index=False
    )

    with open(
        csv_path.with_name(
            csv_path.stem + "_variance.txt"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            f"PC1 Explained Variance Ratio = {explained_ratio[0]:.4f}\n"
        )

        f.write(
            f"PC2 Explained Variance Ratio = {explained_ratio[1]:.4f}\n"
        )

        f.write(
            f"Total Explained Variance = {(explained_ratio[0] + explained_ratio[1]):.4f}\n"
        )

    plt.figure(figsize=(8, 6))

    plt.scatter(
        real_pca[:, 0],
        real_pca[:, 1],
        alpha=0.5,
        label="Real"
    )

    plt.scatter(
        synth_pca[:, 0],
        synth_pca[:, 1],
        alpha=0.5,
        label="Synthetic"
    )

    plt.legend()

    plt.title(title)

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# --------------------------------------------------
# 9) TSNE PLOT
# --------------------------------------------------

def save_tsne_plot(
    real_df,
    synth_df,
    numerical_cols,
    title,
    save_path,
    csv_path
):

    if len(numerical_cols) == 0:
        return

    real = real_df[numerical_cols].dropna()
    synth = synth_df[numerical_cols].dropna()

    min_len = min(
        len(real),
        len(synth),
        1000
    )

    # safety check
    if min_len < 5:
        return

    real = real.sample(
        min_len,
        random_state=42
    )

    synth = synth.sample(
        min_len,
        random_state=42
    )

    combined = pd.concat(
        [real, synth],
        axis=0
    )

    scaler = StandardScaler()

    combined_scaled = scaler.fit_transform(combined)

    perplexity = min(
        30,
        max(5, min_len // 3)
    )

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=42
    )

    embedding = tsne.fit_transform(
        combined_scaled
    )

    real_emb = embedding[:len(real)]
    synth_emb = embedding[len(real):]

    tsne_df = pd.DataFrame({
        "x": embedding[:, 0],
        "y": embedding[:, 1],
        "label": (
            ["Real"] * len(real_emb) +
            ["Synthetic"] * len(synth_emb)
        )
    })

    tsne_df.to_csv(
        csv_path,
        index=False
    )

    plt.figure(figsize=(8, 6))

    plt.scatter(
        real_emb[:, 0],
        real_emb[:, 1],
        alpha=0.5,
        label="Real"
    )

    plt.scatter(
        synth_emb[:, 0],
        synth_emb[:, 1],
        alpha=0.5,
        label="Synthetic"
    )

    plt.legend()

    plt.title(title)

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# --------------------------------------------------
# 10) EVALUATION
# --------------------------------------------------

def evaluate_model_on_dataset(
    dataset_name: str,
    model_name: str,
    dataset_cfg: dict
):

    print(
        f"\n--- {dataset_name.upper()} | {model_name.upper()} ---"
    )

    real_path = dataset_cfg["benchmark_path"]

    synth_path = (
        dataset_cfg["synthetic_dir"] /
        f"{model_name}.csv"
    )

    if not real_path.exists():

        raise FileNotFoundError(
            f"Gerçek veri bulunamadı: {real_path}"
        )

    if not synth_path.exists():

        raise FileNotFoundError(
            f"Sentetik veri bulunamadı: {synth_path}"
        )

    real_df = pd.read_csv(real_path)
    synth_df = pd.read_csv(synth_path)

    target_col = dataset_cfg["target_column"]

    if target_col in real_df.columns:
        real_df[target_col] = real_df[target_col].astype(str)

    if target_col in synth_df.columns:
        synth_df[target_col] = synth_df[target_col].astype(str)

    common_columns = [
        col for col in real_df.columns
        if col in synth_df.columns
    ]

    real_df = real_df[common_columns].copy()
    synth_df = synth_df[common_columns].copy()

    categorical_cols, numerical_cols = get_column_types(
        real_df
    )

    print("Categorical:", categorical_cols)
    print("Numerical:", numerical_cols)

    save_dir = STATS_DIR / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)

    plot_dir = PLOTS_DIR / "stats" / dataset_name
    plot_dir.mkdir(parents=True, exist_ok=True)

    # 1) KS TEST

    ks_df = ks_test(
        real_df,
        synth_df,
        numerical_cols
    )

    ks_df.to_csv(
        save_dir / f"{model_name}_ks.csv",
        index=False
    )

    save_bar_plot(
        ks_df,
        x_col="column",
        y_col="ks_stat",
        title=f"{dataset_name} - {model_name} KS Statistic by Column",
        save_path=plot_dir / f"{model_name}_ks_bar.png"
    )

    # 2) CHI-SQUARE

    chi_df = chi_square_test(
        real_df,
        synth_df,
        categorical_cols
    )

    chi_df.to_csv(
        save_dir / f"{model_name}_chi2.csv",
        index=False
    )

    save_bar_plot(
        chi_df,
        x_col="column",
        y_col="p_value",
        title=f"{dataset_name} - {model_name} Chi-Square p-value by Column",
        save_path=plot_dir / f"{model_name}_chi2_pvalue_bar.png"
    )

    save_bar_plot(
        chi_df,
        x_col="column",
        y_col="chi2",
        title=f"{dataset_name} - {model_name} Chi-Square Statistic by Column",
        save_path=plot_dir / f"{model_name}_chi2_stat_bar.png"
    )

    # 3) CORRELATION

    corr_mae, corr_mse, real_corr, synth_corr, corr_diff = (
        correlation_metrics(
            real_df,
            synth_df,
            numerical_cols
        )
    )

    if not real_corr.empty:

        if not (
            save_dir / "real_corr.csv"
        ).exists():

            real_corr.to_csv(
                save_dir / "real_corr.csv"
            )

            save_heatmap(
                real_corr,
                title=f"{dataset_name} Real Correlation",
                save_path=plot_dir / "real_corr_heatmap.png",
                cmap="coolwarm",
                center=0
            )

        synth_corr.to_csv(
            save_dir / f"{model_name}_corr_synth.csv"
        )

        corr_diff.to_csv(
            save_dir / f"{model_name}_corr_diff.csv"
        )

    save_heatmap(
        synth_corr,
        title=f"{dataset_name} - {model_name} Synthetic Correlation",
        save_path=plot_dir / f"{model_name}_corr_synth_heatmap.png",
        cmap="coolwarm",
        center=0
    )

    save_heatmap(
        corr_diff,
        title=f"{dataset_name} - {model_name} Correlation Difference",
        save_path=plot_dir / f"{model_name}_corr_diff_heatmap.png",
        cmap="coolwarm",
        center=0
    )

    save_text_metrics(
        save_dir / f"{model_name}_corr_metrics.txt",
        {
            "mae": corr_mae,
            "mse": corr_mse
        }
    )

    # 4) WASSERSTEIN

    wasser_df = wasserstein_test(
        real_df,
        synth_df,
        numerical_cols
    )

    wasser_df.to_csv(
        save_dir / f"{model_name}_wasserstein.csv",
        index=False
    )

    save_bar_plot(
        wasser_df,
        x_col="column",
        y_col="wasserstein",
        title=f"{dataset_name} - {model_name} Wasserstein Distance",
        save_path=plot_dir / f"{model_name}_wasserstein_bar.png"
    )

    wasser_avg = wasser_df["wasserstein"].mean()

    # 5) JS DISTANCE

    js_df = js_distance_test(
        real_df,
        synth_df,
        categorical_cols
    )

    js_df.to_csv(
        save_dir / f"{model_name}_js_distance.csv",
        index=False
    )

    save_bar_plot(
        js_df,
        x_col="column",
        y_col="js_distance",
        title=f"{dataset_name} - {model_name} JS Distance",
        save_path=plot_dir / f"{model_name}_js_distance_bar.png"
    )

    js_avg = js_df["js_distance"].mean()

    # 6) CRAMER'S V

    if len(categorical_cols) > 0:

        real_cramer = cramers_v_matrix(
            real_df,
            categorical_cols
        )

        synth_cramer = cramers_v_matrix(
            synth_df,
            categorical_cols
        )

        cramer_diff = (
            real_cramer - synth_cramer
        ).abs()

        if not (
            save_dir / "real_cramer.csv"
        ).exists():

            real_cramer.to_csv(
                save_dir / "real_cramer.csv"
            )

            save_heatmap(
                real_cramer,
                title=f"{dataset_name} Real Cramer's V",
                save_path=plot_dir / "real_cramer_heatmap.png",
                cmap="YlOrRd"
            )

        synth_cramer.to_csv(
            save_dir / f"{model_name}_cramer_synth.csv"
        )

        cramer_diff.to_csv(
            save_dir / f"{model_name}_cramer_diff.csv"
        )

        save_heatmap(
            synth_cramer,
            title=f"{dataset_name} - {model_name} Synthetic Cramer's V",
            save_path=plot_dir / f"{model_name}_cramer_synth_heatmap.png",
            cmap="YlOrRd"
        )

        save_heatmap(
            cramer_diff,
            title=f"{dataset_name} - {model_name} Cramer's V Difference",
            save_path=plot_dir / f"{model_name}_cramer_diff_heatmap.png",
            cmap="Reds"
        )

        cramer_mean_diff = (
            cramer_diff.mean().mean()
        )

    else:

        cramer_mean_diff = np.nan

    # 7) PCA

    save_pca_plot(
        real_df,
        synth_df,
        numerical_cols,
        title=f"{dataset_name} - {model_name} PCA",
        save_path=plot_dir / f"{model_name}_pca.png",
        csv_path=save_dir / f"{model_name}_pca.csv"
    )

    # 8) TSNE

    save_tsne_plot(
        real_df,
        synth_df,
        numerical_cols,
        title=f"{dataset_name} - {model_name} t-SNE",
        save_path=plot_dir / f"{model_name}_tsne.png",
        csv_path=save_dir / f"{model_name}_tsne.csv"
    )

    # SUMMARY

    ks_avg = (
        ks_df["ks_stat"].mean()
        if not ks_df.empty
        else np.nan
    )

    chi2_avg = (
        chi_df["chi2"].mean()
        if not chi_df.empty
        else np.nan
    )

    save_text_metrics(
        save_dir / f"{model_name}_summary.txt",
        {
            "ks_avg": ks_avg,
            "chi2_avg": chi2_avg,
            "corr_mae": corr_mae,
            "corr_mse": corr_mse,
            "wasserstein_avg": wasser_avg,
            "js_avg": js_avg,
            "cramer_mean_diff": cramer_mean_diff
        }
    )

    print(
        f"KS ortalama: {ks_avg:.4f}"
        if pd.notna(ks_avg)
        else "KS ortalama: NaN"
    )

    print(
        f"Chi-square ortalama: {chi2_avg:.4f}"
        if pd.notna(chi2_avg)
        else "Chi-square ortalama: NaN"
    )

    print(
        f"Korelasyon MAE: {corr_mae:.4f}"
        if pd.notna(corr_mae)
        else "Korelasyon MAE: NaN"
    )

    print(
        f"Korelasyon MSE: {corr_mse:.4f}"
        if pd.notna(corr_mse)
        else "Korelasyon MSE: NaN"
    )

    print(
        f"Wasserstein ortalama: {wasser_avg:.4f}"
        if pd.notna(wasser_avg)
        else "Wasserstein ortalama: NaN"
    )

    print(
        f"JS Distance ortalama: {js_avg:.4f}"
        if pd.notna(js_avg)
        else "JS Distance ortalama: NaN"
    )

    print(
        f"Cramer's V fark ortalama: {cramer_mean_diff:.4f}"
        if pd.notna(cramer_mean_diff)
        else "Cramer's V fark ortalama: NaN"
    )


# --------------------------------------------------
# 11) MAIN
# --------------------------------------------------

def main():

    for dataset_name, dataset_cfg in DATASETS.items():

        print(
            f"\n==================== {dataset_name.upper()} ===================="
        )

        for model_name in MODELS:

            try:

                evaluate_model_on_dataset(
                    dataset_name,
                    model_name,
                    dataset_cfg
                )

            except Exception as e:

                print(
                    f"HATA | {dataset_name} | {model_name}: {e}"
                )


if __name__ == "__main__":
    main()