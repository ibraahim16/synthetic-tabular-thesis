import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(r"/synthetic_thesis/data")

REAL_FILES = {
    "adult": BASE / "benchmark" / "adult_20k.csv",
    "credit": BASE / "benchmark" / "credit_20k.csv",
}

SYN_FILES = {
    "adult": {
        "cgan": BASE / "synthetic" / "adult" / "cgan.csv",
        "copulagan": BASE / "synthetic" / "adult" / "copulagan.csv",
        "ctabgan_plus": BASE / "synthetic" / "adult" / "ctabgan_plus.csv",
        "ctgan": BASE / "synthetic" / "adult" / "ctgan.csv",
        "tabsyn": BASE / "synthetic" / "adult" / "tabsyn.csv",
    },
    "credit": {
        "cgan": BASE / "synthetic" / "credit" / "cgan.csv",
        "copulagan": BASE / "synthetic" / "credit" / "copulagan.csv",
        "ctabgan_plus": BASE / "synthetic" / "credit" / "ctabgan_plus.csv",
        "ctgan": BASE / "synthetic" / "credit" / "ctgan.csv",
        "tabsyn": BASE / "synthetic" / "credit" / "tabsyn.csv",
    },
}

TARGET_COLS = {
    "adult": "income",
    "credit": "Class",
}


def print_header(text: str):
    print("\n" + "=" * 80)
    print(text)
    print("=" * 80)


def safe_value_counts(series: pd.Series, normalize: bool = True):
    vc = series.value_counts(normalize=normalize, dropna=False)
    return vc


def compare_target_distribution(real_df: pd.DataFrame, syn_df: pd.DataFrame, target_col: str):
    print(f"\n[Target dağılımı] {target_col}")

    real_dist = safe_value_counts(real_df[target_col], normalize=True)
    syn_dist = safe_value_counts(syn_df[target_col], normalize=True)

    all_idx = real_dist.index.union(syn_dist.index)
    comp = pd.DataFrame({
        "real_ratio": real_dist.reindex(all_idx, fill_value=0.0),
        "syn_ratio": syn_dist.reindex(all_idx, fill_value=0.0),
    })
    comp["abs_diff"] = (comp["real_ratio"] - comp["syn_ratio"]).abs()

    print(comp)
    print(f"\nToplam mutlak fark: {comp['abs_diff'].sum():.6f}")


def compare_numeric_summary(real_df: pd.DataFrame, syn_df: pd.DataFrame, dataset_name: str):
    numeric_cols = real_df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        print("\n[Numeric range] Sayısal sütun yok.")
        return

    print("\n[Numeric range / özet karşılaştırma]")

    rows = []
    for col in numeric_cols:
        if col not in syn_df.columns:
            rows.append({
                "column": col,
                "real_min": np.nan,
                "syn_min": np.nan,
                "real_max": np.nan,
                "syn_max": np.nan,
                "real_mean": np.nan,
                "syn_mean": np.nan,
                "real_std": np.nan,
                "syn_std": np.nan,
                "mean_abs_diff": np.nan,
            })
            continue

        rows.append({
            "column": col,
            "real_min": real_df[col].min(),
            "syn_min": syn_df[col].min(),
            "real_max": real_df[col].max(),
            "syn_max": syn_df[col].max(),
            "real_mean": real_df[col].mean(),
            "syn_mean": syn_df[col].mean(),
            "real_std": real_df[col].std(),
            "syn_std": syn_df[col].std(),
            "mean_abs_diff": abs(real_df[col].mean() - syn_df[col].mean()),
        })

    comp_df = pd.DataFrame(rows)
    print(comp_df.to_string(index=False))

    print("\nEn çok mean sapması olan ilk 10 sütun:")
    print(comp_df.sort_values("mean_abs_diff", ascending=False).head(10).to_string(index=False))


def compare_categorical_summary(real_df: pd.DataFrame, syn_df: pd.DataFrame, top_n: int = 5):
    cat_cols = real_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    if not cat_cols:
        print("\n[Kategorik dağılım] Kategorik sütun yok.")
        return

    print("\n[Kategorik dağılım karşılaştırması]")

    for col in cat_cols:
        if col not in syn_df.columns:
            print(f"\n--- {col} ---")
            print("Sentetik veride sütun yok.")
            continue

        real_dist = real_df[col].value_counts(normalize=True, dropna=False)
        syn_dist = syn_df[col].value_counts(normalize=True, dropna=False)

        all_idx = real_dist.index.union(syn_dist.index)
        comp = pd.DataFrame({
            "real_ratio": real_dist.reindex(all_idx, fill_value=0.0),
            "syn_ratio": syn_dist.reindex(all_idx, fill_value=0.0),
        })
        comp["abs_diff"] = (comp["real_ratio"] - comp["syn_ratio"]).abs()

        print(f"\n--- {col} ---")
        print(comp.sort_values("abs_diff", ascending=False).head(top_n).to_string())


def sanity_checks(real_df: pd.DataFrame, syn_df: pd.DataFrame):
    print("\n[Temel sanity check]")
    print(f"Gerçek shape   : {real_df.shape}")
    print(f"Sentetik shape : {syn_df.shape}")
    print(f"NaN sayısı     : {syn_df.isna().sum().sum()}")
    print(f"Duplicate sayısı: {syn_df.duplicated().sum()}")

    real_cols = list(real_df.columns)
    syn_cols = list(syn_df.columns)

    print(f"\nKolon sayısı gerçek : {len(real_cols)}")
    print(f"Kolon sayısı sentetik: {len(syn_cols)}")

    missing_in_syn = [c for c in real_cols if c not in syn_cols]
    extra_in_syn = [c for c in syn_cols if c not in real_cols]

    print(f"Eksik kolonlar : {missing_in_syn}")
    print(f"Fazla kolonlar : {extra_in_syn}")


def check_one_dataset(dataset_name: str):
    target_col = TARGET_COLS[dataset_name]
    real_df = pd.read_csv(REAL_FILES[dataset_name])

    print_header(f"{dataset_name.upper()} | GERÇEK VERİ")
    print(real_df.head())
    print(f"\nGerçek veri shape: {real_df.shape}")

    for model_name, syn_path in SYN_FILES[dataset_name].items():
        if not syn_path.exists():
            print_header(f"{dataset_name.upper()} | {model_name.upper()} | DOSYA YOK")
            print(syn_path)
            continue

        syn_df = pd.read_csv(syn_path)

        print_header(f"{dataset_name.upper()} | {model_name.upper()}")
        print(syn_df.head())

        sanity_checks(real_df, syn_df)
        compare_target_distribution(real_df, syn_df, target_col)
        compare_numeric_summary(real_df, syn_df, dataset_name)

        # Adult için kategorik karşılaştırma çok önemli
        if dataset_name == "adult":
            compare_categorical_summary(real_df, syn_df, top_n=5)


def main():
    for dataset_name in ["adult", "credit"]:
        check_one_dataset(dataset_name)


if __name__ == "__main__":
    main()