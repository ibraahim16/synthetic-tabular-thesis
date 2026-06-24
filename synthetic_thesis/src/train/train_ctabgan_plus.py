import sys
import pickle
import json
from pathlib import Path

THESIS_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]

SRC_DIR = THESIS_ROOT / "src"
CTABGAN_ROOT = WORKSPACE_ROOT / "CTAB-GAN-Plus"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if str(CTABGAN_ROOT) not in sys.path:
    sys.path.insert(0, str(CTABGAN_ROOT))

import pandas as pd

from config import DATASETS
from model.ctabgan import CTABGAN


def get_categorical_columns(
    df: pd.DataFrame,
    target_col: str,
    regression_target: str | None = None,
) -> list[str]:
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    if regression_target and regression_target in categorical_cols:
        categorical_cols.remove(regression_target)

    if target_col == "Class" and "Class" not in categorical_cols:
        categorical_cols.append("Class")

    return categorical_cols


def get_integer_columns(dataset_name: str, df: pd.DataFrame) -> list[str]:
    if dataset_name == "adult":
        preferred = [
            "age",
            "fnlwgt",
            "capital_gain",
            "capital_loss",
            "hours_per_week",
            "capital-gain",
            "capital-loss",
            "hours-per-week",
        ]
        return [col for col in preferred if col in df.columns]

    integer_cols = []
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            integer_cols.append(col)
    return integer_cols


def build_problem_type(cfg: dict) -> dict:
    target_col = cfg["target_column"]
    return {"Classification": target_col}


def get_mixed_columns(dataset_name: str, df: pd.DataFrame) -> dict:
    """
    CTAB-GAN+ için özel değer yoğunluğu olan sütunlar.
    Adult veri setinde capital_gain ve capital_loss çoğunlukla 0 içerir.
    """
    if dataset_name == "adult":
        mixed_columns = {}

        if "capital_gain" in df.columns:
            mixed_columns["capital_gain"] = [0.0]
        if "capital_loss" in df.columns:
            mixed_columns["capital_loss"] = [0.0]

        # Bazı veri sürümlerinde tireli gelebilir, güvence için ek kontrol
        if "capital-gain" in df.columns:
            mixed_columns["capital-gain"] = [0.0]
        if "capital-loss" in df.columns:
            mixed_columns["capital-loss"] = [0.0]

        return mixed_columns

    return {}


def get_general_columns(dataset_name: str, df: pd.DataFrame, target_col: str) -> list[str]:
    """
    Normal sayısal sütunlar.
    Adult için daha kontrollü liste veriyoruz.
    Credit için hedef dışındaki sayısal sütunları alıyoruz.
    """
    if dataset_name == "adult":
        if "age" in df.columns:
            return ["age"]
        return []

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    return numeric_cols


def train_ctabgan_plus_for_dataset(dataset_name: str, cfg: dict) -> None:
    print(f"\n=== CTAB-GAN+ | {dataset_name.upper()} ===")

    benchmark_path = cfg["benchmark_path"]
    train_path = cfg["train_path"]
    synthetic_dir = cfg["synthetic_dir"]
    model_dir = cfg["models_dir"] / "ctabgan_plus"

    synthetic_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    if not benchmark_path.exists():
        raise FileNotFoundError(f"Girdi veri bulunamadı: {benchmark_path}")

    benchmark_df = pd.read_csv(benchmark_path)
    df = pd.read_csv(train_path)

    print(f"Benchmark veri boyutu: {benchmark_df.shape}")
    print(f"Train veri boyutu: {df.shape}")
    print(f"Sütunlar: {list(df.columns)}")

    target_col = cfg["target_column"]
    regression_target = cfg.get("regression_target")

    categorical_cols = get_categorical_columns(
        df=df,
        target_col=target_col,
        regression_target=regression_target,
    )
    integer_cols = get_integer_columns(dataset_name, df)
    problem_type = build_problem_type(cfg)
    mixed_columns = get_mixed_columns(dataset_name, df)
    general_columns = get_general_columns(dataset_name, df, target_col)

    print(f"Kategorik sütunlar: {categorical_cols}")
    print(f"Tam sayı sütunlar: {integer_cols}")
    print(f"Problem tipi: {problem_type}")
    print(f"Mixed sütunlar: {mixed_columns}")
    print(f"General sütunlar: {general_columns}")

    synthesizer = CTABGAN(
        raw_csv_path=str(train_path),
        test_ratio=0.0,
        categorical_columns=categorical_cols,
        log_columns=[],
        mixed_columns=mixed_columns,
        general_columns=general_columns,
        non_categorical_columns=[],
        integer_columns=integer_cols,
        problem_type=problem_type,
    )

    synthesizer.sample_size = len(benchmark_df)

    print("Eğitim başlıyor...")
    synthesizer.fit()
    print("Eğitim tamamlandı.")

    model_path = model_dir / f"{dataset_name}_ctabgan_plus.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(synthesizer, f)

    info_path = model_dir / f"{dataset_name}_ctabgan_plus_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "dataset_name": dataset_name,
                "benchmark_path": str(benchmark_path),
                "target_column": target_col,
                "categorical_columns": categorical_cols,
                "integer_columns": integer_cols,
                "problem_type": problem_type,
                "mixed_columns": mixed_columns,
                "general_columns": general_columns,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Model kaydedildi: {model_path}")
    print(f"Model bilgi dosyası kaydedildi: {info_path}")

    synthetic_df = synthesizer.generate_samples()

    # Gerçek veriyle aynı sütun sırasına hizala
    missing_cols = [col for col in df.columns if col not in synthetic_df.columns]
    extra_cols = [col for col in synthetic_df.columns if col not in df.columns]

    if missing_cols:
        print(f"UYARI - Sentetik veride eksik sütunlar: {missing_cols}")
    if extra_cols:
        print(f"UYARI - Sentetik veride fazladan sütunlar: {extra_cols}")

    common_cols = [col for col in df.columns if col in synthetic_df.columns]
    synthetic_df = synthetic_df[common_cols].copy()

    synthetic_path = synthetic_dir / "ctabgan_plus.csv"
    synthetic_df.to_csv(synthetic_path, index=False)

    print(f"Sentetik veri kaydedildi: {synthetic_path}")
    print(f"Sentetik veri boyutu: {synthetic_df.shape}")


def main() -> None:
    for dataset_name, cfg in DATASETS.items():
        train_ctabgan_plus_for_dataset(dataset_name, cfg)


if __name__ == "__main__":
    main()