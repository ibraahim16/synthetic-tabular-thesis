import sys
import time
import json
import pickle
import traceback
from pathlib import Path
import gc
import joblib

import numpy as np
import pandas as pd
from ForestDiffusion import ForestDiffusionModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import DATASETS, EPOCHS, BATCH_SIZE, RANDOM_STATE

def format_seconds(seconds: float) -> str:
    return f"{seconds:.2f} sn / {seconds / 60:.2f} dk"


def print_step(dataset_name: str, step: str, message: str) -> None:
    print(f"[{dataset_name.upper()}] {step} | {message}", flush=True)


def get_categorical_indexes(df: pd.DataFrame, target_col: str) -> list[int]:
    cat_indexes: list[int] = []

    for idx, col in enumerate(df.columns):
        if (
            pd.api.types.is_object_dtype(df[col])
            or isinstance(df[col].dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(df[col])
        ):
            cat_indexes.append(idx)

    if target_col in df.columns:
        class_idx = list(df.columns).index(target_col)
        if class_idx not in cat_indexes:
            cat_indexes.append(class_idx)

    return sorted(set(cat_indexes))


def get_integer_indexes(df: pd.DataFrame, cat_indexes: list[int]) -> list[int]:
    int_indexes: list[int] = []

    for idx, col in enumerate(df.columns):
        if idx not in cat_indexes and pd.api.types.is_integer_dtype(df[col]):
            int_indexes.append(idx)

    return sorted(set(int_indexes))


def encode_categoricals(df: pd.DataFrame, cat_indexes: list[int]) -> tuple[np.ndarray, dict]:
    df_enc = df.copy()
    mappings = {}

    for idx in cat_indexes:
        col = df_enc.columns[idx]
        series = df_enc[col].astype(str)
        categories = sorted(series.unique().tolist())

        mapping = {c: i for i, c in enumerate(categories)}

        mappings[idx] = {
            "col": col,
            "mapping": mapping,
            "inv": {i: c for c, i in mapping.items()},
        }

        df_enc[col] = series.map(mapping).astype(int)

    return df_enc.to_numpy(dtype=float), mappings


def decode_categoricals(
    X_syn: np.ndarray,
    columns: list[str],
    mappings: dict,
    int_indexes: list[int],
) -> pd.DataFrame:
    df = pd.DataFrame(X_syn, columns=columns)

    for idx, info in mappings.items():
        col = info["col"]
        inv = info["inv"]

        codes = np.rint(df[col].to_numpy()).astype(int)
        codes = np.clip(codes, 0, max(inv.keys()))

        df[col] = [inv[c] for c in codes]

    for idx in int_indexes:
        col = columns[idx]
        df[col] = np.rint(df[col].to_numpy()).astype(int)

    return df

def save_metadata(
    info_path: Path,
    dataset_name: str,
    train_path: Path,
    benchmark_path: Path,
    target_col: str,
    train_shape: tuple[int, int],
    benchmark_shape: tuple[int, int],
    cat_indexes: list[int],
    int_indexes: list[int],
    columns: list[str],
    backend: str,
    timings: dict,
) -> None:

    metadata = {
        "dataset_name": dataset_name,
        "model_name": "forest_diffusion",
        "train_path": str(train_path),
        "benchmark_path": str(benchmark_path),
        "target_column": target_col,
        "train_shape": train_shape,
        "benchmark_shape": benchmark_shape,
        "columns": columns,
        "categorical_indexes": cat_indexes,
        "integer_indexes": int_indexes,
        "epochs_n_t": EPOCHS,
        "batch_size": BATCH_SIZE,
        "random_state": RANDOM_STATE,
        "backend": backend,
        "timings_seconds": timings,
    }

    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def train_forest_diffusion_for_dataset(dataset_name: str, dataset_config: dict) -> None:
    total_start = time.time()
    timings: dict[str, float] = {}

    print(f"\n=== FOREST-DIFFUSION | {dataset_name.upper()} ===", flush=True)

    train_path = dataset_config["train_path"]
    benchmark_path = dataset_config["benchmark_path"]
    synthetic_dir = dataset_config["synthetic_dir"]
    model_dir = dataset_config["models_dir"] / "forest_diffusion"
    target_col = dataset_config["target_column"]

    synthetic_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    if not train_path.exists():
        raise FileNotFoundError(f"Train veri bulunamadı: {train_path}")

    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark veri bulunamadı: {benchmark_path}")

    step_start = time.time()
    print_step(dataset_name, "1/6", "Veri okunuyor...")

    train_df = pd.read_csv(train_path)
    benchmark_df = pd.read_csv(benchmark_path)

    timings["read_data"] = time.time() - step_start

    print_step(dataset_name, "1/6", f"Veri okundu | süre={format_seconds(timings['read_data'])}")
    print_step(dataset_name, "INFO", f"Train veri boyutu: {train_df.shape}")
    print_step(dataset_name, "INFO", f"Benchmark veri boyutu: {benchmark_df.shape}")
    print_step(dataset_name, "INFO", f"Hedef sütun: {target_col}")

    step_start = time.time()
    print_step(dataset_name, "2/6", "Ön işleme başlıyor...")

    cat_indexes = get_categorical_indexes(train_df, target_col)
    int_indexes = get_integer_indexes(train_df, cat_indexes)
    X_train, mappings = encode_categoricals(train_df, cat_indexes)

    timings["preprocess"] = time.time() - step_start

    print_step(dataset_name, "2/6", f"Ön işleme bitti | süre={format_seconds(timings['preprocess'])}")
    print_step(dataset_name, "INFO", f"Kategorik indexler: {cat_indexes}")
    print_step(dataset_name, "INFO", f"Integer indexler: {int_indexes}")
    print_step(dataset_name, "INFO", f"Model giriş matrisi: {X_train.shape}")

    step_start = time.time()

    print_step(
        dataset_name,
        "3/6",
        f"Eğitim başlıyor | n_t={EPOCHS} | batch_size={BATCH_SIZE} | backend=cpu/default",
    )

    model = ForestDiffusionModel(
        X=X_train,
        n_t=EPOCHS,
        model="xgboost",
        cat_indexes=cat_indexes,
        int_indexes=int_indexes,
        n_batch=BATCH_SIZE,
        seed=RANDOM_STATE,
    )

    backend = "cpu/default"
    timings["training"] = time.time() - step_start

    print_step(
        dataset_name,
        "3/6",
        f"Eğitim tamamlandı | backend={backend} | süre={format_seconds(timings['training'])}",
    )

    step_start = time.time()
    print_step(dataset_name, "4/6", "Model ve bilgi dosyaları kaydediliyor...")

    model_path = model_dir / f"{dataset_name}_forest_diffusion.pkl"
    mappings_path = model_dir / f"{dataset_name}_forest_diffusion_mappings.pkl"
    info_path = model_dir / f"{dataset_name}_forest_diffusion_info.json"

    gc.collect()
    joblib.dump(model, model_path, compress=0)
    gc.collect()
    with open(mappings_path, "wb") as f:
        pickle.dump(mappings, f)
    gc.collect()

    timings["save_model"] = time.time() - step_start

    print_step(dataset_name, "4/6", f"Model kaydedildi: {model_path}")
    print_step(dataset_name, "4/6", f"Mapping kaydedildi: {mappings_path}")
    print_step(dataset_name, "4/6", f"Model kayıt süresi={format_seconds(timings['save_model'])}")

    step_start = time.time()
    print_step(dataset_name, "5/6", f"Sentetik üretim başlıyor | n={len(benchmark_df)}")

    X_syn = model.generate(batch_size=len(benchmark_df))

    timings["generate"] = time.time() - step_start

    print_step(dataset_name, "5/6", f"Sentetik üretim bitti | süre={format_seconds(timings['generate'])}")

    step_start = time.time()
    print_step(dataset_name, "6/6", "Decode ve CSV kaydı başlıyor...")

    synthetic_df = decode_categoricals(
        X_syn=X_syn,
        columns=list(train_df.columns),
        mappings=mappings,
        int_indexes=int_indexes,
    )

    synthetic_df = synthetic_df[train_df.columns]

    synthetic_path = synthetic_dir / "forest_diffusion.csv"
    synthetic_df.to_csv(synthetic_path, index=False)

    timings["save_synthetic"] = time.time() - step_start
    timings["total"] = time.time() - total_start

    save_metadata(
        info_path=info_path,
        dataset_name=dataset_name,
        train_path=train_path,
        benchmark_path=benchmark_path,
        target_col=target_col,
        train_shape=train_df.shape,
        benchmark_shape=benchmark_df.shape,
        cat_indexes=cat_indexes,
        int_indexes=int_indexes,
        columns=list(train_df.columns),
        backend=backend,
        timings=timings,
    )

    print_step(dataset_name, "6/6", f"Sentetik veri kaydedildi: {synthetic_path}")
    print_step(dataset_name, "6/6", f"Sentetik veri boyutu: {synthetic_df.shape}")
    print_step(dataset_name, "6/6", f"Bilgi dosyası kaydedildi: {info_path}")
    print_step(dataset_name, "6/6", f"CSV kayıt süresi={format_seconds(timings['save_synthetic'])}")

    print_step(dataset_name, "BİTTİ", f"Toplam süre={format_seconds(timings['total'])}")


def main() -> None:
    global_start = time.time()

    for dataset_name, dataset_config in DATASETS.items():
        try:
            train_forest_diffusion_for_dataset(dataset_name, dataset_config)

        except Exception as e:
            print(f"\nHATA | {dataset_name} | forest_diffusion: {e}", flush=True)
            traceback.print_exc()

    global_time = time.time() - global_start

    print(
        f"\n=== TÜM FOREST-DIFFUSION ÇALIŞMASI BİTTİ | toplam süre={format_seconds(global_time)} ===",
        flush=True,
    )


if __name__ == "__main__":
    main()