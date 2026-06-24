import sys
from pathlib import Path

# Proje kökünü ve src klasörünü Python path'e ekle
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import CopulaGANSynthesizer

from config import DATASETS, EPOCHS, BATCH_SIZE


def build_metadata(df: pd.DataFrame) -> Metadata:
    return Metadata.detect_from_dataframe(data=df)


def train_copulagan_for_dataset(dataset_name: str, dataset_config: dict) -> None:
    print(f"\n=== COPULAGAN | {dataset_name.upper()} ===")

    train_path = dataset_config["train_path"]
    benchmark_path = dataset_config["benchmark_path"]
    synthetic_dir = dataset_config["synthetic_dir"]
    model_dir = dataset_config["models_dir"] / "copulagan"

    synthetic_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(train_path)
    benchmark_df = pd.read_csv(benchmark_path)

    print(f"Train veri boyutu: {train_df.shape}")
    print(f"Benchmark veri boyutu: {benchmark_df.shape}")

    metadata = build_metadata(train_df)

    metadata_path = model_dir / f"{dataset_name}_metadata.json"
    if metadata_path.exists():
        metadata_path.unlink()

    metadata.save_to_json(filepath=str(metadata_path))
    print(f"Metadata kaydedildi: {metadata_path}")

    synthesizer = CopulaGANSynthesizer(
        metadata=metadata,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        pac=8,
        verbose=True,
    )

    print(f"Eğitim başlıyor | epoch={EPOCHS}, batch_size={BATCH_SIZE}")
    synthesizer.fit(train_df)

    model_path = model_dir / f"{dataset_name}_copulagan.pkl"
    synthesizer.save(filepath=str(model_path))
    print(f"Model kaydedildi: {model_path}")

    synthetic_df = synthesizer.sample(num_rows=len(benchmark_df))
    synthetic_path = synthetic_dir / "copulagan.csv"
    synthetic_df.to_csv(synthetic_path, index=False)

    print(f"Sentetik veri kaydedildi: {synthetic_path}")
    print(f"Sentetik veri boyutu: {synthetic_df.shape}")


def main() -> None:
    for dataset_name, dataset_config in DATASETS.items():
        train_copulagan_for_dataset(dataset_name, dataset_config)


if __name__ == "__main__":
    main()