import sys
from pathlib import Path
import subprocess
import shutil
import json

import pandas as pd
from pandas.api.types import CategoricalDtype

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import DATASETS

TABSYN_ROOT = PROJECT_ROOT.parent / "tabsyn"

def infer_task_type(dataset_name: str, cfg: dict) -> str:
    target_col = cfg["target_column"]
    if not target_col:
        raise ValueError(f"{dataset_name} için target_column tanımlı değil.")
    # Mevcut projede adult ve credit sınıflandırma
    return "binclass"


def is_categorical_series(series: pd.Series) -> bool:
    dtype = series.dtype
    return (
        pd.api.types.is_object_dtype(dtype)
        or isinstance(dtype, CategoricalDtype)
        or pd.api.types.is_bool_dtype(dtype)
    )

def build_metadata(df: pd.DataFrame, target_col: str):
    if target_col not in df.columns:
        raise ValueError(f"Hedef sütun veri içinde bulunamadı: {target_col}")

    column_names = list(df.columns)
    num_col_idx = []
    cat_col_idx = []

    for idx, col in enumerate(column_names):
        if col == target_col:
            continue

        if is_categorical_series(df[col]):
            cat_col_idx.append(idx)
        else:
            num_col_idx.append(idx)

    target_col_idx = [column_names.index(target_col)]
    return num_col_idx, cat_col_idx, target_col_idx


def write_tabsyn_inputs(
    tabsyn_dataset_name: str,
    df: pd.DataFrame,
    target_col: str,
    task_type: str,
) -> tuple[Path, Path]:
    data_dir = TABSYN_ROOT / "data" / tabsyn_dataset_name
    info_dir = TABSYN_ROOT / "data" / "Info"

    data_dir.mkdir(parents=True, exist_ok=True)
    info_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / f"{tabsyn_dataset_name}.csv"
    df.to_csv(csv_path, index=False)

    num_col_idx, cat_col_idx, target_col_idx = build_metadata(df, target_col)

    metadata = {
        "name": tabsyn_dataset_name,
        "task_type": task_type,
        "header": "infer",
        "column_names": None,
        "num_col_idx": num_col_idx,
        "cat_col_idx": cat_col_idx,
        "target_col_idx": target_col_idx,
        "file_type": "csv",
        "data_path": f"data/{tabsyn_dataset_name}/{tabsyn_dataset_name}.csv",
        "test_path": None,
        "use_pre_split": True,
        "dummy_test_size": 4000,
    }

    info_path = info_dir / f"{tabsyn_dataset_name}.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return csv_path, info_path


def run_command(cmd: list[str], cwd: Path) -> None:
    print("Çalıştırılıyor:", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)

def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)

    if src.is_file():
        shutil.copy2(src, dst)
        return True

    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        return True

    return False

def train_tabsyn_for_dataset(dataset_name: str, cfg: dict) -> None:
    print(f"\n=== TABSYN | {dataset_name.upper()} ===")

    benchmark_path = cfg["benchmark_path"]
    train_path = cfg["train_path"]
    synthetic_dir = cfg["synthetic_dir"]
    model_dir = cfg["models_dir"] / "tabsyn"
    target_col = cfg["target_column"]

    synthetic_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    if not benchmark_path.exists():
        raise FileNotFoundError(f"Benchmark veri bulunamadı: {benchmark_path}")

    if not train_path.exists():
        raise FileNotFoundError(f"Train veri bulunamadı: {train_path}")

    if not TABSYN_ROOT.exists():
        raise FileNotFoundError(f"TabSyn repo bulunamadı: {TABSYN_ROOT}")

    venv_python = TABSYN_ROOT / ".venv_tabsyn" / "Scripts" / "python.exe"
    if not venv_python.exists():
        raise FileNotFoundError(f"TabSyn python bulunamadı: {venv_python}")

    process_script = TABSYN_ROOT / "process_dataset.py"
    main_script = TABSYN_ROOT / "main.py"

    if not process_script.exists():
        raise FileNotFoundError(f"TabSyn process_dataset.py bulunamadı: {process_script}")

    if not main_script.exists():
        raise FileNotFoundError(f"TabSyn main.py bulunamadı: {main_script}")

    benchmark_df = pd.read_csv(benchmark_path)
    df = pd.read_csv(train_path)

    print(f"Benchmark veri boyutu: {benchmark_df.shape}")
    print(f"Girdi veri boyutu: {df.shape}")
    print(f"Hedef sütun: {target_col}")

    task_type = infer_task_type(dataset_name, cfg)
    tabsyn_dataset_name = f"{dataset_name}_tabsyn"

    csv_path, info_path = write_tabsyn_inputs(
        tabsyn_dataset_name=tabsyn_dataset_name,
        df=df,
        target_col=target_col,
        task_type=task_type,
    )

    print(f"TabSyn CSV hazırlandı: {csv_path}")
    print(f"TabSyn metadata hazırlandı: {info_path}")

    # Kritik düzeltme: custom dataset ismiyle işle
    run_command(
        [str(venv_python), "process_dataset.py", "--dataname", tabsyn_dataset_name],
        cwd=TABSYN_ROOT,
    )

    run_command(
        [
            str(venv_python),
            "main.py",
            "--dataname",
            tabsyn_dataset_name,
            "--method",
            "vae",
            "--mode",
            "train",
            "--gpu",
            "-1",
        ],
        cwd=TABSYN_ROOT,
    )

    run_command(
        [
            str(venv_python),
            "main.py",
            "--dataname",
            tabsyn_dataset_name,
            "--method",
            "tabsyn",
            "--mode",
            "train",
            "--gpu",
            "-1",
        ],
        cwd=TABSYN_ROOT,
    )

    run_command(
        [
            str(venv_python),
            "main.py",
            "--dataname",
            tabsyn_dataset_name,
            "--method",
            "tabsyn",
            "--mode",
            "sample",
            "--gpu",
            "-1",
            "--num-samples",
            str(len(benchmark_df)),
            "--save_path",
            str(TABSYN_ROOT / "synthetic" / tabsyn_dataset_name / "tabsyn.csv"),
        ],
        cwd=TABSYN_ROOT,
    )

    generated_path = TABSYN_ROOT / "synthetic" / tabsyn_dataset_name / "tabsyn.csv"
    if not generated_path.exists():
        raise FileNotFoundError(f"TabSyn çıktı dosyası bulunamadı: {generated_path}")

    output_path = synthetic_dir / "tabsyn.csv"
    shutil.copy2(generated_path, output_path)
    print(f"Sentetik veri kaydedildi: {output_path}")

    checkpoint_model = TABSYN_ROOT / "ckpt" / tabsyn_dataset_name / "model.pt"
    if checkpoint_model.exists():
        thesis_model_path = model_dir / f"{dataset_name}_tabsyn_model.pt"
        shutil.copy2(checkpoint_model, thesis_model_path)
        print(f"TabSyn model dosyası kaydedildi: {thesis_model_path}")
    else:
        print(f"UYARI: TabSyn model dosyası bulunamadı: {checkpoint_model}")

    copy_if_exists(csv_path, model_dir / f"{tabsyn_dataset_name}_input.csv")
    copy_if_exists(info_path, model_dir / f"{tabsyn_dataset_name}_info.json")
    copy_if_exists(generated_path, model_dir / f"{tabsyn_dataset_name}_generated_raw.csv")

    possible_artifacts = [
        (TABSYN_ROOT / "ckpt" / tabsyn_dataset_name, model_dir / "ckpt"),
        (TABSYN_ROOT / "checkpoints" / tabsyn_dataset_name, model_dir / "checkpoints"),
        (TABSYN_ROOT / "exp" / tabsyn_dataset_name, model_dir / "exp"),
        (TABSYN_ROOT / "outputs" / tabsyn_dataset_name, model_dir / "outputs"),
        (TABSYN_ROOT / "results" / tabsyn_dataset_name, model_dir / "results"),
        (TABSYN_ROOT / "synthetic" / tabsyn_dataset_name, model_dir / "synthetic_repo_output"),
    ]

    possible_files = [
        (TABSYN_ROOT / "ckpt" / f"{tabsyn_dataset_name}.pt", model_dir / f"{tabsyn_dataset_name}.pt"),
        (TABSYN_ROOT / "checkpoints" / f"{tabsyn_dataset_name}.pt", model_dir / f"{tabsyn_dataset_name}.pt"),
        (TABSYN_ROOT / "tabsyn" / f"{tabsyn_dataset_name}.pt", model_dir / f"{tabsyn_dataset_name}.pt"),
    ]

    copied_any = False

    for src, dst in possible_artifacts:
        if copy_if_exists(src, dst):
            print(f"Artifact klasörü kopyalandı: {src} -> {dst}")
            copied_any = True

    for src, dst in possible_files:
        if copy_if_exists(src, dst):
            print(f"Artifact dosyası kopyalandı: {src} -> {dst}")
            copied_any = True

    if not copied_any:
        print(
            "UYARI: TabSyn model/checkpoint artifact'i bulunamadı. Sadece sentetik veri ve input/metadata kopyalandı.")


def main() -> None:
    for dataset_name, cfg in DATASETS.items():
        try:
            train_tabsyn_for_dataset(dataset_name, cfg)
        except Exception as e:
            print(f"HATA | {dataset_name} | tabsyn: {e}")


if __name__ == "__main__":
    main()