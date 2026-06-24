import pandas as pd
import requests
from pathlib import Path
from sklearn.datasets import fetch_openml

from config import DATASETS, BENCHMARK_ROWS, RANDOM_STATE, TRAIN_RATIO, TEST_RATIO
from sklearn.model_selection import train_test_split


ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]


def download_file(url: str, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if save_path.exists():
        print(f"Zaten var: {save_path}")
        return

    print(f"İndiriliyor: {url}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    with open(save_path, "wb") as file:
        file.write(response.content)

    print(f"Kaydedildi: {save_path}")


def ensure_adult_dataset() -> None:
    adult_path = DATASETS["adult"]["raw_path"] / "adult.data"
    adult_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    download_file(adult_url, adult_path)


def ensure_credit_dataset() -> None:
    credit_path = DATASETS["credit"]["raw_path"] / "credit.csv"
    credit_path.parent.mkdir(parents=True, exist_ok=True)

    if credit_path.exists():
        print(f"Zaten var: {credit_path}")
        return

    print("OpenML üzerinden credit fraud veri seti indiriliyor...")

    dataset = fetch_openml(data_id=42175, as_frame=True)
    df = dataset.frame.copy()

    # Hedef sütunu standartlaştır
    if "Class" not in df.columns:
        possible_target = dataset.target.name if hasattr(dataset.target, "name") else None
        if possible_target and possible_target in df.columns:
            df = df.rename(columns={possible_target: "Class"})
        elif df.columns[-1] not in ["Class"]:
            df = df.rename(columns={df.columns[-1]: "Class"})

    df.to_csv(credit_path, index=False)
    print(f"Kaydedildi: {credit_path}")


def ensure_datasets() -> None:
    ensure_adult_dataset()
    ensure_credit_dataset()


def load_adult(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        header=None,
        names=ADULT_COLUMNS,
        skipinitialspace=True,
    )
    return df


def load_credit(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "Class" not in df.columns:
        raise ValueError(
            f"Credit veri setinde 'Class' sütunu bulunamadı. "
            f"İlk sütunlar: {list(df.columns)[:10]}"
        )

    return df


def preprocess_common(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    object_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in object_cols:
        df[col] = df[col].astype(str).str.strip()

    df.replace("?", pd.NA, inplace=True)
    df.replace("", pd.NA, inplace=True)
    df.dropna(inplace=True)

    return df.reset_index(drop=True)


def preprocess_adult(df: pd.DataFrame) -> pd.DataFrame:
    df = preprocess_common(df)

    if "income" not in df.columns:
        raise ValueError("Adult veri setinde 'income' sütunu bulunamadı.")

    df["income"] = df["income"].astype(str).str.strip().str.rstrip(".")
    return df.reset_index(drop=True)


def preprocess_credit(df: pd.DataFrame) -> pd.DataFrame:
    df = preprocess_common(df)

    if "Class" not in df.columns:
        raise ValueError("Credit veri setinde 'Class' sütunu bulunamadı.")

    # Tezdeki 30 sütun düzenine hizalama:
    # 29 sürekli = V1..V28 + Amount
    # 1 ayrık = Class
    # Time dışarıda bırakılır
    if "Time" in df.columns:
        df = df.drop(columns=["Time"])

    return df.reset_index(drop=True)


def create_benchmark(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    if len(df) < BENCHMARK_ROWS:
        raise ValueError(
            f"{dataset_name} veri seti {BENCHMARK_ROWS} satırdan küçük: {len(df)}"
        )

    benchmark_df = df.sample(
        n=BENCHMARK_ROWS,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    return benchmark_df

def create_train_test_split(
    df: pd.DataFrame,
    dataset_name: str,
    dataset_config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    target_col = dataset_config["target_column"]

    # target kontrolü
    if target_col not in df.columns:
        raise ValueError(f"{dataset_name} veri setinde target bulunamadı: {target_col}")

    # oran kontrolü
    if abs((TRAIN_RATIO + TEST_RATIO) - 1.0) > 1e-9:
        raise ValueError(
            f"TRAIN_RATIO + TEST_RATIO = 1.0 olmalı. "
            f"Mevcut: {TRAIN_RATIO} + {TEST_RATIO} = {TRAIN_RATIO + TEST_RATIO}"
        )

    # stratify (classification için)
    stratify_col = df[target_col]

    # split
    train_df, test_df = train_test_split(
        df,
        train_size=TRAIN_RATIO,
        test_size=TEST_RATIO,
        random_state=RANDOM_STATE,
        stratify=stratify_col,
    )

    # index reset
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # sanity check
    total = len(df)
    if len(train_df) + len(test_df) != total:
        raise RuntimeError("Train + Test toplamı orijinal veri boyutuna eşit değil!")

    return train_df, test_df

def process_dataset(dataset_name: str, dataset_config: dict) -> None:
    print(f"\n=== {dataset_name.upper()} işleniyor ===")

    if dataset_name == "adult":
        csv_path = dataset_config["raw_path"] / "adult.data"
        if not csv_path.exists():
            raise FileNotFoundError(f"Adult veri dosyası bulunamadı: {csv_path}")

        print(f"Dosya: {csv_path}")
        df = load_adult(csv_path)
        print(f"Ham veri: {df.shape}")
        df = preprocess_adult(df)

    elif dataset_name == "credit":
        csv_path = dataset_config["raw_path"] / "credit.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Credit veri dosyası bulunamadı: {csv_path}")

        print(f"Dosya: {csv_path}")
        df = load_credit(csv_path)
        print(f"Ham veri: {df.shape}")
        df = preprocess_credit(df)

    else:
        raise ValueError(f"Desteklenmeyen veri seti: {dataset_name}")

    print(f"Temiz veri: {df.shape}")

    benchmark_df = create_benchmark(df, dataset_name)
    print(f"20k veri: {benchmark_df.shape}")

    save_path = dataset_config["benchmark_path"]
    save_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_df.to_csv(save_path, index=False)

    print(f"Kaydedildi: {save_path}")

    train_df, test_df = create_train_test_split(
        benchmark_df,
        dataset_name,
        dataset_config,
    )

    train_path = dataset_config["train_path"]
    test_path = dataset_config["test_path"]

    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train kaydedildi: {train_path} | boyut={train_df.shape}")
    print(f"Test kaydedildi: {test_path} | boyut={test_df.shape}")


def main() -> None:
    print("=== Veri setleri kontrol ediliyor / indiriliyor ===")
    ensure_datasets()

    for dataset_name, dataset_config in DATASETS.items():
        try:
            process_dataset(dataset_name, dataset_config)
        except Exception as error:
            print(f"{dataset_name} HATA: {error}")


if __name__ == "__main__":
    main()