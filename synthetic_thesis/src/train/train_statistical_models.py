import sys
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.combine import SMOTETomek

from sklearn.preprocessing import OrdinalEncoder

from config import (
    DATASETS,
    RANDOM_STATE,
    MODELS,
)


def build_resampler(
    method_name: str,
    dataset_name: str,
):

    # Dataset bazlı oversampling oranı
    if dataset_name == "adult":

        # Adult zaten çok aşırı imbalance değil
        sampling_strategy = 0.35

    elif dataset_name == "credit":

        # Credit aşırı imbalance
        sampling_strategy = 0.05

    else:

        sampling_strategy = "auto"

    if method_name == "smote":

        return SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=RANDOM_STATE,
        )

    elif method_name == "borderline_smote":

        return BorderlineSMOTE(
            sampling_strategy=sampling_strategy,
            random_state=RANDOM_STATE,
            kind="borderline-1",
        )

    elif method_name == "smote_tomek":

        return SMOTETomek(
            sampling_strategy=sampling_strategy,
            random_state=RANDOM_STATE,
        )

    else:

        raise ValueError(
            f"Desteklenmeyen yöntem: {method_name}"
        )


def encode_categorical_features(
    X_train: pd.DataFrame,
):
    X_encoded = X_train.copy()

    categorical_cols = X_encoded.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()

    if len(categorical_cols) == 0:
        return X_encoded, None, []

    encoder = OrdinalEncoder()

    X_encoded[categorical_cols] = encoder.fit_transform(
        X_encoded[categorical_cols]
    )

    return X_encoded, encoder, categorical_cols


def print_distribution_stats(
    dataset_name: str,
    method_name: str,
    original_y: pd.Series,
    synthetic_y: pd.Series,
) -> None:

    print("\n" + "=" * 40)

    print(f"DATASET : {dataset_name.upper()}")
    print(f"METHOD  : {method_name.upper()}")

    print("=" * 40)

    print("\nORIGINAL DISTRIBUTION")

    original_counts = original_y.value_counts()

    for cls, count in original_counts.items():

        percentage = (
            count / len(original_y)
        ) * 100

        print(
            f"{cls}: "
            f"{count} "
            f"(%{percentage:.2f})"
        )

    print("\nSYNTHETIC DISTRIBUTION")

    synthetic_counts = synthetic_y.value_counts()

    for cls, count in synthetic_counts.items():

        percentage = (
            count / len(synthetic_y)
        ) * 100

        print(
            f"{cls}: "
            f"{count} "
            f"(%{percentage:.2f})"
        )


def train_resampling_for_dataset(
    dataset_name: str,
    dataset_config: dict,
    method_name: str,
) -> None:

    print(
        f"\n=== "
        f"{method_name.upper()} | "
        f"{dataset_name.upper()} "
        f"==="
    )

    train_path = dataset_config["train_path"]

    benchmark_path = dataset_config[
        "benchmark_path"
    ]

    synthetic_dir = dataset_config[
        "synthetic_dir"
    ]

    synthetic_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_df = pd.read_csv(train_path)

    benchmark_df = pd.read_csv(
        benchmark_path
    )

    print(
        f"Train veri boyutu: "
        f"{train_df.shape}"
    )

    print(
        f"Benchmark veri boyutu: "
        f"{benchmark_df.shape}"
    )

    target_column = dataset_config[
        "target_column"
    ]

    X = train_df.drop(
        columns=[target_column]
    )

    y = train_df[target_column]

    # Adult categorical encoding
    X, encoder, categorical_cols = \
        encode_categorical_features(X)

    # Resampler oluştur
    resampler = build_resampler(
        method_name,
        dataset_name,
    )

    print(f"{method_name} uygulanıyor...")

    start_time = time.time()

    X_resampled, y_resampled = (
        resampler.fit_resample(X, y)
    )

    elapsed_time = (
        time.time() - start_time
    )

    synthetic_df = pd.concat(
        [
            pd.DataFrame(
                X_resampled,
                columns=X.columns,
            ),

            pd.Series(
                y_resampled,
                name=target_column,
            ),
        ],
        axis=1,
    )

    if encoder is not None:
        synthetic_df[categorical_cols] = (
            encoder.inverse_transform(
                synthetic_df[categorical_cols]
            )
        )

    # Benchmark size eşitle
    if len(synthetic_df) > len(
        benchmark_df
    ):

        synthetic_df = synthetic_df.sample(
            n=len(benchmark_df),
            random_state=RANDOM_STATE,
        ).reset_index(drop=True)

    elif len(synthetic_df) < len(
        benchmark_df
    ):

        synthetic_df = synthetic_df.sample(
            n=len(benchmark_df),
            replace=True,
            random_state=RANDOM_STATE,
        ).reset_index(drop=True)

    synthetic_path = (
        synthetic_dir /
        f"{method_name}.csv"
    )

    synthetic_df.to_csv(
        synthetic_path,
        index=False,
    )

    print(
        f"Sentetik veri kaydedildi: "
        f"{synthetic_path}"
    )

    print(
        f"Sentetik veri boyutu: "
        f"{synthetic_df.shape}"
    )

    print(
        f"Üretim süresi: "
        f"{elapsed_time:.2f} saniye"
    )

    print("\nSınıf dağılımı:")

    print(
        synthetic_df[
            target_column
        ].value_counts()
    )

    print_distribution_stats(
        dataset_name=dataset_name,
        method_name=method_name,
        original_y=y,
        synthetic_y=synthetic_df[
            target_column
        ],
    )


def main() -> None:

    resampling_methods = [

        model_name

        for model_name in MODELS

        if model_name in [
            "smote",
            "borderline_smote",
            "smote_tomek",
        ]
    ]

    for (
        dataset_name,
        dataset_config,
    ) in DATASETS.items():

        for method_name in (
            resampling_methods
        ):

            train_resampling_for_dataset(
                dataset_name,
                dataset_config,
                method_name,
            )


if __name__ == "__main__":

    main()