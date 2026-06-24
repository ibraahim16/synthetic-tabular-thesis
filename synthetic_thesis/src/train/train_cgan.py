import sys
from pathlib import Path
import json
import pickle

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

from config import DATASETS, EPOCHS, BATCH_SIZE


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Generator(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def get_column_groups(df: pd.DataFrame):
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numerical_cols = df.select_dtypes(include=["number"]).columns.tolist()
    return categorical_cols, numerical_cols


def fit_preprocessors(X: pd.DataFrame):
    categorical_cols, numerical_cols = get_column_groups(X)

    ohe = None
    if categorical_cols:
        ohe = OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore",
        )
        X_cat = ohe.fit_transform(X[categorical_cols])
    else:
        X_cat = np.empty((len(X), 0), dtype=np.float32)

    scaler = None
    if numerical_cols:
        scaler = MinMaxScaler()
        X_num = scaler.fit_transform(X[numerical_cols])
    else:
        X_num = np.empty((len(X), 0), dtype=np.float32)

    X_enc = np.concatenate([X_num, X_cat], axis=1).astype(np.float32)

    return X_enc, categorical_cols, numerical_cols, ohe, scaler


def transform_features(
    X: pd.DataFrame,
    categorical_cols,
    numerical_cols,
    ohe,
    scaler,
):
    if numerical_cols:
        X_num = scaler.transform(X[numerical_cols]).astype(np.float32)
    else:
        X_num = np.empty((len(X), 0), dtype=np.float32)

    if categorical_cols:
        X_cat = ohe.transform(X[categorical_cols]).astype(np.float32)
    else:
        X_cat = np.empty((len(X), 0), dtype=np.float32)

    X_enc = np.concatenate([X_num, X_cat], axis=1).astype(np.float32)
    return X_enc


def decode_features(
    X_fake: np.ndarray,
    categorical_cols,
    numerical_cols,
    ohe,
    scaler,
) -> pd.DataFrame:
    X_fake = np.asarray(X_fake)

    num_dim = len(numerical_cols)

    if num_dim > 0:
        fake_num = X_fake[:, :num_dim]
        fake_num = np.clip(fake_num, 0.0, 1.0)
        fake_num = scaler.inverse_transform(fake_num)
        df_num = pd.DataFrame(fake_num, columns=numerical_cols)
    else:
        df_num = pd.DataFrame(index=range(len(X_fake)))

    if categorical_cols:
        fake_cat = X_fake[:, num_dim:]
        cat_dims = [len(cats) for cats in ohe.categories_]

        reconstructed_blocks = []
        start = 0
        for dim in cat_dims:
            block = fake_cat[:, start:start + dim]
            idx = np.argmax(block, axis=1)
            one_hot = np.zeros_like(block)
            one_hot[np.arange(len(block)), idx] = 1
            reconstructed_blocks.append(one_hot)
            start += dim

        reconstructed_cat = np.concatenate(reconstructed_blocks, axis=1)
        decoded_cat = ohe.inverse_transform(reconstructed_cat)
        df_cat = pd.DataFrame(decoded_cat, columns=categorical_cols)
    else:
        df_cat = pd.DataFrame(index=range(len(X_fake)))

    df = pd.concat([df_num.reset_index(drop=True), df_cat.reset_index(drop=True)], axis=1)
    return df


def restore_column_order(df_decoded: pd.DataFrame, original_feature_columns):
    missing_cols = [col for col in original_feature_columns if col not in df_decoded.columns]
    for col in missing_cols:
        df_decoded[col] = np.nan

    df_decoded = df_decoded[original_feature_columns]
    return df_decoded


def save_training_artifacts(model_dir: Path, artifacts: dict):
    model_dir.mkdir(parents=True, exist_ok=True)

    with open(model_dir / "artifacts.pkl", "wb") as f:
        pickle.dump(artifacts, f)

    with open(model_dir / "artifacts_info.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "categorical_cols": artifacts["categorical_cols"],
                "numerical_cols": artifacts["numerical_cols"],
                "target_mapping": artifacts["target_mapping"],
                "feature_dim": artifacts["feature_dim"],
                "noise_dim": artifacts["noise_dim"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def round_like_reference(synthetic_df: pd.DataFrame, reference_df: pd.DataFrame):
    result = synthetic_df.copy()

    for col in reference_df.columns:
        if col not in result.columns:
            continue

        if pd.api.types.is_integer_dtype(reference_df[col]):
            result[col] = np.round(result[col]).astype("Int64")
        elif pd.api.types.is_float_dtype(reference_df[col]):
            result[col] = result[col].astype(float)

    return result


def train_cgan_for_dataset(name, cfg):
    print(f"\n=== CGAN | {name.upper()} ===")

    benchmark_path = cfg["benchmark_path"]
    train_path = cfg["train_path"]
    synthetic_dir = cfg["synthetic_dir"]
    model_dir = cfg["models_dir"] / "cgan"
    target = cfg["target_column"]

    synthetic_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    benchmark_df = pd.read_csv(benchmark_path)
    train_df = pd.read_csv(train_path)

    if target not in train_df.columns:
        raise ValueError(f"Hedef sütun bulunamadı: {target}")

    print(f"Benchmark veri boyutu: {benchmark_df.shape}")
    print(f"Train veri boyutu: {train_df.shape}")

    y_raw = train_df[target].copy()
    X = train_df.drop(columns=[target]).copy()
    original_feature_columns = X.columns.tolist()

    X_enc, categorical_cols, numerical_cols, ohe, scaler = fit_preprocessors(X)

    y_cat = y_raw.astype("category")
    target_classes = y_cat.cat.categories.tolist()
    y_codes = y_cat.cat.codes.to_numpy().reshape(-1, 1).astype(np.float32)

    target_mapping = {int(i): str(cls) for i, cls in enumerate(target_classes)}

    data = np.concatenate([X_enc, y_codes], axis=1).astype(np.float32)
    tensor_data = torch.tensor(data, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(tensor_data),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    feature_dim = X_enc.shape[1]
    noise_dim = 32

    G = Generator(noise_dim + 1, feature_dim).to(device)
    D = Discriminator(feature_dim + 1).to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=0.0002)
    opt_D = torch.optim.Adam(D.parameters(), lr=0.0002)

    criterion = nn.BCELoss()

    for epoch in range(EPOCHS):
        for (real_batch,) in loader:
            real_batch = real_batch.to(device)

            real_X = real_batch[:, :-1]
            real_y = real_batch[:, -1:]

            noise = torch.randn(len(real_batch), noise_dim, device=device)

            # D eğitimi
            fake_gen_input = torch.cat([noise, real_y], dim=1)
            fake_X = G(fake_gen_input)

            real_input = torch.cat([real_X, real_y], dim=1)
            fake_input = torch.cat([fake_X.detach(), real_y], dim=1)

            D_real = D(real_input)
            D_fake = D(fake_input)

            loss_D_real = criterion(D_real, torch.ones_like(D_real))
            loss_D_fake = criterion(D_fake, torch.zeros_like(D_fake))
            loss_D = loss_D_real + loss_D_fake

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # G eğitimi
            noise = torch.randn(len(real_batch), noise_dim, device=device)
            fake_gen_input = torch.cat([noise, real_y], dim=1)
            fake_X = G(fake_gen_input)

            fake_input = torch.cat([fake_X, real_y], dim=1)
            D_fake = D(fake_input)

            loss_G = criterion(D_fake, torch.ones_like(D_fake))

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

        if epoch % 50 == 0 or epoch == EPOCHS - 1:
            print(
                f"Epoch {epoch:03d} | "
                f"D Loss: {loss_D.item():.4f} | "
                f"G Loss: {loss_G.item():.4f}"
            )

    print("Eğitim bitti")

    torch.save(G.state_dict(), model_dir / f"{name}_generator.pt")
    torch.save(D.state_dict(), model_dir / f"{name}_discriminator.pt")

    save_training_artifacts(
        model_dir,
        {
            "categorical_cols": categorical_cols,
            "numerical_cols": numerical_cols,
            "target_mapping": target_mapping,
            "feature_dim": feature_dim,
            "noise_dim": noise_dim,
            "original_feature_columns": original_feature_columns,
            "target_column": target,
            "target_classes": target_classes,
            "ohe": ohe,
            "scaler": scaler,
        },
    )

    # Üretim
    G.eval()
    with torch.no_grad():
        sample_count = len(benchmark_df)

        class_probs = (
            y_raw.astype("category")
            .cat.codes
            .value_counts(normalize=True)
            .sort_index()
            .values
        )

        sampled_labels = np.random.choice(
            np.arange(len(target_classes)),
            size=sample_count,
            replace=True,
            p=class_probs,
        ).astype(np.float32).reshape(-1, 1)

        noise = torch.randn(sample_count, noise_dim, device=device)
        label_tensor = torch.tensor(sampled_labels, dtype=torch.float32, device=device)

        gen_input = torch.cat([noise, label_tensor], dim=1)
        fake_X = G(gen_input).cpu().numpy()

    decoded_X = decode_features(
        X_fake=fake_X,
        categorical_cols=categorical_cols,
        numerical_cols=numerical_cols,
        ohe=ohe,
        scaler=scaler,
    )

    decoded_X = restore_column_order(decoded_X, original_feature_columns)
    decoded_X = round_like_reference(decoded_X, X)

    sampled_label_series = pd.Series(
        sampled_labels.flatten().astype(int)
    ).map(target_mapping)

    synthetic_df = decoded_X.copy()
    synthetic_df[target] = sampled_label_series

    output_columns = original_feature_columns + [target]
    synthetic_df = synthetic_df[output_columns]

    synthetic_path = synthetic_dir / "cgan.csv"
    synthetic_df.to_csv(synthetic_path, index=False)

    print(f"Sentetik veri kaydedildi: {synthetic_path}")
    print(f"Sentetik veri boyutu: {synthetic_df.shape}")


def main():
    for name, cfg in DATASETS.items():
        train_cgan_for_dataset(name, cfg)


if __name__ == "__main__":
    main()