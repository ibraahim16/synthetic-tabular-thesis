from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"
DATA_SPLIT_DIR = PROJECT_ROOT / "data" / "split"
DATA_SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"

MODELS_DIR = PROJECT_ROOT / "models"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
STATS_DIR = OUTPUTS_DIR / "stats"
DM_DIR = OUTPUTS_DIR / "dm"
EM_DIR = OUTPUTS_DIR / "em"
PLOTS_DIR = OUTPUTS_DIR / "plots"
LOGS_DIR = OUTPUTS_DIR / "logs"

RANDOM_STATE = 42
BENCHMARK_ROWS = 20_000
EPOCHS = 500
BATCH_SIZE = 512
TRAIN_RATIO = 0.80
TEST_RATIO = 0.20

DATASETS = {
    "adult": {
        "raw_path": DATA_RAW_DIR / "adult",
        "benchmark_path": DATA_BENCHMARK_DIR / "adult_20k.csv",
        "train_path": DATA_SPLIT_DIR / "adult_train.csv",
        "test_path": DATA_SPLIT_DIR / "adult_test.csv",
        "synthetic_dir": DATA_SYNTHETIC_DIR / "adult",
        "models_dir": MODELS_DIR / "adult",
        "target_column": "income",          # classification
        "regression_target": "age",         # regression
    },
    "credit": {
        "raw_path": DATA_RAW_DIR / "credit",
        "benchmark_path": DATA_BENCHMARK_DIR / "credit_20k.csv",
        "train_path": DATA_SPLIT_DIR / "credit_train.csv",
        "test_path": DATA_SPLIT_DIR / "credit_test.csv",
        "synthetic_dir": DATA_SYNTHETIC_DIR / "credit",
        "models_dir": MODELS_DIR / "credit",
         "target_column": "Class",           # classification
        "regression_target": "Amount",      # regression
    },
}

MODELS = ["smote", "borderline_smote", "smote_tomek", "cgan", "ctgan", "copulagan", "ctabgan_plus", "tabsyn", "forest_diffusion"]
