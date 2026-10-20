from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "assesiment" / "data" / "train-test.csv"
VALIDATION_PATH = BASE_DIR / "assesiment" / "data" / "validation.csv"
DECEMBER_INPUTS_PATH = BASE_DIR / "assesiment" / "data" / "december-chart-inputs.csv"
RESULTS_DIR = BASE_DIR / "results"

# Seeds and protocols
SEEDS = [0, 1, 2, 3, 4]
PROTOCOLS = ["protocol_a", "protocol_b"]
MODELS = ["ridge", "lightgbm", "mlp"]

# Flags and options
USE_QUOTE_SIGNAL = True
TARGET_MODE = "raw"  # Options: 'raw', 'rate_per_mile', 'log'

# Constants
EARTH_RADIUS_MILES = 3958.8

# Feature engineering parameters
TARGET_ENCODING_SMOOTHING = 10.0
TARGET_ENCODING_FOLDS = 5

# LightGBM fixed sensible defaults
LGBM_PARAMS = {
    "learning_rate": 0.05,
    "n_estimators": 2000,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "verbose": -1,
    "n_jobs": -1,
}
EARLY_STOPPING_ROUNDS = 50

# MLP fixed hyperparameters
MLP_NORM = None  # Options: None, "batchnorm", "layernorm"
MLP_LR = 1e-3
MLP_WEIGHT_DECAY = 1e-4
MLP_BATCH_SIZE = 256
MLP_MAX_EPOCHS = 200
MLP_PATIENCE = 15
MLP_LR_PATIENCE = 5
MLP_LR_FACTOR = 0.5
