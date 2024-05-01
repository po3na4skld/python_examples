# Data specific
DATA_PATH = "../data/train.csv"
INFERENCE_DATA_PATH = "../data/hidden_test.csv"
LABEL_NAME = "target"

# Model training
RANDOM_STATE = 42
VAL_SIZE = 0.1
N_ROUNDS = 1000
N_EARLY_STOPPING = 10

# Artifacts
MODEL_PATH = "artifacts/xgb_model.json"
INFERENCE_RESULTS = "artifacts/inference_results.csv"
