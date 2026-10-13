"""Configuration constants for anomaly detection pipeline."""

SEED = 41  # Random seed for reproducibility

TRAIN_SIZE = 0.60
VAL_SIZE = 0.15
TEST_SIZE = 0.25

INJECTION_FRAC = 0.12  # Synthetic anomalies for evaluation
NOISE_SCALE = 1.8  # Noise scale in std deviations

BOTTLENECK_DIM = 4  # Compression dimension
EPOCHS = 80
BATCH_SIZE = 16

LEDGER_PATH = "blockchain_ledger.jsonl"  # Blockchain storage

# Threshold Optimization
OPTIMIZE_THRESHOLD = True
THRESHOLD_METRIC = "f1"  # "f1" (best balance), "precision" (fewer false positives), "recall" (fewer false negatives)