"""Configuration constants for anomaly detection pipeline."""

SEED = 42  # Random seed for reproducibility

TRAIN_SIZE = 0.70
TEST_SIZE = 0.30

INJECTION_FRAC = 0.02  # Synthetic anomalies for evaluation
NOISE_SCALE = 3.0  # Noise scale in std deviations

BOTTLENECK_DIM = 4  # Compression dimension
EPOCHS = 80
BATCH_SIZE = 16

LEDGER_PATH = "blockchain_ledger.jsonl"  # Blockchain storage
