"""Configuration constants for anomaly detection pipeline.

REAL DATA LABELING APPROACH
===========================
This pipeline now uses REAL student outcome data instead of synthetic anomalies.

Labeling Definition:
- NORMAL outcomes: Pass, Distinction (successful course completion)
- ABNORMAL outcomes: Fail, Withdrawn (course failure or dropout)

Training Strategy:
- Autoencoder: Trained on NORMAL students only (unsupervised)
- Isolation Forest: Multiple strategies available (see ISOLATION_FOREST_STRATEGY)

See LABELING_STRATEGY.md for detailed documentation.
"""

SEED = 41  # Random seed for reproducibility

# ==================== REAL DATA LABELING ====================
# Define which outcomes are considered "normal" vs "abnormal"
NORMAL_OUTCOME_LABELS = {'Pass', 'Distinction'}      # Successful completion
ABNORMAL_OUTCOME_LABELS = {'Fail', 'Withdrawn'}      # Failed or dropped out

# Data splitting: 75% train (normal students), 25% test (normal + abnormal)
NORMAL_TRAIN_FRACTION = 0.75  # Of normal students only, not all students
NORMAL_TEST_FRACTION = 0.25   # Of normal students only

# ==================== ISOLATION FOREST STRATEGIES ====================
# Strategy A: "normal_only" - Train on normal students only (conservative)
# Strategy B: "all_data" - Train on all students, use true contamination rate
# Strategy C: "informed" - Train on all students, add buffer to contamination
ISOLATION_FOREST_STRATEGY = "informed"  # "normal_only", "all_data", or "informed"
CONTAMINATION_BUFFER = 0.05  # Add 5% buffer to true contamination for robustness

# ==================== AUTOENCODER PARAMETERS ====================
BOTTLENECK_DIM = 4  # Compression dimension
EPOCHS = 80
BATCH_SIZE = 16

# ==================== BLOCKCHAIN LOGGING ====================
LEDGER_PATH = "blockchain_ledger.jsonl"  # Blockchain storage

# ==================== THRESHOLD OPTIMIZATION ====================
# Set OPTIMIZE_THRESHOLD=True to automatically find best threshold 
# using precision-recall tradeoff on validation data
OPTIMIZE_THRESHOLD = True
THRESHOLD_METRIC = "f1"  # "f1" (best balance), "precision" (fewer false positives), "recall" (fewer false negatives)