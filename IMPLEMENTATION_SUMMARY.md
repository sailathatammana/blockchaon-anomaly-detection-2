# Real Data Implementation Summary

## Overview

The project has been successfully updated to use **real student outcome data** instead of synthetic anomalies. This document summarizes all changes and provides guidance on running the updated pipeline.

---

## Key Changes Made

### 1. **Configuration Updates** (`scripts/config.py`)

**Removed:**
- `TRAIN_SIZE` / `TEST_SIZE` (replaced with outcome-based splitting)
- `INJECTION_FRAC` (no synthetic anomalies)
- `NOISE_SCALE` (no synthetic noise)

**Added:**
- `NORMAL_OUTCOME_LABELS` = `{'Pass', 'Distinction'}` — Successful course completion
- `ABNORMAL_OUTCOME_LABELS` = `{'Fail', 'Withdrawn'}` — Course failure or dropout
- `NORMAL_TRAIN_FRACTION` = 0.75 — Fraction of normal students for training
- `ISOLATION_FOREST_STRATEGY` — Strategy for isolation forest ("normal_only", "all_data", "informed")
- `CONTAMINATION_BUFFER` — Buffer for robust contamination estimation

---

### 2. **Preprocessing Enhancements** (`scripts/preprocessing.py`)

**New Methods Added:**

#### `create_anomaly_labels(profiles, normal_labels, abnormal_labels)`
- Creates binary labels from actual `final_result` field
- Returns labels (+1 normal, -1 abnormal) and statistics
- **Why:** Grounds anomaly detection in real student outcomes

#### `split_normal_abnormal(X, y, normal_train_fraction, random_state)`
- Separates normal from abnormal students BEFORE splitting
- Train: 75% of normal students only (unsupervised training)
- Test: 25% normal students + 100% abnormal students
- **Why:** Ensures autoencoder learns only "normal" patterns

**Key Principle:** Data splitting happens AFTER label creation, not before.

---

### 3. **Pipeline Updates** (`scripts/pipeline.py`)

**Major Changes:**

#### New Return Values
```python
X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler, label_info = \
    load_and_preprocess_data()
```

- **y_train**: Training labels (all +1, normal students only)
- **y_test**: Test labels (mix of +1 and -1)
- **label_info**: Dictionary with label statistics (proportions, contamination rate)

#### Autoencoder Training
- **Before:** Trained on 75% of all students (mixed)
- **After:** Trained on 75% of NORMAL students only
- **Benefit:** Truly unsupervised anomaly detection

#### No Synthetic Anomalies
- **Removed:** Gaussian noise injection step
- **Why:** Using real abnormal students (failures, withdrawals) as ground truth

#### Evaluation on Real Labels
- Compares predictions to actual `final_result` outcomes
- Provides realistic performance metrics
- Saved results now include label composition info

---

### 4. **Isolation Forest Enhancement** (`scripts/isolation_forest_baseline.py`)

**New Function:** `train_isolation_forest_real_data()`

Supports three strategies:

| Strategy | Training Data | Contamination | Use Case |
|----------|---|---|---|
| **normal_only** | Normal students only | 0.0 | Conservative, fewer false positives |
| **all_data** | All students | True contamination rate | Leverages outcome knowledge |
| **informed** | All students | True rate + buffer | **Recommended** — robust to uncertainty |

**Example:**
```python
model_if, strategy_info = train_isolation_forest_real_data(
    X_train,
    y_train=y_train,
    strategy="informed",
    contamination_buffer=0.05
)
```

---

### 5. **Baseline Comparison Updates** (`scripts/compare_baselines.py`)

**Key Changes:**
- No longer injects synthetic anomalies
- Uses real labels from pipeline
- Tests both autoencoder and isolation forest on same real data
- Displays strategy information for isolation forest
- Results include training composition and labeling approach

---

## Data Labeling Strategy (CRITICAL)

### Label Definition
```
final_result in data
│
├─ Pass          → Label: +1 (NORMAL)    ✓ Successful completion
├─ Distinction   → Label: +1 (NORMAL)    ✓ High performance
├─ Fail          → Label: -1 (ABNORMAL)  ✗ Course failure
├─ Withdrawn     → Label: -1 (ABNORMAL)  ✗ Dropout/disengagement
└─ Other         → Label: 0 (Unknown)    ? Excluded from analysis
```

### Data Split Process
```
1. Load all students
2. Create labels from final_result
3. Separate: normal_students vs abnormal_students
4. Split normal students: 75% train, 25% test
5. Combine: test_normal + all_abnormal → test_set
6. Fit scaler on train_normal ONLY
7. Apply scaler to test_set
```

**Important:** Scaler is fitted on training data only to prevent data leakage!

---

## Running the Updated Pipeline

### Option 1: Run Unified Pipeline (Recommended)
```bash
python scripts/pipeline.py
```

**Output:** Trained autoencoder on real data, metrics, blockchain logs

### Option 2: Run Baseline Comparison
```bash
python scripts/compare_baselines.py
```

**Output:** Compares autoencoder vs isolation forest on same real data

---

## Expected Results

### Performance Metrics Comparison

| Metric | Old (Synthetic) | New (Real Data) | Interpretation |
|--------|---|---|---|
| Accuracy | ~99% (suspiciously high) | ~80-90% | More realistic |
| Precision | ~95% | ~80-90% | Better reflects real world |
| Recall | ~98% | ~75-85% | Real challenges visible |
| F1 Score | ~0.96 | ~0.80-0.87 | Balanced, realistic |

**Key Insight:** Real data shows lower (but more believable) performance than synthetic data.

### Output Files

1. **label_info.json** — Label distribution statistics
2. **student_profiles_with_anomaly_labels.csv** — All profiles with labels
3. **pipeline_outputs.json** — Metrics, thresholds, configuration
4. **model_comparison.json** — Autoencoder vs Isolation Forest results
5. **predictions_comparison.csv** — Detailed predictions for each sample
6. **roc_data.json** — ROC curve data for visualization

---

## Key Implementation Details

### Autoencoder Training
```
Input: Only normal students (y_train = +1)
Process:
  1. Learn to reconstruct normal behavior
  2. Reconstruction error on normal ≈ low (~0.05-0.15)
  3. Reconstruction error on abnormal ≈ high (~0.2-0.5)
Output: Anomaly scores based on reconstruction error
```

### Isolation Forest Strategies
```
Strategy A: normal_only
  - Train on normal students only
  - Conservative: fewer false positives
  - Similar to autoencoder approach

Strategy B: all_data
  - Train on all students
  - Use true contamination rate
  - Leverages outcome knowledge

Strategy C: informed (RECOMMENDED)
  - Train on all students
  - Use contamination + buffer
  - Robust to proportion uncertainty
```

---

## Configuration Options

### In `scripts/config.py`:

```python
# What outcomes are normal vs abnormal
NORMAL_OUTCOME_LABELS = {'Pass', 'Distinction'}
ABNORMAL_OUTCOME_LABELS = {'Fail', 'Withdrawn'}

# Data splitting (of NORMAL students only)
NORMAL_TRAIN_FRACTION = 0.75

# Isolation Forest strategy
ISOLATION_FOREST_STRATEGY = "informed"  # or "normal_only", "all_data"
CONTAMINATION_BUFFER = 0.05

# Autoencoder hyperparameters (unchanged)
BOTTLENECK_DIM = 4
EPOCHS = 80
BATCH_SIZE = 16

# Threshold optimization (unchanged)
OPTIMIZE_THRESHOLD = True
THRESHOLD_METRIC = "f1"
```

---

## Validation Checklist

- [x] Anomaly labels created from `final_result` field
- [x] Normal vs abnormal students separated before splitting
- [x] Autoencoder trained on normal students only (75%)
- [x] Test set contains real abnormal samples (25% normal + 100% abnormal)
- [x] Scaler fitted on training data only (no data leakage)
- [x] Evaluation metrics use real labels, not synthetic
- [x] Isolation forest supports multiple strategies
- [x] Results saved with approach documentation
- [x] Backward compatibility with old evaluation metrics maintained

---

## Understanding the Results

### When You See Low Metrics
**Don't be alarmed!** This is expected with real data:
- Real anomalies (failures, withdrawals) are complex patterns
- Not as clearly separable as synthetic Gaussian noise
- Better reflects actual deployment performance
- Indicates the approach is realistic

### When to Tune

1. **Precision too low?** Students are being false-alarmed
   - Increase threshold (fewer detections)
   - Switch to "normal_only" isolation forest strategy

2. **Recall too low?** Missing at-risk students
   - Decrease threshold (more detections)
   - Use "informed" isolation forest strategy

3. **Want different anomaly definition?**
   - Edit `NORMAL_OUTCOME_LABELS` and `ABNORMAL_OUTCOME_LABELS` in config
   - Re-run pipeline

---

## Documentation

- **LABELING_STRATEGY.md** — Detailed labeling approach and justification
- **README.md** — Quick start and overview
- **scripts/pipeline.py** — Inline code documentation
- **scripts/preprocessing.py** — Feature engineering details
- **scripts/isolation_forest_baseline.py** — Strategy documentation

---

## Next Steps

1. Run `python scripts/pipeline.py` to train on real data
2. Review output files in `results/` directory
3. Examine `label_info.json` to understand data composition
4. Check `pipeline_outputs.json` for metrics
5. If needed, adjust config parameters and re-run
6. Use `compare_baselines.py` to benchmark different approaches

---

## Technical Notes

### Data Leakage Prevention
- Scaler is ALWAYS fit on training data only
- Test labels are NOT used during training
- Threshold optimization uses cross-validation or held-out validation data

### Reproducibility
- Random seed set to 41 throughout
- TensorFlow deterministic ops enabled
- Data splits are reproducible across runs

### Scalability
- Supports large student datasets
- Efficient feature engineering
- GPU acceleration available for autoencoder (optional)

