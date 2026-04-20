# Real Data Labeling Strategy for Anomaly Detection

## Overview

This document describes how **real student outcome data** replaces **synthetic anomalies** in the anomaly detection pipeline. The approach carefully handles different training strategies for autoencoders vs. isolation forests.

---

## 1. Data Labeling: Defining "Normal" vs "Abnormal"

### Definition
We use **student course outcomes** (`final_result` field) as ground truth labels:

| final_result | Label | Interpretation | Count |
|---|---|---|---|
| **Pass** | NORMAL | Student completed course successfully | ~ |
| **Distinction** | NORMAL | Student completed with high performance | ~ |
| **Fail** | ABNORMAL | Student failed the course | ~ |
| **Withdrawn** | ABNORMAL | Student dropped out / disengaged | ~ |

### Justification
- **Grounded in reality**: Based on actual student outcomes, not synthetic noise
- **Pedagogically meaningful**: Pass/Distinction indicates normal learning progression; Fail/Withdrawn indicates abnormal engagement or performance
- **Balanced anomalies**: Real abnormal cases (failures, withdrawals) are typically ~20-40% of data
- **Available**: This field exists in the `studentInfo.csv` dataset

### Label Extraction
```python
def create_anomaly_labels(profiles):
    """
    Create binary labels based on final_result.
    
    Returns:
        y: array where 1 = normal, -1 = abnormal
    """
    normal_outcomes = {'Pass', 'Distinction'}
    abnormal_outcomes = {'Fail', 'Withdrawn'}
    
    y = profiles['final_result'].map(
        lambda x: 1 if x in normal_outcomes else -1 if x in abnormal_outcomes else None
    )
    return y
```

---

## 2. Data Splitting Strategy

### Phase 1: Separate by Outcome (NOT random split)

```
All Students (N=10000)
│
├─ NORMAL Students (Pass/Distinction) [~7000, 70%]
│  ├─ Train Autoencoder [5250, 75% of normal]
│  └─ Test Set: Normal Subset [1750, 25% of normal]
│
└─ ABNORMAL Students (Fail/Withdrawn) [~3000, 30%]
   └─ Test Set: Full Abnormal Set [3000, 100% of abnormal]

Test Set = Normal Subset + All Abnormal = ~4750 total
```

### Phase 2: Normalize Features (Critical!)
- Fit scaler on training data (normal students only)
- Apply same scaler to test set (both normal and abnormal)
- **Never fit scaler on test/abnormal data** → prevents data leakage

---

## 3. Model Training Approaches

### 3.1 Autoencoder Training (Unsupervised on Normal Data Only)

**Why train on normal data only?**
- Autoencoders learn to reconstruct "normal" patterns
- When presented with abnormal samples, they fail to reconstruct well
- High reconstruction error = anomaly detected
- This is the core principle of autoencoder-based anomaly detection

**Algorithm**:
```
1. Load all student profiles
2. Label students as normal/abnormal
3. Extract ONLY normal student features
4. Randomly split: 75% train, 25% validation
5. Normalize using training data statistics ONLY
6. Train autoencoder on training normal students
7. Evaluate on: 
   a) Validation normal students (should have LOW reconstruction error)
   b) Test abnormal students (should have HIGH reconstruction error)
```

**Expected Behavior**:
- Normal students: reconstruction error ≈ 0.05-0.15 (low)
- Abnormal students: reconstruction error ≈ 0.2-0.5 (high)
- Clear separation → good detection

### 3.2 Isolation Forest: Multiple Strategies

**Strategy A: Train on Normal Only (Conservative)**
- Same approach as autoencoder
- Contamination parameter = 0 (no anomalies expected in training)
- More conservative, fewer false positives

**Strategy B: Train on All Data (Informed)**
- Use all students (normal + abnormal)
- Contamination parameter = true_abnormal_fraction (e.g., 0.30)
- Leverages knowledge of anomaly proportion
- Better false negative rate

**Strategy C: Train on Normal + Add Anomaly Hints (Recommended)**
- Train on all students
- Set contamination parameter = slightly higher than true value (e.g., 0.35 if true=0.30)
- Robust to true proportion uncertainty

---

## 4. Evaluation Protocol

### Ground Truth
```python
y_true = np.array([1, -1, -1, 1, 1, -1, ...])  # From final_result labels
```

### Predictions
```python
# Autoencoder
y_pred_ae = predict_anomalies(model, X_test, threshold)  # 1 or -1

# Isolation Forest (Strategy B)
y_pred_if = predict_anomalies_if(model, X_test)  # 1 or -1
```

### Metrics (Consistent with Synthetic Approach)
- **Accuracy**: (TP + TN) / Total
- **Precision**: TP / (TP + FP) — How many detections are correct?
- **Recall**: TP / (TP + FN) — How many anomalies did we find?
- **F1-Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under ROC curve (threshold-independent)

### Important: Label Encoding
- Normal students: y_true = +1
- Abnormal students: y_true = -1
- Predicted normal: y_pred = +1
- Predicted anomaly: y_pred = -1

This matches the synthetic evaluation approach for comparison.

---

## 5. Key Differences from Synthetic Approach

| Aspect | Synthetic (Old) | Real Data (New) |
|---|---|---|
| **Anomaly Source** | Injected Gaussian noise | Actual course failures/withdrawals |
| **Train Data** | 75% of all students (mixed) | 75% of NORMAL students only |
| **Test Set** | Random 25% + corrupted | 25% normal + all abnormal |
| **Realistic?** | Low (noise artifacts) | High (real student behavior) |
| **Model Generalization** | Overfitting risk | Better generalization |
| **Autoencoder** | Sees anomalies during training | Truly unsupervised |
| **Isolation Forest** | Flexibility in strategies | Can use outcome knowledge |

---

## 6. Implementation Files Modified

1. **config.py**
   - Remove: `INJECTION_FRAC`, `NOISE_SCALE`
   - Add: `NORMAL_OUTCOME_LABELS`, `ABNORMAL_OUTCOME_LABELS`, `ISOLATION_FOREST_STRATEGY`

2. **preprocessing.py**
   - Add: `create_anomaly_labels()` function
   - Add: `split_normal_abnormal()` function

3. **pipeline.py**
   - Replace: Synthetic anomaly injection with real labels
   - Update: Data splitting logic (normal/abnormal separation)
   - Update: Autoencoder training (normal only)
   - Update: Evaluation (use real labels)

4. **isolation_forest_baseline.py**
   - Add: Strategy parameter for different training approaches
   - Add: Contamination control based on strategy

---

## 7. Expected Outcomes

### Metrics Comparison (Estimated)
```
Metric               Synthetic (Old)    Real Data (New)
─────────────────────────────────────────────────────
Accuracy             ~99% (suspiciously high)    ~85-92%
Precision            ~95%               ~80-90%
Recall               ~98%               ~75-85%
ROC AUC              ~0.99              ~0.80-0.90
─────────────────────────────────────────────────────
```

The new approach will show:
- **Lower accuracy** (realistic, not artificially high)
- **More interpretable results** (based on real outcomes)
- **Better generalization** (to real unseen students)
- **Actionable insights** (can identify struggling students)

---

## 8. Validation Checklist

- [ ] Labeled students correctly (Pass/Distinction → normal, Fail/Withdrawn → abnormal)
- [ ] Separated train/test BEFORE normalization
- [ ] Fitted scaler on training normal data only
- [ ] Autoencoder sees only normal students during training
- [ ] Test set contains real abnormal samples
- [ ] Evaluation uses correct label encoding (+1/-1)
- [ ] Metrics computed on real labels, not injected anomalies
- [ ] Document anomaly proportions in test set
- [ ] Report results with disclaimer about real vs. synthetic

