# Ablation Study: Detailed Implementation Guide

## Overview

This document provides a comprehensive guide to understanding and using the ablation study for the blockchain anomaly detection system. An ablation study is a scientific approach to understanding model behavior by systematically testing the contribution of individual components.

## Why Ablation Studies Matter

### Research Value
- **Interpretability**: Understand which components drive performance
- **Reproducibility**: Document what matters in your model
- **Credibility**: Justify design decisions with evidence
- **Generalization**: Learn principles applicable to other projects

### Practical Benefits
- **Optimization**: Find sweet spots in the performance-complexity tradeoff
- **Debugging**: Identify unexpected dependencies
- **Collaboration**: Communicate model design clearly
- **Documentation**: Create a record of your exploration process

## Architecture Overview

### Data Pipeline

```
Raw Data (7 CSV files)
    ↓
[Load and merge] ← data_loader.py
    ↓
Merged DataFrame (~9,000 students × 28 features)
    ↓
[Preprocess] ← preprocessing.py (scaling, encoding, aggregation)
    ↓
Feature Matrix X (~9,000 × 55 features)
    ↓
[Create Labels] ← Outcome-based (Pass/Distinction = normal, Fail/Withdrawn = abnormal)
    ↓
Split: 75% training (normal only), 25% testing (normal + abnormal)
    ↓
[Train & Evaluate] ← ablation_study.py
    ↓
Results JSON
    ↓
[Analyze] ← analyze_ablation.py
    ↓
Charts, Heatmaps, Report
```

### Model Architecture

All tested models follow this symmetric autoencoder pattern:

```
Input Features (55)
    ↓
[Dense + ReLU + Dropout] ← Encoder Layer 1
    ↓
[Dense + ReLU + Dropout] ← Encoder Layer 2  (optional, depends on depth)
    ↓
[Dense + ReLU + Dropout] ← Encoder Layer 3  (optional, depends on depth)
    ↓
[Dense + ReLU] ← Bottleneck Layer (dimension varies: 2-32)
    ↓
[Dropout]
    ↓
[Dense + ReLU + Dropout] ← Decoder Layer 1
    ↓
[Dense + ReLU + Dropout] ← Decoder Layer 2  (mirrored)
    ↓
[Dense + ReLU + Dropout] ← Decoder Layer 3  (mirrored)
    ↓
[Dense + Linear] ← Output Layer (reconstructs 55 features)
```

## Components Being Tested

### 1. Bottleneck Dimension

**What it controls**: The degree of compression through the network

**Values tested**: 2, 4, 8, 16, 32

**How it works**:
- Bottleneck dimension is the width of the narrowest layer
- Smaller dimension = stronger compression = forces learning of core patterns
- Larger dimension = less compression = more capacity to memorize

**Expected behavior**:
```
Performance
    ↑
    │     ●
    │    ● ●
    │   ●   ●
    │        ●
    └─────────────→ Bottleneck Dimension
      2  4  8  16  32

Optimal usually around 4-8
(sweet spot between compression and capacity)
```

**Why test this?**
- Too small (2): Can't fit normal patterns, loses information
- Too large (32): Just memorizes training data, overfits
- Just right (4-8): Captures essential patterns, rejects anomalies

### 2. Network Depth

**What it controls**: Number of hidden layers (encoder depth only; decoder mirrors it)

**Values tested**: 1, 2, 3, 4 layers

**How it works**:
- Depth 1: Single layer compresses directly to bottleneck
- Depth 2: Two intermediate layers before bottleneck
- Depth 3: Three intermediate layers (baseline)
- Depth 4: Four layers (deeper, more parameters)

**Architecture examples**:

```
Depth 1:              Depth 3:              Depth 4:
Input (55)          Input (55)            Input (55)
  ↓                   ↓                      ↓
Dense(64)           Dense(128)            Dense(256)
  ↓                   ↓                      ↓
Bottleneck(4)       Dense(64)             Dense(128)
  ↓                   ↓                      ↓
Dense(64)           Dense(32)             Dense(64)
  ↓                   ↓                      ↓
Output(55)          Bottleneck(4)         Dense(32)
                      ↓                      ↓
                    Dense(32)             Bottleneck(4)
                      ↓                      ↓
                    Dense(64)             Dense(32)
                      ↓                      ↓
                    Dense(128)            Dense(64)
                      ↓                      ↓
                    Output(55)            Dense(128)
                                            ↓
                                          Dense(256)
                                            ↓
                                          Output(55)
```

**Expected behavior**:
- Depth 1: Minimal capacity, may underfit
- Depth 2-3: Sweet spot (performance vs parameters)
- Depth 4: More parameters, higher training cost, risk of overfitting

### 3. Dropout Rate

**What it controls**: Regularization strength (prevents overfitting)

**Values tested**: 0.0, 0.1, 0.2, 0.3

**How it works**:
- During training, randomly deactivate (drop) neurons with given probability
- Forces network to learn redundant representations
- Acts as ensemble of many models

**Effect on training**:
```
Loss Curve
    ↑
    │ No Dropout (0.0)     High Dropout (0.3)
    │  /‾‾‾‾‾‾→           /‾‾‾‾‾‾→
    │ /    ↑ Overfitting   /
    │ Training gap is      Training slower
    │ large (overfit)      but more robust
    └────────────────→ Epochs
```

**Expected behavior**:
- 0.0 (no dropout): Train loss low, test loss high (overfitting)
- 0.1-0.2 (moderate): Best generalization
- 0.3 (high): May undershoot capacity

### 4. Activation Function

**What it controls**: The non-linearity of the network

**Values tested**: relu, sigmoid, tanh

**Characteristics**:

| Function | Range | Properties | Use Case |
|----------|-------|-----------|----------|
| ReLU | [0, ∞) | Simple, fast, sparse | Standard (recommended) |
| Sigmoid | (0, 1) | Smooth, bounded | Output layers |
| Tanh | (-1, 1) | Centered, symmetric | Some architectures prefer |

**Behavior comparison**:
```
        ReLU              Sigmoid              Tanh
         │                  │ ╱                  │ ╱
         │ ╱                 ╱                   ╱
     ────┼───→         ──────┼──→          ──────┼──→
        0│                 0│╲                0│╲
         │╲                  │╲                │╲
```

**Why test this?**
- ReLU: Computationally efficient, modern standard
- Sigmoid: Can suffer from vanishing gradients
- Tanh: Centered output might help with convergence

### 5. Feature Subset

**What it controls**: Dimensionality (how many input features to use)

**Values tested**: 100%, 50%, 25% (selected by variance)

**How it works**:
- Compute variance of each feature across training data
- Keep top N% features with highest variance
- This removes uninformative features

**Expected behavior**:
```
F1 Score
    ↑
    │ ●
    │  ●
    │   ●
    └────────────→ Features Used
      25%  50%  100%

Usually:
- 100% features: May include noise
- 50-75% features: Often best (removing noise, keeping signal)
- 25% features: May lose important information
```

**Why test this?**
- Reduces computational cost (smaller models)
- May improve generalization (remove noise)
- Interpretation: Which features matter most?

## Running the Ablation Study

### Complete Workflow

#### Step 1: Prepare Environment

```bash
cd scripts

# Verify dependencies
python -c "import tensorflow; import sklearn; print('✓ Dependencies OK')"
```

#### Step 2: Run Ablation Study

```bash
python ablation_study.py
```

**What happens**:
1. Loads data from `data/` folder (7 CSV files)
2. Preprocesses features (scaling, encoding)
3. Creates labels from student outcomes
4. Trains baseline model (bottleneck=4, depth=3, dropout=0.1, relu)
5. Tests 5 component variations:
   - 5 bottleneck dimensions (2, 4, 8, 16, 32) = 5 models
   - 4 network depths (1, 2, 3, 4) = 4 models
   - 4 dropout rates (0.0, 0.1, 0.2, 0.3) = 4 models
   - 3 activation functions (relu, sigmoid, tanh) = 3 models
   - 3 feature subsets (100%, 50%, 25%) = 3 models

**Total models trained**: 1 baseline + 5 + 4 + 4 + 3 + 3 = **20 models**

**Time estimate**:
- Per model: 1-3 minutes (depending on data size and early stopping)
- Total: 20-60 minutes (sequential training)

**Output**:
```
results/ablation_study.json (raw data)
  ├─ baseline
  │   ├─ config
  │   ├─ f1, roc_auc, accuracy, ...
  │   └─ epochs_run, final_train_loss, ...
  ├─ variations
  │   ├─ bottleneck_dimension
  │   ├─ network_depth
  │   ├─ dropout_rate
  │   ├─ activation_function
  │   └─ feature_subset
  └─ timestamp, data_summary
```

#### Step 3: Analyze Results

```bash
python analyze_ablation.py
```

**What happens**:
1. Loads JSON results
2. Creates comparison visualizations:
   - 6-panel chart (bottleneck, depth, dropout, activation, features, baseline)
   - 2 heatmaps (metrics vs components)
3. Generates analysis report with:
   - Summary statistics
   - Per-component rankings
   - Best configuration
   - Improvement percentages
   - Recommendations

**Output files**:
```
results/
  ├─ ablation_study.json (created by step 2)
  ├─ ablation_comparison.png (6-panel chart)
  ├─ ablation_heatmap.png (performance heatmaps)
  └─ ablation_analysis_report.txt (detailed analysis)
```

## Interpreting Results

### Reading the Comparison Chart

The 6-panel chart shows:

1. **Top-left: Bottleneck Dimension**
   - X-axis: 2, 4, 8, 16, 32
   - Y-axis: Score (0.8-1.0)
   - Two lines: F1 (blue dots) and ROC-AUC (orange squares)
   - **Interpretation**: Which compression level achieves best balance?

2. **Top-middle: Network Depth**
   - X-axis: 1, 2, 3, 4
   - Y-axis: Score (0.8-1.0)
   - **Interpretation**: Is deeper network better?

3. **Top-right: Dropout Rate**
   - X-axis: 0.0, 0.1, 0.2, 0.3
   - Y-axis: Score (0.8-1.0)
   - **Interpretation**: How much regularization helps?

4. **Bottom-left: Activation Function**
   - Bar chart: relu vs sigmoid vs tanh
   - **Interpretation**: Which non-linearity works best?

5. **Bottom-middle: Feature Subset**
   - Bar chart: 25%, 50%, 100%
   - **Interpretation**: Can we reduce dimensionality?

6. **Bottom-right: Baseline Metrics**
   - Bar chart of all metrics for baseline
   - **Interpretation**: How strong is baseline overall?

### Reading the Heatmap

Heatmaps show how metrics vary with component values:

```
                        Bottleneck Dimension
                    2      4      8     16     32
             ┌──────┬──────┬──────┬──────┬──────┐
        F1   │0.94 │0.95 │0.95 │0.94 │0.93 │  ← Best at 4 and 8
             ├──────┼──────┼──────┼──────┼──────┤
     ROC-AUC │0.94 │0.95 │0.95 │0.94 │0.93 │  ← Consistent pattern
             ├──────┼──────┼──────┼──────┼──────┤
   Precision │0.95 │0.94 │0.94 │0.95 │0.96 │  ← Different trend!
             ├──────┼──────┼──────┼──────┼──────┤
      Recall │0.94 │0.96 │0.96 │0.93 │0.92 │  ← Best at 4 and 8
             └──────┴──────┴──────┴──────┴──────┘

Color coding: Green = high (0.95+), Yellow = medium (0.90-0.95), Red = low
```

### Analyzing the Report

The text report provides:

1. **Data Summary**
   - Total features, training samples, test samples
   - Number of abnormal samples

2. **Baseline Performance**
   - All metrics for the original configuration
   - Reference point for improvements

3. **Component Impact Analysis**
   - For each component (bottleneck, depth, etc.)
   - Lists all variants sorted by F1
   - Shows percentage improvement vs baseline
   - Highlights best variant

4. **Key Findings**
   - Best overall configuration
   - Component sensitivity ranking
   - Recommendations

## Statistical Interpretation

### Significant Improvements

How to interpret performance differences:

| Improvement | Significance | Action |
|-------------|--------------|--------|
| > +3% | **Very significant** | Definitely use this config |
| +1% to +3% | **Significant** | Consider using, worth evaluating |
| +0.1% to +1% | **Marginal** | Probably not worth it |
| -0.1% to +0.1% | **Noise** | Baseline is fine |
| < -0.1% | **Degradation** | Don't use |

### Metric-Specific Interpretation

**F1 Score** (our primary metric):
- Ranges 0-1, higher is better
- Balance of precision and recall
- Most important for anomaly detection

**ROC-AUC**:
- Ranges 0-1, higher is better
- Probability of correct ranking (anomaly > normal)
- Less sensitive to threshold choice

**Precision**:
- TP / (TP + FP)
- Higher = fewer false alarms
- Important if false positives are costly

**Recall**:
- TP / (TP + FN)
- Higher = fewer missed anomalies
- Important if false negatives are costly

**Specificity**:
- TN / (TN + FP)
- Ability to identify normal samples
- Important for avoiding false alarms

## Advanced: Customizing the Study

### Adding New Components

To test a new component (e.g., learning rate):

1. **Add to the study** (in `ablation_study.py`):
```python
# Around line 250, add new section:
print("\n[8/8] Testing learning rates [1e-3, 5e-4, 1e-4]...")

learning_rate_results = {}
for lr in [1e-3, 5e-4, 1e-4]:
    print(f"  Testing lr={lr}...", end=" ")
    # Modify build_autoencoder to accept learning_rate parameter
    results = train_variant(
        X_train, X_val, X_test, y_test,
        bottleneck_dim=4,
        depth=3,
        dropout_rate=0.1,
        activation="relu",
        learning_rate=lr,  # Add parameter
    )
    results["config"] = {...}
    learning_rate_results[f"lr_{lr}"] = results
    print(f"F1={results['f1']:.4f}")

all_results["variations"]["learning_rate"] = learning_rate_results
```

2. **Modify model builder** (in `autoencoder.py`):
```python
def build_autoencoder_variable_depth(..., learning_rate=5e-4):
    ...
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),  # Use parameter
        loss="mse"
    )
```

3. **Update analysis** (in `analyze_ablation.py`):
```python
# Add new subplot for learning rate
ax = axes[2, 0]
lr_df = df[df["category"] == "learning_rate"].copy()
lr_df = lr_df.sort_values("learning_rate")
# ... plot code
```

### Reducing Execution Time

If the study takes too long:

1. **Reduce values tested**:
```python
# Instead of 5 bottleneck values, test 3:
for bottleneck_dim in [2, 4, 16]:  # Reduced from [2, 4, 8, 16, 32]
```

2. **Reduce epochs**:
```python
# In train_variant() call:
results = train_variant(..., epochs=40)  # Reduced from 80
```

3. **Reduce data size** (for development):
```python
# In load_and_preprocess_data():
X_train = X_train[:1000]  # Use subset
X_val = X_val[:300]
X_test = X_test[:500]
```

### Extending the Study

Add new components to test:
- **Batch sizes**: [8, 16, 32, 64]
- **Learning rates**: [1e-4, 5e-4, 1e-3]
- **L1/L2 regularization**: [0, 1e-5, 1e-4, 1e-3]
- **Optimizer**: [Adam, SGD, RMSprop]
- **Layer widths**: Vary dimension progression

## Troubleshooting

### Issue: Study runs very slowly

**Cause**: Training too many models or models are complex

**Solution**:
```bash
# Option 1: Reduce epochs
# Edit ablation_study.py, line ~80
epochs=40  # Instead of 80

# Option 2: Skip some components
# Comment out sections you don't need
# Around lines 220-280

# Option 3: Use subset of values
# Edit the for loops to use fewer values
for bottleneck_dim in [2, 4, 32]:  # Instead of [2, 4, 8, 16, 32]
```

### Issue: Out of memory errors

**Cause**: Network too deep or features too many

**Solution**:
```bash
# Reduce batch size (more frequent updates, smaller batches):
# Edit ablation_study.py
batch_size=8  # Instead of 16

# Or reduce feature dimensionality:
# In analyze: Test only 100% and 50% features, skip 25%
```

### Issue: Results look too good / too bad

**Cause**: Data leakage or configuration error

**Check**:
1. Data split is correct (separate train/test)
2. No test data in training
3. Labels are correct
4. Preprocessing is reproducible

### Issue: Visualization fails

**Cause**: Missing matplotlib or wrong data format

**Solution**:
```bash
# Reinstall visualization dependencies
pip install matplotlib seaborn

# Re-run analysis
python analyze_ablation.py
```

## Best Practices

### 1. Reproducibility
```python
# Always set seeds before running
os.environ["PYTHONHASHSEED"] = "41"
np.random.seed(41)
tf.random.set_seed(41)
```

### 2. Documentation
- Save results with timestamp
- Keep record of data used
- Note any changes to preprocessing

### 3. Comparison
- Always compare to baseline
- Use consistent metrics
- Report multiple metrics (not just F1)

### 4. Validation
- Use separate test set
- Don't optimize on test set
- Validate improvements on hold-out data

## References

### Academic Foundation
- **Ablation Study**: https://en.wikipedia.org/wiki/Ablation_study
- **Autoencoders**: https://www.deeplearningbook.org/contents/autoencoders.html
- **Hyperparameter Tuning**: https://arxiv.org/abs/1603.06393

### Technical Documentation
- **TensorFlow/Keras**: https://www.tensorflow.org/guide/keras
- **Scikit-learn Metrics**: https://scikit-learn.org/stable/modules/model_evaluation.html
- **Matplotlib/Seaborn**: https://matplotlib.org/

### In This Project
- [ABLATION_QUICK_START.md](ABLATION_QUICK_START.md) - Quick reference
- [scripts/ablation_study.py](scripts/ablation_study.py) - Implementation
- [scripts/analyze_ablation.py](scripts/analyze_ablation.py) - Analysis
- [scripts/autoencoder.py](scripts/autoencoder.py) - Base model

## FAQ

**Q: Why test components separately and not all combinations?**
A: Full factorial design (all combinations) = 5 × 4 × 4 × 3 × 3 = 720 models!
   Ablation is more tractable (20 models) and still provides valuable insights.

**Q: Can I run multiple ablation studies with different data?**
A: Yes! Each run creates `ablation_study_<timestamp>.json`
   Compare results to see if findings are consistent.

**Q: Should I retune other hyperparameters after finding best component values?**
A: Yes, components interact. After finding best values:
   1. Use them in pipeline.py
   2. Retune contamination threshold
   3. Validate on full test set

**Q: How do I know if improvement is statistically significant?**
A: Use cross-validation or bootstrap confidence intervals
   (Advanced topic, see THEORETICAL_FOUNDATION.md)

**Q: Can this ablation approach be applied to other models?**
A: Yes! This methodology works for any configurable model:
   - Isolation Forest parameters
   - VAE architectures
   - Gradient Boosting hyperparameters
   - etc.
