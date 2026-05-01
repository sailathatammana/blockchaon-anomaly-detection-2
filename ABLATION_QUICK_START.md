# Ablation Study Quick Start Guide

## What is an Ablation Study?

An ablation study systematically removes or modifies individual components of a model to understand their contribution to performance. This helps answer: "Which components are actually important?"

## Quick Run (5-15 minutes)

### Step 1: Run the Ablation Study
```bash
cd scripts
python ablation_study.py
```

This will:
- Load and preprocess your data
- Train a baseline model
- Test 5 different component variations
- Save raw results to `results/ablation_study.json`

**Expected output:**
```
█ ABLATION STUDY: Autoencoder Component Analysis
█████████████████████████████████████████████████...

[1/7] Loading and preprocessing data...
  ✓ Training samples: 6,513 (normal)
  ✓ Validation samples: 2,171 (normal)
  ✓ Test samples: 3,118 (445 normal, 2,673 abnormal)
  ✓ Features: 55

[2/7] Training baseline model...
  ✓ F1: 0.9491, ROC-AUC: 0.9530

[3/7] Testing bottleneck dimensions...
[4/7] Testing network depths...
[5/7] Testing dropout rates...
[6/7] Testing activation functions...
[7/7] Testing feature subsets...
```

### Step 2: Analyze Results
```bash
python analyze_ablation.py
```

This will:
- Load results from `ablation_study.json`
- Create comparison charts
- Generate heatmaps
- Produce a detailed analysis report

**Output files:**
- `results/ablation_comparison.png` - 6-panel comparison chart
- `results/ablation_heatmap.png` - Performance heatmaps
- `results/ablation_analysis_report.txt` - Detailed text report

## What Gets Tested

| Component | Values Tested | Impact |
|-----------|---------------|--------|
| **Bottleneck Dimension** | 2, 4, 8, 16, 32 | Controls compression level |
| **Network Depth** | 1, 2, 3, 4 layers | Controls architecture capacity |
| **Dropout Rate** | 0.0, 0.1, 0.2, 0.3 | Controls regularization |
| **Activation Function** | relu, sigmoid, tanh | Controls non-linearity |
| **Feature Subset** | 100%, 50%, 25% | Tests dimensionality |

## Understanding Results

### Key Metrics
- **F1 Score**: Harmonic mean of precision & recall (best overall metric)
- **ROC-AUC**: Area under the receiver operating characteristic curve
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **Specificity**: True negatives / (True negatives + False positives)

### Reading the Charts

1. **Bottleneck Dimension Impact**
   - X-axis: Compression level (2=extreme, 32=minimal)
   - Y-axis: F1 Score / ROC-AUC
   - Interpretation: Which compression level works best?

2. **Network Depth Impact**
   - X-axis: Number of hidden layers (1-4)
   - Y-axis: F1 Score / ROC-AUC
   - Interpretation: Is a deeper network better?

3. **Dropout Rate Impact**
   - X-axis: Dropout probability (0.0=no dropout, 0.3=high dropout)
   - Y-axis: F1 Score / ROC-AUC
   - Interpretation: How much regularization is needed?

4. **Activation Function Impact**
   - Bar chart comparing relu vs sigmoid vs tanh
   - Interpretation: Which activation works best?

5. **Feature Subset Impact**
   - Bar chart showing 25%, 50%, 100% of features
   - Interpretation: Do we need all features?

## Expected Baseline Performance

From the current configuration (bottleneck=4, depth=3, dropout=0.1, relu):
- **F1 Score**: ~0.9491
- **ROC-AUC**: ~0.9530
- **Accuracy**: ~95.8%
- **Precision**: ~0.9398
- **Recall**: ~0.9585
- **Specificity**: ~0.8360

## Common Findings

### Typical Patterns

1. **Bottleneck dimension**: Usually 4-8 is optimal (good compression without losing info)
2. **Network depth**: Often 2-3 layers is best (more depth → more parameters but risk of overfitting)
3. **Dropout**: Moderate values (0.1-0.2) often best
4. **Activation**: ReLU typically outperforms sigmoid/tanh
5. **Features**: 50-100% of features usually needed (variance-based selection works well)

### Interpreting Improvements

- **+2% improvement**: Significant, consider using
- **+0.5-2% improvement**: Marginal, depends on computational cost tradeoff
- **-0.5% to +0.5%**: Likely noise, baseline is fine
- **>-0.5% degradation**: Avoid this configuration

## Next Steps

1. **Review the charts** to understand which components matter
2. **Read the analysis report** for detailed findings
3. **Identify top 2-3 configurations** that improve baseline
4. **Update config.py** with the best configuration
5. **Retrain the full pipeline** with the new settings
6. **Evaluate on hold-out test set** to confirm improvements

## Customizing the Ablation Study

To test different values, edit [scripts/ablation_study.py](scripts/ablation_study.py):

```python
# Line ~150: Bottleneck dimensions
for bottleneck_dim in [2, 4, 8, 16, 32]:  # ← Edit here

# Line ~165: Network depths
for depth in [1, 2, 3, 4]:  # ← Edit here

# Line ~180: Dropout rates
for dropout_rate in [0.0, 0.1, 0.2, 0.3]:  # ← Edit here

# Line ~195: Activation functions
for activation in ["relu", "sigmoid", "tanh"]:  # ← Edit here

# Line ~210: Feature subsets
for pct in [1.0, 0.5, 0.25]:  # ← Edit here
```

## Troubleshooting

**Q: The study is too slow**
- A: Reduce the number of values tested for each component
- A: Reduce epochs in [scripts/ablation_study.py](scripts/ablation_study.py) (line 80)

**Q: Results show no clear winner**
- A: This is normal! It means the baseline is already well-optimized
- A: Try testing more extreme values (e.g., bottleneck = [1, 2, 32, 64])

**Q: Some configurations crash**
- A: Very small bottleneck dimensions (< 2) or depths (< 1) may cause issues
- A: Ensure bottleneck_dim is at least 1

**Q: Results don't match the report**
- A: Make sure both scripts ran successfully without errors
- A: Delete [results/ablation_study.json](results/ablation_study.json) and re-run from scratch

## References

- **Ablation Study Introduction**: https://en.wikipedia.org/wiki/Ablation_study
- **Our Autoencoder Design**: See [scripts/autoencoder.py](../scripts/autoencoder.py)
- **Detailed Ablation Guide**: See [ABLATION_STUDY_GUIDE.md](ABLATION_STUDY_GUIDE.md)
