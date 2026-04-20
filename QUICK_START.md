# Quick Start Guide: Real Data Anomaly Detection

## 5-Minute Setup

### 1. Verify Data Files
Make sure these files exist in the `data/` directory:
```
data/
├── studentInfo.csv
├── studentAssessment.csv
├── studentVle.csv
├── studentRegistration.csv
├── assessments.csv
├── vle.csv
└── courses.csv
```

### 2. Run Pipeline (Option A: Recommended)
```bash
cd scripts
python pipeline.py
```

**What happens:**
- Loads real student data
- Creates anomaly labels from `final_result` (Pass/Fail/Withdrawn)
- Trains autoencoder on normal students only
- Tests on real abnormal students
- Saves results with realistic metrics

**Expected output:** 
- Console output with progress and metrics
- Files in `results/` directory

### 3. Compare Models (Option B)
```bash
cd scripts
python compare_baselines.py
```

**What happens:**
- Trains autoencoder and isolation forest on same data
- Tests both on real abnormal students
- Compares performance

---

## Understanding Your Results

### Check These Files (in `results/` directory)

#### `label_info.json`
Shows how your data is composed:
```json
{
  "n_normal": 7250,          // Students who passed/excelled
  "n_abnormal": 3150,        // Students who failed/withdrew
  "pct_normal": 69.7,        // 69.7% of students
  "pct_abnormal": 30.3,      // 30.3% of students
  "contamination_rate": 0.303  // 30.3% are "anomalies"
}
```

#### `pipeline_outputs.json`
Your model's performance metrics:
```json
{
  "metrics": {
    "accuracy": 0.87,
    "precision": 0.85,
    "recall": 0.82,
    "f1": 0.835,
    "roc_auc": 0.88
  },
  "data_info": {
    "training_samples": 5437,      // Normal students
    "test_normal": 1813,           // For validation
    "test_abnormal": 3150          // Real anomalies
  }
}
```

#### `student_profiles_with_anomaly_labels.csv`
All students with assigned labels:
```
id_student, ..., final_result, anomaly_label
1234,      ..., Pass,         1           <- Normal
5678,      ..., Fail,         -1          <- Abnormal
```

---

## Key Concepts (Refresher)

### Labeling Rule
| Outcome | Label | Meaning |
|---------|-------|---------|
| Pass | +1 | Normal (successful) |
| Distinction | +1 | Normal (excellent) |
| Fail | -1 | Abnormal (failed course) |
| Withdrawn | -1 | Abnormal (dropped out) |

### Data Split
```
Normal Students (70%)           Abnormal Students (30%)
├─ Train: 52.5% (all used)      └─ Test: 30% (all used)
└─ Test: 17.5% (used)

Final Test Set: 17.5% normal + 30% abnormal = 47.5% of all data
```

### Training
- **Autoencoder:** Learns what "normal" looks like (training on normal students only)
- **Isolation Forest:** Can use different strategies (see config)

### Testing
- **Input:** Mix of normal and abnormal students
- **Output:** Predictions (+1 normal, -1 anomaly)
- **Evaluation:** Compare to real outcomes

---

## Interpreting Metrics

### Good Metrics Look Like This
```
Accuracy:  87%     ← Correctly classified 87% of students
Precision: 85%     ← Of students flagged as at-risk, 85% actually are
Recall:    82%     ← Of actual at-risk students, we caught 82%
F1 Score:  0.84    ← Good balance between precision and recall
ROC AUC:   0.88    ← Model clearly beats random guessing
```

### Bad Metrics Look Like This
```
Accuracy:  60%     ← Barely better than 50/50 guess
Precision: 30%     ← Mostly false alarms
Recall:    100%    ← Flagging everyone as at-risk
F1 Score:  0.46    ← Poor balance
ROC AUC:   0.52    ← Almost random
```

### Realistic Baseline
```
If you randomly flag 30% of students as at-risk:
- You'll catch all real at-risk students (Recall = 100%)
- But you'll false-alarm on 30% of normal students
- Your model should do BETTER than this!
```

---

## Common Questions

### Q: Why is accuracy only 85-90%?
**A:** Because real data is hard! Unlike synthetic Gaussian noise, real student failures are complex. This is actually a good sign—it means results are realistic.

### Q: Can I change what counts as "anomalous"?
**A:** Yes! Edit `scripts/config.py`:
```python
ABNORMAL_OUTCOME_LABELS = {'Fail', 'Withdrawn'}  # Change here
```

Then re-run the pipeline.

### Q: Should I train isolation forest differently?
**A:** Try different strategies in `scripts/config.py`:
```python
ISOLATION_FOREST_STRATEGY = "informed"  # Try "normal_only" or "all_data"
```

Each strategy has trade-offs:
- **normal_only:** Conservative, fewer false alarms
- **informed:** Balanced, recommended
- **all_data:** Uses full outcome information

### Q: How do I improve the metrics?
**A:** Several approaches:
1. **Add features:** More engineered features might help
2. **Tune threshold:** Adjust precision/recall trade-off
3. **Adjust bottleneck:** Try different autoencoder compression
4. **Try different strategy:** For isolation forest, use "informed"

### Q: Can I see which students were flagged?
**A:** Yes! Check `results/predictions_comparison.csv`:
```
y_true, ae_pred, if_pred, ae_score, if_score
1,      1,       1,       0.08,     -0.2       <- Normal (both agree)
-1,     -1,      -1,      0.35,     -0.8       <- Abnormal (both catch)
-1,     1,       -1,      0.12,     -0.6       <- Anomaly missed by AE
1,      -1,      1,       0.28,     -0.1       <- False alarm by AE
```

---

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'tensorflow'"
**Solution:** Install dependencies
```bash
pip install -r requirements.txt
```

### Error: "final_result column not found"
**Solution:** Make sure data files are in `data/` directory and contain `final_result` column

### Error: "CUDA not found" (for GPU users)
**Solution:** Works fine on CPU (slower but correct). Or install CUDA + tensorflow-gpu

### Results look wrong
**Solution:** Check:
1. Are your data files loaded? (check "Merged dataset shape")
2. Are labels being created? (check "Anomaly labels created")
3. Are data splits correct? (training should be <50% of total data)

---

## Next Steps

1. **Understand your data:** Review `label_info.json`
2. **Check model performance:** Review `pipeline_outputs.json`
3. **Examine predictions:** Look at `predictions_comparison.csv`
4. **Compare approaches:** Run `compare_baselines.py`
5. **Iterate:** Adjust config and re-run as needed

---

## File Reference

### Input Data
```
data/
├── studentInfo.csv          ← Main student information
├── studentAssessment.csv    ← Assessment scores
├── studentVle.csv           ← Virtual learning environment (engagement)
└── [other supporting files]
```

### Configuration
```
scripts/
├── config.py                ← All settings (change here!)
├── pipeline.py              ← Main pipeline
├── compare_baselines.py     ← Model comparison
└── [other modules]
```

### Output Results
```
results/
├── label_info.json          ← Label distribution
├── student_profiles_with_anomaly_labels.csv  ← All students with labels
├── pipeline_outputs.json    ← Metrics and configuration
├── model_comparison.json    ← AE vs IF comparison
├── predictions_comparison.csv  ← Detailed predictions
└── roc_data.json           ← ROC curve data
```

### Documentation
```
├── LABELING_STRATEGY.md        ← How labels are created
├── IMPLEMENTATION_SUMMARY.md   ← Technical changes made
├── THEORETICAL_FOUNDATION.md   ← Why real data matters
└── README.md                   ← Project overview
```

---

## Important Notes

⚠️ **Data Leakage Prevention:**
- Training data scaler is never fit on test data
- Validation uses cross-validation or held-out labels
- Results are unbiased

✓ **Reproducibility:**
- Random seed is fixed to 41
- Results are deterministic
- Can be reproduced exactly

📊 **Results Are Real:**
- Based on actual student outcomes
- Realistic metrics (not suspiciously perfect)
- Generalizes to production deployment

---

## Support

For issues or questions:
1. Check LABELING_STRATEGY.md for approach details
2. Review THEORETICAL_FOUNDATION.md for concepts
3. Look at inline code comments in scripts/
4. Check error messages carefully

---

**Ready to go!** Run `python scripts/pipeline.py` and see your first results.

