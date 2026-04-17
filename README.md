# Blockchain-Enabled Anomaly Detection in Educational Data with Autoencoders

A comprehensive machine learning system that combines **autoencoders for unsupervised anomaly detection** with **blockchain technology for data integrity** in educational settings. This project detects unusual patterns in student engagement and performance while maintaining tamper-resistant logs of all observations.

## Project Overview

### Background and Motivation

Educational platforms generate vast amounts of data from student interactions. Anomaly detection in this data can identify:
- Sudden drops in engagement
- Unusual assessment patterns
- Potential learning difficulties
- System errors or data integrity issues

By combining autoencoders (powerful unsupervised ML for anomaly detection) with blockchain (distributed ledgers for data integrity), this system ensures both **detection accuracy** and **data trustworthiness**.

## Project Structure

```
d:\HKR\Thesis\test\
├── data/                      # Input datasets
│   ├── studentInfo.csv
│   ├── studentAssessment.csv
│   ├── studentRegistration.csv
│   ├── studentVle.csv
│   ├── assessments.csv
│   ├── courses.csv
│   └── vle.csv
├── scripts/                   # Python modules
│   ├── config.py              # Configuration constants
│   ├── data_loader.py         # Data loading and merging
│   ├── preprocessing.py       # Feature engineering
│   ├── pipeline.py            # Unified training and testing pipeline
│   ├── autoencoder.py         # Autoencoder model implementation
│   ├── isolation_forest_baseline.py  # Isolation Forest baseline
│   ├── compare_baselines.py   # Baseline comparison script
│   └── blockchain_logger.py   # Blockchain integration
├── results/                   # Output files
│   ├── merged_dataset.csv
│   ├── student_profiles.csv
│   ├── feature_names.json
│   ├── model_comparison.json  # Comparison metrics
│   ├── predictions_comparison.csv  # Detailed predictions
│   └── roc_data.json          # ROC curve data
├── requirements.txt           # Python dependencies
└── README.md                  # This file
└── visualize_data.ipynb       # Data visualization notebook

```

## Installation and Setup

### Prerequisites
- Python 3.8+
- CUDA 11.x (optional, for GPU acceleration)
- Virtual environment (recommended)

### Step 1: Clone/Setup Repository
```bash
cd d:\HKR\Thesis\test
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run Data Setup
```bash
# Copy CSV files to data/ directory
# Files should be available in downloads
```

## Quick Start

### Option 1: Run Unified Pipeline (Data Loading → Training → Testing → Blockchain Logging)
```bash
python scripts/pipeline.py
```

This unified pipeline performs:
1. Load and explore data from multiple CSV sources
2. Preprocess and create student performance profiles
3. Train autoencoder model on normalized features
4. Inject synthetic Gaussian noise anomalies into test set
5. Evaluate autoencoder performance with optimized threshold
6. Initialize permissioned blockchain and log detected anomalies
7. Verify blockchain integrity with auditor access
8. Report comprehensive metrics (accuracy, precision, recall, F1, ROC AUC)

**Output files generated:**
- `results/merged_dataset.csv` - Complete merged dataset
- `results/student_profiles.csv` - Aggregated student performance profiles
- `results/feature_names.json` - List of features used for model training
- `blockchain_ledger.jsonl` - Audit trail of detected anomalies

### Option 2: Compare Autoencoder vs Isolation Forest
```bash
python scripts/compare_baselines.py
```

This script performs a comprehensive baseline comparison:
1. Load and preprocess educational data (uses unified pipeline)
2. Train both Autoencoder and Isolation Forest models
3. Inject identical synthetic anomalies for fair comparison
4. Evaluate both models using same evaluation metrics
5. Generate detailed performance comparison and ROC curves
6. Save results to `results/model_comparison.json`

**Output files generated:**
- `results/model_comparison.json` - Detailed metrics for both models
- `results/predictions_comparison.csv` - Per-sample predictions from both models
- `results/roc_data.json` - ROC curve data for visualization


## Evaluation Metrics
Both models are evaluated on:
- **Accuracy** - Overall correctness
- **Precision** - Fraction of detected anomalies that are true positives
- **Recall** - Fraction of true anomalies that were detected
- **F1 Score** - Harmonic mean of precision and recall
- **ROC AUC** - Area under the receiver operating characteristic curve

## Isolation Forest Baseline
**Location:** `scripts/isolation_forest_baseline.py`

Isolation Forest is a tree-based ensemble method that:
- Isolates observations through random feature/split selection
- Relies on the principle that anomalies are "few and far between", easier to isolate
- Provides O(n log n) time complexity, efficient for large datasets
- Makes no assumptions about data distribution

Key parameters:
- `contamination` - Expected proportion of anomalies (0.02 = 2%)
- `n_estimators` - Number of isolation trees (100)

## Autoencoder Method
The autoencoder compresses normal data through a bottleneck (4-dim) and reconstructs it. Normal data reconstructs well (low error), anomalies fail to reconstruct (high error). Reconstruction error is used as the anomaly score.

## Performance Results

Detailed comparison of Autoencoder vs Isolation Forest on the educational anomaly detection task:

| Metric | Autoencoder | Isolation Forest | Winner |
|--------|-------------|------------------|--------|
| **Accuracy** | 99.73% | 97.14% | ✓ Autoencoder |
| **Precision** | 97.99% | 88.04% | ✓ Autoencoder |
| **Recall** | 99.80% | 88.13% | ✓ Autoencoder |
| **F1 Score** | 98.88% | 88.08% | ✓ Autoencoder |
| **ROC AUC** | 99.96% | 99.17% | ✓ Autoencoder |

### Key Findings

- **Autoencoder** consistently outperforms Isolation Forest across all evaluation metrics
- **Superior anomaly detection**: 99.8% recall captures nearly all injected anomalies
- **Excellent discrimination**: 99.96% ROC AUC demonstrates excellent separation between normal and anomalous patterns
- **Deep learning advantage**: Autoencoder leverages feature compression through bottleneck architecture for better generalization
- **Isolation Forest trade-off**: Faster training and inference but lower detection sensitivity on complex patterns

## Dataset Description

The project uses 6 interconnected CSV files from the Open University Learning Analytics dataset:

| File | Records | Key Features |
|------|---------|--------------|
| **studentInfo.csv** | 32,593 | Demographics, education, disabilities, final results |
| **studentAssessment.csv** | 173,912 | Assessment IDs, scores, submission dates |
| **assessments.csv** | 206 | Assessment metadata, types, dates, weights |
| **studentRegistration.csv** | 32,593 | Registration and unregistration dates |
| **studentVle.csv** | 10.6M+ | VLE resource interactions, clicks, timestamps |
| **vle.csv** | 10,000+ | Virtual learning environment activities, types |
| **courses.csv** | 22 | Course metadata, presentation lengths |

