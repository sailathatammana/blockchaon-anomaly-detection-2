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
│   ├── train_pipeline.py      # Main training pipeline
│   ├── autoencoder.py         # Autoencoder model implementation
│   ├── isolation_forest_baseline.py  # Isolation Forest baseline
│   ├── compare_baselines.py   # Baseline comparison script
│   ├── test_autoencoder.py    # Unit tests
│   └── blockchain_logger.py   # Blockchain integration
├── notebooks/                 # Jupyter notebooks
│   ├── analysis.ipynb         # Complete analysis notebook
│   └── visualize_data.ipynb   # Data visualization
├── results/                   # Output files
│   ├── merged_dataset.csv
│   ├── student_profiles.csv
│   ├── feature_names.json
│   ├── model_comparison.json  # Comparison metrics
│   ├── predictions_comparison.csv  # Detailed predictions
│   └── roc_data.json          # ROC curve data
├── requirements.txt           # Python dependencies
└── README.md                  # This file
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

### Option 1: Run Full Pipeline
```bash
python scripts/train_pipeline.py
```

This will:
1. Load and explore data
2. Preprocess and engineer features
3. Train autoencoder model
4. Detect anomalies
5. Initialize blockchain and log results
6. Generate report

### Option 2: Test Autoencoder with Synthetic Anomalies
```bash
python scripts/test_autoencoder.py
```

This will:
1. Run the full pipeline
2. Inject synthetic Gaussian noise anomalies
3. Evaluate autoencoder performance
4. Report accuracy, precision, recall, F1 score, and ROC AUC

### Option 3: Compare Autoencoder vs Isolation Forest
```bash
python scripts/compare_baselines.py
```

This will:
1. Train both Autoencoder and Isolation Forest
2. Use identical synthetic anomalies for fair comparison
3. Generate detailed comparison metrics
4. Compare ROC curves
5. Save results to `results/model_comparison.json`

### Option 4: Interactive Jupyter Notebook
```bash
jupyter notebook notebooks/analysis.ipynb
```

## Usage

### Using the Data Loader
```python
from scripts.data_loader import EducationalDataLoader

loader = EducationalDataLoader(data_dir="data")
datasets = loader.load_all_datasets()
merged_df = loader.create_merged_dataset()
```

### Using the Preprocessor
```python
from scripts.preprocessing import EducationalDataPreprocessor

preprocessor = EducationalDataPreprocessor()
profiles = preprocessor.create_student_profiles(merged_df)
X, feature_names = preprocessor.prepare_for_autoencoder(profiles)
```

## Model Comparison: Autoencoder vs Isolation Forest

This project includes a comprehensive baseline comparison framework that evaluates the autoencoder against **Isolation Forest**, a industry-standard anomaly detection algorithm.

### Running the Comparison
```bash
python scripts/compare_baselines.py
```

This will:
1. Load and preprocess the educational data
2. Train both the Autoencoder and Isolation Forest models
3. Inject synthetic anomalies (Gaussian noise) into the test set
4. Evaluate both models using the same ground truth
5. Generate detailed comparison metrics and save results

### Comparison Outputs
- `results/model_comparison.json` - Detailed metrics for both models
- `results/predictions_comparison.csv` - Per-sample predictions from both models
- `results/roc_data.json` - ROC curve data for visualization

### Evaluation Metrics
Both models are evaluated on:
- **Accuracy** - Overall correctness
- **Precision** - Fraction of detected anomalies that are true positives
- **Recall** - Fraction of true anomalies that were detected
- **F1 Score** - Harmonic mean of precision and recall
- **ROC AUC** - Area under the receiver operating characteristic curve
- **Specificity** - True negative rate (normal samples correctly identified)
- **False Positive Rate** - Normal samples incorrectly flagged as anomalies

### Isolation Forest Baseline
**Location:** `scripts/isolation_forest_baseline.py`

Isolation Forest is a tree-based ensemble method that:
- Isolates observations through random feature/split selection
- Relies on the principle that anomalies are "few and far between", easier to isolate
- Provides O(n log n) time complexity, efficient for large datasets
- Makes no assumptions about data distribution

Key parameters:
- `contamination` - Expected proportion of anomalies (0.02 = 2%)
- `n_estimators` - Number of isolation trees (100)

### Autoencoder Method
The autoencoder compresses normal data through a bottleneck (4-dim) and reconstructs it. Normal data reconstructs well (low error), anomalies fail to reconstruct (high error). Reconstruction error is used as the anomaly score.

### Expected Performance
On the educational anomaly injection task:
- **Autoencoder** - Typically achieves 95-99% accuracy with deep feature learning
- **Isolation Forest** - Achieves 85-95% accuracy, computationally faster
- Both models handle high-dimensional data (55+ features) effectively

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

