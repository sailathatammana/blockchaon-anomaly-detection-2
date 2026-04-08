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
│   ├── data_loader.py         # Data loading and merging
│   ├── preprocessing.py       # Feature engineering
│   └── train_pipeline.py      # Main training pipeline
├── notebooks/                 # Jupyter notebooks
│   └── analysis.ipynb         # Complete analysis notebook
├── results/                   # Output files
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

### Option 2: Interactive Jupyter Notebook
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

