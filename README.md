# Blockchain-Enabled Anomaly Detection in Educational Data with Autoencoders

A comprehensive machine learning system that combines **autoencoders for unsupervised anomaly detection** with a **permissioned Proof-of-Authority (PoA) blockchain for data integrity and provenance** in educational settings. This project detects unusual patterns in student engagement and performance while maintaining tamper-evident records of detected anomalies.

## Project Overview

### Background and Motivation

Educational platforms generate large amounts of data from student interactions. Anomaly detection in this data can help identify:

- Sudden changes in engagement
- Unusual assessment patterns
- Potential learning difficulties
- Unusual student behavioural profiles
- System errors or data integrity issues

The machine learning component is responsible for detecting anomalous patterns, while the blockchain component provides integrity and provenance for the resulting anomaly records.

The project uses the **Open University Learning Analytics Dataset (OULAD)** and evaluates Autoencoder, Variational Autoencoder (VAE), Isolation Forest, and PCA-based anomaly detection.

---

## Project Structure

```text
project/
├── data/                            # OULAD input data
├── notebooks/                       # Analysis and visualization notebooks
├── outputs/                         # Generated figures and outputs
├── results/                         # Generated experimental results
├── scripts/
│   ├── analyze_ablation.py          # Analysis of ablation experiments
│   ├── autoencoder.py               # Autoencoder implementation
│   ├── blockchain_logger.py         # Blockchain/provenance logging utilities
│   ├── compare_baselines.py         # Model comparison experiments
│   ├── config.py                    # Experimental configuration
│   ├── data_loader.py               # OULAD data loading and merging
│   ├── isolation_forest_baseline.py # Isolation Forest baseline
│   ├── pipeline.py                  # Main experimental pipeline
│   ├── poa_blockchain.py            # Permissioned PoA blockchain
│   ├── preprocessing.py             # Feature engineering and preprocessing
│   ├── run_ablation.py              # Ablation experiments
│   ├── run_multiseed_comparison.py  # Five-seed robustness evaluation
│   ├── statistical_comparison.py    # Statistical model comparison
│   └── vae_baseline.py              # Variational Autoencoder baseline
├── requirements.txt                 # Fixed Python dependency versions
└── README.md                         # This file
```

The `results/`, `outputs/`, and `scripts/results/` directories contain files generated when experiments are executed. Generated outputs are not necessarily stored permanently in the repository.

---

## Dataset Description

The project uses the **Open University Learning Analytics Dataset (OULAD)**.

The dataset contains information about student demographics, assessments, course registrations, and interactions with the Virtual Learning Environment (VLE).

The main dataset files used are:

| File | Description |
|------|-------------|
| **studentInfo.csv** | Student demographics, education information and final results |
| **studentAssessment.csv** | Student assessment scores and submission information |
| **assessments.csv** | Assessment metadata, types, dates and weights |
| **studentRegistration.csv** | Student registration and unregistration information |
| **studentVle.csv** | Student interactions with VLE resources |
| **vle.csv** | Virtual learning environment activity information |
| **courses.csv** | Course and presentation information |

Dataset reference:

Kuzilek, J., Hlosta, M., & Zdrahal, Z. (2017). Open University Learning Analytics dataset. *Scientific Data, 4*, 170171.

DOI: 10.1038/sdata.2017.171

---

## Installation and Setup

### Prerequisites

- Python
- Virtual environment (recommended)
- OULAD dataset

### Step 1: Clone the Repository

Clone the repository and navigate to the project directory.

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

The `requirements.txt` file contains fixed package versions to improve reproducibility.

### Step 4: Add the Dataset

Place the required OULAD CSV files inside the `data/` directory.

---

## Quick Start

### Run the Unified Pipeline

```bash
python scripts/pipeline.py
```

The unified pipeline performs:

1. Load and merge the OULAD dataset
2. Create student-level educational profiles
3. Prepare features for Autoencoder training
4. Split the data into training, validation, and test sets using a 60/15/25 split
5. Standardize features using statistics fitted only on the training set
6. Train the Autoencoder
7. Inject synthetic anomalies into the validation and test sets
8. Select the anomaly threshold using validation data only
9. Apply the fixed threshold to the held-out test set
10. Evaluate anomaly detection performance
11. Log detected anomalies to the permissioned PoA blockchain
12. Replicate blocks across three authorized validator ledgers
13. Verify blockchain integrity and replica consistency
14. Save experimental results

---

## Data Preprocessing

The preprocessing pipeline prepares the OULAD data for anomaly detection.

The main steps include:

1. Loading and merging the relevant OULAD tables
2. Creating student-level educational profiles
3. Handling missing values
4. Encoding categorical variables
5. Removing identifiers and outcome labels from Autoencoder inputs
6. Splitting the data into training, validation, and test sets
7. Standardizing numerical features

Feature standardization is performed using `StandardScaler`.

The scaler is fitted only on the training set:

```text
Training data → fit StandardScaler
Validation data → transform using training scaler
Test data → transform using training scaler
```

This prevents information from the validation or test sets from influencing preprocessing.

---

## Train / Validation / Test Split

The revised experimental protocol uses:

- **60% training data**
- **15% validation data**
- **25% test data**

Each partition has a separate purpose:

- **Training set** — used for model training
- **Validation set** — used for anomaly-threshold selection
- **Test set** — used only for final evaluation

The test labels are not used during threshold optimization.

---

## Synthetic Anomaly Detection

Synthetic anomalies are injected after the dataset has been split.

For the main OULAD synthetic experiment, approximately **12% synthetic anomalies** are injected into the validation and test sets.

The validation anomalies are used to select the anomaly threshold.

The test anomalies are used only for final model evaluation.

This provides controlled anomaly labels for evaluating detection performance.

---

## Threshold Optimization

The anomaly threshold is selected using the validation set only.

The procedure is:

1. Train the model using the training set
2. Calculate reconstruction errors or anomaly scores on validation data
3. Evaluate candidate thresholds on the validation set
4. Select the threshold that maximizes validation F1-score
5. Fix the selected threshold
6. Apply the fixed threshold to the held-out test set

The test set is not used for threshold optimization.

---

## Autoencoder Method

The Autoencoder learns to reconstruct normal educational patterns through a compressed latent representation.

The basic process is:

```text
Input Features
      ↓
    Encoder
      ↓
Bottleneck Representation
      ↓
    Decoder
      ↓
Reconstructed Features
```

Normal patterns are expected to produce relatively low reconstruction error, while unusual patterns produce higher reconstruction error.

The reconstruction error is therefore used as the anomaly score.

The main experimental Autoencoder configuration is kept separate from alternative configurations evaluated in the ablation experiments.

---

## Variational Autoencoder Baseline

A Variational Autoencoder (VAE) is included as a deep-learning comparison model.

The VAE learns a probabilistic latent representation rather than a deterministic bottleneck.

The implementation is available in:

```text
scripts/vae_baseline.py
```

The VAE is evaluated using the same experimental framework to provide a comparison with the standard Autoencoder.

---

## Isolation Forest Baseline

Isolation Forest is included as a classical anomaly detection baseline.

The implementation is available in:

```text
scripts/isolation_forest_baseline.py
```

Isolation Forest identifies unusual observations by isolating samples using randomly generated decision trees.

The baseline is evaluated using the same train/validation/test protocol, with the held-out test set reserved for final evaluation.

---

## PCA Baseline

PCA-based anomaly detection is also included as a simpler reconstruction-based baseline.

PCA reconstructs samples from a lower-dimensional representation, and reconstruction error is used as an anomaly score.

This provides an additional comparison with the neural-network-based reconstruction methods.

---

## Model Comparison

Model comparison experiments evaluate:

- Autoencoder
- Variational Autoencoder
- Isolation Forest
- PCA

The models are compared using the same experimental splits and evaluation procedure.

The relevant comparison code is available in:

```text
scripts/compare_baselines.py
```

---

## Repeated-Seed Evaluation

To evaluate whether the results are stable across different random initializations, the main model comparison is repeated across five random seeds:

```text
41
42
43
44
45
```

The experiment is implemented in:

```text
scripts/run_multiseed_comparison.py
```

Results are summarized using:

- Mean performance
- Standard deviation

This provides an indication of variability across repeated experimental runs.

---

## Statistical Comparison

Additional statistical comparison of model performance is implemented in:

```text
scripts/statistical_comparison.py
```

This supports the repeated-seed analysis by comparing model performance across repeated runs.

---

## Ablation Analysis

Ablation experiments investigate how different Autoencoder configurations and anomaly settings affect detection performance.

The relevant scripts are:

```text
scripts/run_ablation.py
scripts/analyze_ablation.py
```

The ablation experiments are treated separately from the main experimental configuration.

They are used as sensitivity analyses and do not replace the configuration used to produce the primary results.

---

## Evaluation Metrics

Models are evaluated using several metrics, including:

- **Accuracy** — Overall classification correctness
- **Precision** — Fraction of detected anomalies that are true anomalies
- **Recall** — Fraction of true anomalies successfully detected
- **F1 Score** — Harmonic mean of precision and recall
- **ROC AUC** — Ability to discriminate between normal and anomalous samples
- **PR AUC** — Precision-recall performance under class imbalance
- **Specificity** — Ability to correctly identify normal samples
- **Balanced Accuracy** — Average performance across both classes
- **Matthews Correlation Coefficient (MCC)** — Balanced measure based on all confusion-matrix outcomes

The evaluation also reports:

- True Positives
- False Positives
- False Negatives
- True Negatives

---

## Outcome-Based OULAD Evaluation

The project also contains an evaluation using OULAD final course outcomes.

For this experiment:

- `Pass` and `Distinction` are treated as the reference normal group
- `Fail` and `Withdrawn` are treated as an **at-risk/outcome-based proxy group**

Failing or withdrawing from a course is not considered proof that a student is statistically anomalous.

This experiment therefore evaluates whether unusual educational profiles are associated with adverse academic outcomes rather than treating those outcomes as definitive anomaly ground truth.

The experiment is retrospective and uses aggregated course-level information.

It should not be interpreted as an early-warning prediction experiment.

---

## Gender Subgroup Error Analysis

The OULAD evaluation also includes subgroup error analysis by gender.

For each subgroup, the analysis reports:

- Number of samples
- True Positives
- False Positives
- False Negatives
- True Negatives
- False Positive Rate
- False Negative Rate

This analysis is intended to provide an initial indication of whether model errors are distributed differently across the available gender groups.

It is not intended to constitute a complete fairness assessment.

---

## Permissioned Proof-of-Authority Blockchain

Detected anomalies are logged to a permissioned Proof-of-Authority blockchain.

The implementation is available in:

```text
scripts/poa_blockchain.py
```

The network contains three authorized validator identities:

```text
validator_1
validator_2
validator_3
```

Only validators belonging to the authorized validator list are permitted to create blocks.

Validator authority rotates using a simple round-robin mechanism.

---

## Blockchain Block Creation

When an anomaly is logged, the blockchain creates a block containing the anomaly provenance information.

Each block contains information including:

- Block index
- Timestamp
- Validator identity
- Anomaly data
- Previous block hash
- SHA-256 block hash
- HMAC-SHA256 authentication tag

The block is then replicated across all three validator ledgers.

---

## Blockchain Integrity Verification

Blockchain verification checks:

1. The block creator is an authorized validator
2. The previous-block hash is correct
3. The stored SHA-256 block hash matches the block contents
4. The HMAC-SHA256 authentication tag is valid
5. All three validator ledgers contain consistent blockchain replicas

If one validator ledger is modified after block creation, its cryptographic values no longer correspond to the stored data and the inconsistency can be detected.

---

## Security and Trust Assumptions

The blockchain implementation is a proof-of-concept permissioned provenance mechanism.

The validator identities operate within an authorized environment.

HMAC-SHA256 provides symmetric authentication and does not provide the same security properties as asymmetric digital signatures.

A party with access to the shared authentication key could potentially generate a valid authentication tag.

The implementation therefore demonstrates tamper-evident provenance under the stated trust assumptions rather than absolute immutability against every possible attacker.

---

## Blockchain Logging

Blockchain logging is integrated into the main pipeline.

Running:

```bash
python scripts/pipeline.py
```

logs detected anomalies to the permissioned PoA blockchain.

The pipeline then verifies:

- Blockchain integrity
- Authorized validator participation
- Hash-chain consistency
- Authentication tags
- Replication across all three validator ledgers

---

## Generated Outputs

Experimental outputs are generated locally when the corresponding scripts are executed.

Depending on the experiment, generated files may be stored in:

```text
results/
outputs/
scripts/results/
```

Generated outputs may include:

- Merged datasets
- Student profiles
- Feature names
- Training metrics
- Model evaluation metrics
- Predictions
- ROC data
- Ablation results
- Repeated-seed results
- Statistical comparison results
- Blockchain ledgers
- Pipeline output files
- Figures and plots

These are generated experimental outputs and are not necessarily stored permanently in the repository.

Running the relevant scripts recreates the corresponding output files.

---

## Reproducibility

The repository includes fixed dependency versions in:

```text
requirements.txt
```

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

The repository provides the code used for:

- Data preprocessing
- Autoencoder training
- Variational Autoencoder comparison
- Isolation Forest comparison
- PCA comparison
- Validation-based threshold selection
- Synthetic anomaly experiments
- Outcome-based evaluation
- Ablation analysis
- Five-seed robustness evaluation
- Statistical model comparison
- Gender subgroup error analysis
- Permissioned PoA blockchain logging
- Validator-ledger replication
- Blockchain integrity verification

Machine learning results may vary slightly depending on hardware, operating system, and numerical libraries. The repeated-seed experiments are included to provide an indication of this variability.

---

## Performance Results

Performance results are generated by the current experimental scripts and stored in the corresponding output directories.

The README does not contain a fixed performance table because experimental results were updated during the revision process.

For results corresponding to the current experimental protocol, refer to:

- The revised manuscript
- Generated experiment outputs
- Repeated-seed result files
- Statistical comparison outputs

This avoids confusion between results produced by earlier development versions and the current revised experiments.

---

## Important Interpretation Notes

### Synthetic Anomaly Evaluation

Synthetic anomaly experiments provide controlled ground-truth labels by introducing known perturbations into the validation and test sets.

These experiments evaluate whether the models can identify intentionally perturbed samples.

### Outcome-Based Evaluation

The real-outcome OULAD experiment uses academic outcomes as proxy labels.

Failing or withdrawing does not inherently mean that a student is anomalous.

The experiment instead investigates whether unusual educational profiles are associated with students experiencing adverse academic outcomes.

### Detection and Blockchain Integrity

The two main components of the framework have separate responsibilities:

**Anomaly Detection**

- Learns educational patterns
- Calculates anomaly scores
- Identifies unusual samples

**Blockchain Provenance**

- Records detected anomalies
- Maintains hash-linked records
- Replicates records across authorized validators
- Verifies stored-record integrity
- Detects inconsistencies between validator copies

The blockchain does not improve the Autoencoder's detection accuracy. Its purpose is to provide provenance and integrity protection for anomaly records after detection.

---

## Code Availability

This repository contains the OULAD implementation used to support the experiments described in the associated manuscript.

The main implementation includes:

- Autoencoder anomaly detection
- Variational Autoencoder baseline
- Isolation Forest baseline
- PCA baseline
- Validation-only threshold selection
- Ablation experiments
- Five-seed robustness evaluation
- Statistical model comparison
- Outcome-based evaluation
- Subgroup error analysis
- Permissioned PoA blockchain
- Replicated validator ledgers
- Blockchain integrity verification

---

## Associated Manuscript

**Autoencoder-Based Anomaly Detection with Blockchain-Assisted Provenance for Educational Data**

This repository corresponds to the revised OULAD experimental implementation described in the manuscript.