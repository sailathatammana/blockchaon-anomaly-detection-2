"""Unified pipeline for educational anomaly detection.

REAL DATA APPROACH - UPDATED
This pipeline now uses REAL student outcomes instead of synthetic anomalies:

1. Load and preprocess data
2. Create anomaly labels from student outcomes (Pass/Distinction = normal, Fail/Withdrawn = abnormal)
3. Split data: Train autoencoder on NORMAL students only (unsupervised)
4. Test on: Normal students (validation) + Abnormal students (real anomalies)
5. Evaluate and log results to blockchain

See LABELING_STRATEGY.md for detailed documentation of the approach.
"""
import os
import sys
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score

# Set random seeds FIRST, before any TensorFlow imports
os.environ["PYTHONHASHSEED"] = "41"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import tensorflow as tf

# Import project modules
import config
from data_loader import EducationalDataLoader
from preprocessing import EducationalDataPreprocessor
from autoencoder import train_detector, predict_anomalies, score_anomalies, optimize_threshold
from blockchain_logger import PermissionedBlockchain

# Set seeds after imports
random.seed(config.SEED)
np.random.seed(config.SEED)
tf.random.set_seed(config.SEED)


def load_and_preprocess_data(data_dir: str = "data", results_dir: str = "results") -> Tuple:
    """Load, preprocess, and label data for real anomaly detection.
    
    REAL DATA APPROACH:
    - Creates labels from actual student outcomes (final_result)
    - Separates normal vs abnormal students before splitting
    - Trains autoencoder on normal students only
    
    Args:
        data_dir: Directory containing source CSV files
        results_dir: Directory for output results and profiles
    
    Returns:
        Tuple of:
            - X_train_scaled: Training features (normalized, normal students only)
            - X_test_scaled: Test features (normalized, normal + abnormal mix)
            - y_train: Training labels (all +1, only normal)
            - y_test: Test labels (+1 normal, -1 abnormal)
            - feature_names: List of feature names
            - scaler: Fitted scaler for inverse transformation
            - label_info: Dictionary with label statistics
    """
    data_path = Path(data_dir)
    results_path = Path(results_dir)
    results_path.mkdir(exist_ok=True)
    
    print("\n" + "█" * 70)
    print("█ STEP 1: DATA LOADING & LABELING (REAL DATA APPROACH)")
    print("█" * 70)
    
    try:
        # Load raw data
        loader = EducationalDataLoader(data_dir=str(data_path))
        loader.load_all_datasets()
        loader.get_summary_statistics()
        
        merged_df = loader.create_merged_dataset()
        print(f"\n✓ Merged dataset shape: {merged_df.shape}")
        print(f"  Rows:    {merged_df.shape[0]} samples")
        print(f"  Columns: {merged_df.shape[1]} features")
        print(f"  Missing values: {merged_df.isnull().sum().sum()}")
        merged_df.to_csv(results_path / "merged_dataset.csv", index=False)
        print(f"✓ Merged dataset saved")

        # Preprocess data
        print("\n" + "█" * 70)
        print("█ STEP 2: DATA PREPROCESSING & ANOMALY LABELING")
        print("█" * 70)
        
        preprocessor = EducationalDataPreprocessor()
        
        # Create student performance profiles
        profiles = preprocessor.create_student_profiles(merged_df)
        print(f"\n✓ Student profiles created")
        print(f"  Shape: {profiles.shape}")
        print(f"  Students: {profiles.shape[0]}")
        print(f"  Features: {profiles.shape[1]}")
        
        profiles.to_csv(results_path / "student_profiles.csv", index=False)
        print(f"✓ Profiles saved")
        
        # CREATE ANOMALY LABELS FROM REAL OUTCOMES
        print(f"\n" + "─" * 70)
        print("█ REAL DATA LABELING")
        print("─" * 70)
        
        y, label_info = preprocessor.create_anomaly_labels(
            profiles,
            normal_labels=config.NORMAL_OUTCOME_LABELS,
            abnormal_labels=config.ABNORMAL_OUTCOME_LABELS
        )
        
        print(f"\n✓ Anomaly labels created from student outcomes:")
        print(f"  NORMAL outcomes (Pass, Distinction):")
        print(f"    Count: {label_info['n_normal']} ({label_info['pct_normal']:.1f}%)")
        print(f"  ABNORMAL outcomes (Fail, Withdrawn):")
        print(f"    Count: {label_info['n_abnormal']} ({label_info['pct_abnormal']:.1f}%)")
        print(f"  Contamination Rate: {label_info['contamination_rate']:.2%}")
        print(f"  Unknown/Other outcomes:")
        print(f"    Count: {label_info['n_unknown']}")
        
        # Save label info
        with open(results_path / "label_info.json", 'w') as f:
            json.dump(label_info, f, indent=2)
        
        # Save labeled profiles
        profiles_labeled = profiles.copy()
        profiles_labeled['anomaly_label'] = y
        profiles_labeled.to_csv(results_path / "student_profiles_with_anomaly_labels.csv", index=False)
        print(f"✓ Labeled profiles saved")
        
        # Prepare features for autoencoder
        X, feature_names = preprocessor.prepare_for_autoencoder(profiles)
        print(f"\n✓ Features prepared for autoencoder")
        print(f"  Input shape: {X.shape}")
        print(f"  Num features: {len(feature_names)}")

        # Save feature names
        with open(results_path / "feature_names.json", 'w') as f:
            json.dump(feature_names, f, indent=2)
        print(f"✓ Feature names saved")
        
        # Split into train/test using REAL LABELS
        print("\n" + "█" * 70)
        print("█ STEP 3: DATA SPLIT (NORMAL/ABNORMAL SEPARATION)")
        print("█" * 70)
        print(f"\n✓ Splitting strategy:")
        print(f"  - Train: {config.NORMAL_TRAIN_FRACTION*100:.0f}% of NORMAL students (autoencoder trains unsupervised)")
        print(f"  - Test:  Remaining normal students + ALL abnormal students")
        
        X_train, X_test, y_train, y_test, train_idx, test_idx = \
            preprocessor.split_normal_abnormal(
                X, y,
                normal_train_fraction=config.NORMAL_TRAIN_FRACTION,
                random_state=config.SEED
            )
        
        n_train_normal = int(np.sum(y_train == 1))
        n_test_normal = int(np.sum(y_test == 1))
        n_test_abnormal = int(np.sum(y_test == -1))
        
        print(f"\n✓ Data split completed:")
        print(f"  Training set (NORMAL only):")
        print(f"    Size: {X_train.shape[0]} samples ({X_train.shape[0]/X.shape[0]*100:.1f}% of total)")
        print(f"    Label: All normal (y=+1)")
        print(f"  Test set (NORMAL + ABNORMAL):")
        print(f"    Total size: {X_test.shape[0]} samples ({X_test.shape[0]/X.shape[0]*100:.1f}% of total)")
        if X_test.shape[0] > 0:
            print(f"    Normal: {n_test_normal} ({n_test_normal/X_test.shape[0]*100:.1f}%)")
            print(f"    Abnormal: {n_test_abnormal} ({n_test_abnormal/X_test.shape[0]*100:.1f}%)")
        else:
            print(f"    Normal: {n_test_normal} (N/A%)")
            print(f"    Abnormal: {n_test_abnormal} (N/A%)")
        
        # Normalize features - ONLY using training data statistics (CRITICAL!)
        print("\n" + "█" * 70)
        print("█ STEP 4: FEATURE NORMALIZATION (On Training Data Only)")
        print("█" * 70)
        
        scaler, X_train_scaled = preprocessor.normalize_features(X_train)
        X_test_scaled = scaler.transform(X_test)

        print(f"\n✓ Features normalized using training data statistics")
        print(f"  Training set shape:   {X_train_scaled.shape}")
        print(f"  Test set shape:       {X_test_scaled.shape}")
        print(f"  Scaler fit on:        {len(X_train)} normal training samples")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler, label_info
            
    except Exception as e:
        print("\n" + "█" * 70)
        print("█ DATA LOADING ERROR")
        print("█" * 70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None, None, None


def train_and_evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test, 
                            feature_names, scaler, label_info) -> None:
    """Train autoencoder on normal data and evaluate on real normal + abnormal mix.
    
    REAL DATA EVALUATION:
    - Trains autoencoder on normal students only
    - Tests on real abnormal students (failures, withdrawals)
    - No synthetic anomalies injected
    - Evaluation compares predictions to real outcome labels
    
    Args:
        X_train_scaled: Training features (normal students only, normalized)
        X_test_scaled: Test features (normal + abnormal, normalized)
        y_train: Training labels (all +1, normal students)
        y_test: Test labels (+1 normal, -1 abnormal)
        feature_names: List of feature names
        scaler: Fitted scaler
        label_info: Dictionary with label statistics
    """
    if X_train_scaled is None:
        print("Data loading failed. Stopping.")
        return
    
    # Train autoencoder on NORMAL students only
    print("\n" + "█" * 70)
    print("█ STEP 5: AUTOENCODER TRAINING (NORMAL DATA ONLY)")
    print("█" * 70)
    print(f"\nTraining on {len(X_train_scaled)} NORMAL student samples...")
    print(f"(This may take a moment...)")
    
    model, threshold = train_detector(
        X_train_scaled, 
        contamination=0.0,  # No contamination expected in training (normal only)
        bottleneck_dim=config.BOTTLENECK_DIM,
        epochs=config.EPOCHS, 
        batch_size=config.BATCH_SIZE,
        random_state=config.SEED
    )
    
    print("\n✓ Training completed successfully!")
    print(f"  Default threshold: {threshold:.5f}")

    # Score test set (mix of normal and abnormal)
    print("\n" + "█" * 70)
    print("█ STEP 6: EVALUATE ON REAL DATA (NORMAL + ABNORMAL MIX)")
    print("█" * 70)
    
    test_scores = score_anomalies(model, X_test_scaled)
    
    print(f"\n✓ Reconstruction error scores computed for {len(X_test_scaled)} test samples")
    print(f"  Min score:   {test_scores.min():.5f}")
    print(f"  Max score:   {test_scores.max():.5f}")
    print(f"  Mean score:  {test_scores.mean():.5f}")
    print(f"  Std score:   {test_scores.std():.5f}")

    # Optimize threshold
    print("\n" + "█" * 70)
    print("█ STEP 7: THRESHOLD OPTIMIZATION (ON REAL LABELS)")
    print("█" * 70)
    
    if config.OPTIMIZE_THRESHOLD:
        print(f"\nOptimizing threshold for {config.THRESHOLD_METRIC.upper()} balance...")
        print(f"Using real outcome labels (not synthetic)...")
        optimal_threshold, best_score = optimize_threshold(
            model, X_test_scaled, y_test, metric=config.THRESHOLD_METRIC
        )
        print(f"✓ Optimal threshold found: {optimal_threshold:.5f}")
        print(f"  Expected {config.THRESHOLD_METRIC.upper()}: {best_score:.4f}")
    else:
        print("\nUsing default threshold from training...")
        optimal_threshold = threshold
        print(f"✓ Default threshold: {optimal_threshold:.5f}")
    
    # Evaluate model
    print("\n" + "█" * 70)
    print("█ STEP 8: MODEL EVALUATION (REAL DATA METRICS)")
    print("█" * 70)
    
    test_preds = predict_anomalies(model, X_test_scaled, optimal_threshold)
    
    accuracy = accuracy_score(y_test, test_preds)
    precision = precision_score(y_test, test_preds, pos_label=-1)
    recall = recall_score(y_test, test_preds, pos_label=-1)
    f1 = f1_score(y_test, test_preds, pos_label=-1)
    tn, fp, fn, tp = confusion_matrix(y_test, test_preds, labels=[1, -1]).ravel()
    
    # Calculate ROC AUC
    y_test_binary = (y_test == -1).astype(int)
    roc_auc = roc_auc_score(y_test_binary, test_scores)
    
    n_normal_test = int(np.sum(y_test == 1))
    n_abnormal_test = int(np.sum(y_test == -1))
    
    print("\n" + "=" * 50)
    print("DETECTION RESULTS (REAL DATA):")
    print("=" * 50)
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")
    print(f"\nDetected: {int(np.sum(test_preds == -1))}/{len(test_preds)} as anomalies")
    print(f"\n--- Confusion Matrix Breakdown ---")
    print(f"True Positives (Correctly found abnormal students):  {int(tp)}")
    print(f"False Positives (Normal flagged as abnormal):        {int(fp)}")
    print(f"False Negatives (Missed abnormal students):          {int(fn)}")
    print(f"True Negatives (Correctly labeled normal):           {int(tn)}")
    print(f"\n--- Test Set Composition ---")
    print(f"Normal students in test:     {n_normal_test}")
    print(f"Abnormal students in test:   {n_abnormal_test}")

    # Log to blockchain
    print("\n" + "█" * 70)
    print("█ STEP 9: BLOCKCHAIN LOGGING")
    print("█" * 70)
    print(f"\nLogging to permissioned blockchain ({config.LEDGER_PATH})...")
    
    blockchain = PermissionedBlockchain(ledger_path=config.LEDGER_PATH)
    
    # Login as detector
    blockchain.login("detector", "detector_password")
    print(f"✓ Authenticated as '{blockchain.current_user.username}' (role: {blockchain.current_user.role})")
    
    # Log each detected anomaly
    anomaly_indices = [int(i) for i, p in enumerate(test_preds) if p == -1]
    for idx in anomaly_indices:
        blockchain.log_anomaly(idx, {
            "score": float(test_scores[idx]),
            "true_label": int(y_test[idx]),
            "is_correct": bool((y_test[idx] == -1) and (test_preds[idx] == -1))
        })
    
    print(f"✓ Logged {len(anomaly_indices)} anomalies to blockchain")
    
    # Verify blockchain integrity
    blockchain.logout()
    blockchain.login("auditor", "auditor_password")
    is_valid = blockchain.verify_blockchain()
    if is_valid:
        print(f"✓ Blockchain verification passed (role: {blockchain.current_user.role})")
    else:
        print(f"✗ Blockchain verification FAILED - possible tampering detected!")
    
    blockchain.logout()

    # Save all results to JSON file
    output = {
        "approach": "real_data_labeling",
        "description": "Uses real student outcome labels (Pass/Distinction=normal, Fail/Withdrawn=abnormal)",
        "data_info": {
            "training_samples": len(X_train_scaled),
            "training_composition": "100% normal students",
            "test_samples": len(X_test_scaled),
            "test_normal": int(n_normal_test),
            "test_abnormal": int(n_abnormal_test),
            "contamination_rate": float(label_info['contamination_rate'])
        },
        "config": {
            "normal_train_fraction": config.NORMAL_TRAIN_FRACTION,
            "bottleneck_dim": config.BOTTLENECK_DIM,
            "epochs": config.EPOCHS,
            "batch_size": config.BATCH_SIZE,
            "isolation_forest_strategy": config.ISOLATION_FOREST_STRATEGY
        },
        "metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "tp": int(int(tp)),
            "fp": int(int(fp)),
            "fn": int(int(fn)),
            "tn": int(int(tn))
        },
        "threshold": float(optimal_threshold),
        "note": "These metrics are computed on REAL student outcome data, not synthetic anomalies"
    }
    out_path = Path("results") / "pipeline_outputs.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to {out_path}")
    
    print("\n" + "█" * 70)
    print("█ PIPELINE COMPLETE ✓")
    print("█" * 70)
    print("\nKey Points:")
    print("- Autoencoder trained on NORMAL students only (unsupervised)")
    print("- Evaluated on REAL abnormal students (failures, withdrawals)")
    print("- No synthetic anomalies - uses real outcome data")
    print("- See LABELING_STRATEGY.md for detailed approach documentation")
    print()


def main():
    """Run the complete anomaly detection pipeline with real data."""
    print("\n" + "█" * 70)
    print("█ EDUCATIONAL ANOMALY DETECTION PIPELINE")
    print("█ REAL DATA APPROACH (NO SYNTHETIC ANOMALIES)")
    print("█" * 70)
    
    # Load and preprocess data with real labeling
    X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler, label_info = \
        load_and_preprocess_data(data_dir="data", results_dir="results")
    
    # Train and evaluate model
    if X_train_scaled is not None:
        train_and_evaluate_model(X_train_scaled, X_test_scaled, y_train, y_test, 
                               feature_names, scaler, label_info)


if __name__ == "__main__":
    main()