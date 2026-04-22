"""Unified pipeline for educational anomaly detection.

Orchestrates the complete workflow:
1. Load and preprocess data
2. Train autoencoder model
3. Inject synthetic anomalies
4. Evaluate and log results to blockchain
"""
import os
import sys
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from sklearn.model_selection import train_test_split
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
    """Load and preprocess data for anomaly detection.
    
    Args:
        data_dir: Directory containing source CSV files
        results_dir: Directory for output results and profiles
    
    Returns:
        Tuple of:
            - X_train_scaled: Training set features (normalized)
            - X_test_scaled: Test set features (normalized)
            - feature_names: List of feature names
            - scaler: Fitted scaler for inverse transformation
    """
    data_path = Path(data_dir)
    results_path = Path(results_dir)
    results_path.mkdir(exist_ok=True)
    
    print("\n" + "█" * 70)
    print("█ STEP 1: DATA LOADING")
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
        print("█ STEP 2: DATA PREPROCESSING")
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
        
        # Extract student IDs before preparing for autoencoder
        student_ids = profiles['id_student'].values if 'id_student' in profiles.columns else None
        
        # Prepare features for autoencoder
        X, feature_names = preprocessor.prepare_for_autoencoder(profiles)
        print(f"\n✓ Features prepared for autoencoder")
        print(f"  Input shape: {X.shape}")
        print(f"  Num features: {len(feature_names)}")

        # Save feature names
        with open(results_path / "feature_names.json", 'w') as f:
            json.dump(feature_names, f, indent=2)
        print(f"✓ Feature names saved")
        
        # Split into train/test
        print("\n" + "█" * 70)
        print("█ STEP 3: TRAIN/TEST SPLIT")
        print("█" * 70)
        
        if student_ids is not None:
            X_train, X_test, train_ids, test_ids = train_test_split(
                X, student_ids, test_size=config.TEST_SIZE, random_state=config.SEED
            )
        else:
            X_train, X_test = train_test_split(
                X, test_size=config.TEST_SIZE, random_state=config.SEED
            )
            test_ids = None
        
        print(f"\n✓ Data split completed ({(1-config.TEST_SIZE)*100:.0f}/{config.TEST_SIZE*100:.0f})")
        print(f"  Training set:   {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
        print(f"  Test set:       {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
        
        # Normalize features
        print("\n" + "█" * 70)
        print("█ STEP 4: FEATURE NORMALIZATION")
        print("█" * 70)
        
        scaler, X_train_scaled = preprocessor.normalize_features(X_train)
        X_test_scaled = scaler.transform(X_test)

        print(f"\n✓ Features normalized to [0, 1] range")
        print(f"  Training set shape:   {X_train_scaled.shape}")
        print(f"  Test set shape:       {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, feature_names, scaler, test_ids
            
    except Exception as e:
        print("\n" + "█" * 70)
        print("█ DATA LOADING ERROR")
        print("█" * 70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None


def train_and_evaluate_model(X_train_scaled, X_test_scaled, feature_names, scaler, test_ids=None) -> None:
    """Train autoencoder, inject anomalies, evaluate performance, and blockchain logging.
    
    Args:
        X_train_scaled: Training set features (normalized)
        X_test_scaled: Test set features (normalized)
        feature_names: List of feature names
        scaler: Fitted scaler for inverse transformation
    """
    if X_train_scaled is None:
        print("Data loading failed. Stopping.")
        return
        
    # Train autoencoder
    print("\n" + "█" * 70)
    print("█ STEP 5: AUTOENCODER TRAINING")
    print("█" * 70)
    print(f"\nTraining on {len(X_train_scaled)} samples (This may take a moment...)")
    
    model, threshold = train_detector(
        X_train_scaled, 
        contamination=config.INJECTION_FRAC, 
        bottleneck_dim=config.BOTTLENECK_DIM,
        epochs=config.EPOCHS, 
        batch_size=config.BATCH_SIZE,
        random_state=config.SEED
    )
    
    print("\nTraining completed successfully!")

    # Inject synthetic anomalies
    print("\n" + "█" * 70)
    print(f"█ STEP 6: ANOMALY INJECTION ({config.INJECTION_FRAC*100:.0f}% Gaussian Noise)")
    print("█" * 70)
    
    np.random.seed(config.SEED)
    n_test = len(X_test_scaled)
    n_anomalies = int(n_test * config.INJECTION_FRAC)
    
    # Create Ground Truth Array: 1 for normal, -1 for anomaly
    y_true = np.ones(n_test)
    
    # Randomly select test set indices to corrupt
    anomaly_indices = np.random.choice(n_test, size=n_anomalies, replace=False)
    y_true[anomaly_indices] = -1
    
    # Corrupt these samples by adding Gaussian noise
    noise = np.random.normal(loc=0.0, scale=config.NOISE_SCALE, size=(n_anomalies, X_test_scaled.shape[1]))
    X_test_scaled[anomaly_indices] += noise
    
    print(f"✓ Successfully injected {n_anomalies} anomalies into the test set.")

    # Score and optimize threshold
    print("\n" + "█" * 70)
    print("█ STEP 7: THRESHOLD OPTIMIZATION")
    print("█" * 70)
    
    test_scores = score_anomalies(model, X_test_scaled)
    
    if config.OPTIMIZE_THRESHOLD:
        print(f"\nOptimizing threshold for {config.THRESHOLD_METRIC.upper()} balance...")
        optimal_threshold, best_score = optimize_threshold(
            model, X_test_scaled, y_true, metric=config.THRESHOLD_METRIC
        )
        print(f"✓ Optimal threshold found: {optimal_threshold:.5f}")
        print(f"  Expected score: {best_score:.4f}")
    else:
        print("\nUsing default threshold from training...")
        optimal_threshold = threshold
        print(f"✓ Default threshold: {optimal_threshold:.5f}")
    
    # Evaluate model
    print("\n" + "█" * 70)
    print("█ STEP 8: MODEL EVALUATION")
    print("█" * 70)
    
    test_preds = predict_anomalies(model, X_test_scaled, optimal_threshold)
    
    accuracy = accuracy_score(y_true, test_preds)
    precision = precision_score(y_true, test_preds, pos_label=-1)
    recall = recall_score(y_true, test_preds, pos_label=-1)
    f1 = f1_score(y_true, test_preds, pos_label=-1)
    tn, fp, fn, tp = confusion_matrix(y_true, test_preds, labels=[1, -1]).ravel()
    
    # Calculate ROC AUC
    y_true_binary = (y_true == -1).astype(int)
    roc_auc = roc_auc_score(y_true_binary, test_scores)
    
    print("\n" + "=" * 50)
    print("DETECTION RESULTS:")
    print("=" * 50)
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")
    print(f"Detected:  {int(np.sum(test_preds == -1))}/{len(test_preds)}")
    print("\n--- Detection Breakdown ---")
    print(f"True Positives (Correctly caught injected anomalies):  {tp}")
    print(f"False Positives (Normal profiles flagged as anomaly):  {fp}")
    print(f"False Negatives (Missed injected anomalies):           {fn}")
    print(f"True Negatives (Correctly labeled normal profiles):    {tn}")

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
        student_id = int(test_ids[idx]) if test_ids is not None else None
        blockchain.log_anomaly(idx, {"score": float(test_scores[idx])}, student_id=student_id)
    
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

    # 9) Save all results to JSON file for later analysis and reporting
    # Store configuration, metrics, threshold, and anomaly detection results.
    output = {
        "config": {
            "train_size": config.TRAIN_SIZE,
            "test_size": config.TEST_SIZE,
            "injection_frac": config.INJECTION_FRAC,
            "bottleneck_dim": config.BOTTLENECK_DIM,
            "epochs": config.EPOCHS,
        },
        "metrics": {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
        },
        "threshold": float(threshold),
    }
    out_path = Path("results") / "pipeline_outputs.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to {out_path}")
    
    print("\n" + "█" * 70)
    print("█ PIPELINE COMPLETE ✓")
    print("█" * 70 + "\n")


def main():
    """Run the complete anomaly detection pipeline."""
    print("\n" + "█" * 70)
    print("█ EDUCATIONAL ANOMALY DETECTION PIPELINE (UNIFIED)")
    print("█" * 70)
    
    # Load and preprocess data
    X_train_scaled, X_test_scaled, feature_names, scaler, test_ids = load_and_preprocess_data(
        data_dir="data",
        results_dir="results"
    )
    
    # Train and evaluate model
    train_and_evaluate_model(X_train_scaled, X_test_scaled, feature_names, scaler, test_ids)


if __name__ == "__main__":
    main()
