"""Ablation study: Systematically test autoencoder component variations.

This script tests how different architectural and hyperparameter choices impact
the model's anomaly detection performance. Each component is tested independently
while keeping others constant at baseline values.

Components tested:
1. Bottleneck dimension: Controls compression level (2, 4, 8, 16, 32)
2. Network depth: Number of hidden layers (1, 2, 3, 4)
3. Dropout rate: Regularization strength (0.0, 0.1, 0.2, 0.3)
4. Activation function: Non-linearity type (relu, sigmoid, tanh)
5. Feature subset: Percentage of features by variance (100%, 50%, 25%)
6. Batch size: Training mini-batch size (8, 16, 32)
7. Epoch budget: Maximum training epochs (40, 80, 120)

Threshold policy:
- Tune threshold on a dedicated validation split
- Report final metrics on a separate holdout split

Baseline configuration:
- Bottleneck: 4
- Depth: 3 (128→64→32)
- Dropout: 0.1
- Activation: relu
- Features: 100%
- Batch size: 16, Epochs: 80
"""

import os
import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Set seeds FIRST
os.environ["PYTHONHASHSEED"] = "41"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score
)

# Set seeds after imports
random.seed(41)
np.random.seed(41)
tf.random.set_seed(41)

# Import project modules
import config
from data_loader import EducationalDataLoader
from preprocessing import EducationalDataPreprocessor
from autoencoder import optimize_threshold  # Import the real function


# ==================== AUTOENCODER BUILDERS ====================

def build_autoencoder_variable_depth(
    input_dim: int,
    bottleneck_dim: int = 4,
    depth: int = 3,
    dropout_rate: float = 0.1,
    activation: str = "relu"
) -> keras.Model:
    """Build autoencoder with configurable depth and regularization.
    
    Args:
        input_dim: Number of input features
        bottleneck_dim: Compression dimension
        depth: Number of hidden layers (1-4)
        dropout_rate: Dropout probability
        activation: Activation function name (relu, sigmoid, tanh)
    
    Returns:
        Compiled Keras Model
    """
    inp = keras.Input(shape=(input_dim,))
    
    # Define encoder layer dimensions based on depth
    # Always compress progressively towards bottleneck
    if depth == 1:
        encoder_dims = [64]
    elif depth == 2:
        encoder_dims = [128, 64]
    elif depth == 3:
        encoder_dims = [128, 64, 32]
    elif depth == 4:
        encoder_dims = [256, 128, 64, 32]
    else:
        raise ValueError(f"Depth must be 1-4, got {depth}")
    
    # ENCODER: Compress to bottleneck
    x = inp
    for dim in encoder_dims:
        x = layers.Dense(dim, activation=activation)(x)
    
    # Bottleneck layer
    bottleneck = layers.Dense(bottleneck_dim, activation=activation)(x)
    x = layers.Dropout(dropout_rate)(bottleneck)
    
    # DECODER: Expand back (mirror encoder)
    decoder_dims = list(reversed(encoder_dims))
    for dim in decoder_dims:
        x = layers.Dense(dim, activation=activation)(x)
        x = layers.Dropout(dropout_rate)(x)
    
    # Output layer
    out = layers.Dense(input_dim, activation="linear")(x)
    
    model = keras.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss="mse"
    )
    return model


def train_variant(
    X_train: np.ndarray,
    X_threshold: np.ndarray,
    y_threshold: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    bottleneck_dim: int = 4,
    depth: int = 3,
    dropout_rate: float = 0.1,
    activation: str = "relu",
    batch_size: int = 16,
    epochs: int = 80,
    threshold_metric: str = "f1",
    random_state: int = 41,
) -> Dict[str, Any]:
    """Train a variant and evaluate it.
    
    Args:
        X_train: Training data (normal samples only)
        X_threshold: Threshold-tuning split (normal + abnormal)
        y_threshold: Labels for threshold-tuning split
        X_test: Holdout test split (normal + abnormal)
        y_test: Labels for holdout test split (1=normal, -1=abnormal)
        bottleneck_dim: Compression dimension
        depth: Network depth
        dropout_rate: Dropout probability
        activation: Activation function
        batch_size: Batch size for training
        epochs: Maximum epochs
        threshold_metric: Metric used to optimize threshold on threshold split
    
    Returns:
        Dictionary with metrics and configuration
    """
    np.random.seed(random_state)
    keras.utils.set_random_seed(random_state)

    # Build model with specified configuration
    input_dim = X_train.shape[1]
    model = build_autoencoder_variable_depth(
        input_dim=input_dim,
        bottleneck_dim=bottleneck_dim,
        depth=depth,
        dropout_rate=dropout_rate,
        activation=activation
    )
    
    # Train with early stopping using internal validation split (same as original)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True,
            min_delta=1e-5
        )
    ]
    
    history = model.fit(
        X_train,
        X_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_split=0.15,  # Let Keras handle the 15% validation split
        callbacks=callbacks,
        verbose=0
    )
    
    # Optimize threshold on dedicated threshold-tuning split
    threshold, threshold_score = optimize_threshold(
        model, X_threshold, y_threshold, metric=threshold_metric
    )

    # Evaluate only on holdout test split
    X_test_recon = model.predict(X_test, verbose=0)
    test_errors = np.mean((X_test - X_test_recon) ** 2, axis=1)
    
    # Final predictions with optimized threshold
    predictions = np.where(test_errors > threshold, -1, 1)
    
    # Compute metrics
    y_test_binary = (y_test == -1).astype(int)
    if len(np.unique(y_test_binary)) > 1:
        roc_auc = float(roc_auc_score(y_test_binary, test_errors))
    else:
        roc_auc = 0.5

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1": float(f1_score(y_test, predictions, pos_label=-1, zero_division=0)),
        "precision": float(precision_score(y_test, predictions, pos_label=-1, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, pos_label=-1, zero_division=0)),
        "roc_auc": roc_auc,
        "threshold": float(threshold),
        "threshold_metric": threshold_metric,
        "threshold_metric_score": float(threshold_score),
    }
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[1, -1]).ravel()
    metrics.update({
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
    })
    
    # Store epochs actually run (before early stopping)
    metrics["epochs_run"] = len(history.history["loss"])
    metrics["final_train_loss"] = float(history.history["loss"][-1])
    metrics["final_val_loss"] = float(history.history["val_loss"][-1])
    
    return metrics


# ==================== ABLATION STUDY RUNNER ====================

def run_ablation_study(
    data_dir: str = "data",
    results_dir: str = "results",
    output_file: str = "ablation_study.json"
) -> Path:
    """Run complete ablation study testing all component variations.
    
    Args:
        data_dir: Directory with input data
        results_dir: Directory for output results
        output_file: Filename for JSON results
    
    Returns:
        Path to results JSON file
    """
    results_path = Path(results_dir)
    results_path.mkdir(exist_ok=True)
    output_path = results_path / output_file
    
    print("\n" + "█" * 80)
    print("█ ABLATION STUDY: Autoencoder Component Analysis")
    print("█" * 80)
    
    # ==================== LOAD & PREPROCESS DATA ====================
    print("\n[1/10] Loading and preprocessing data...")
    
    data_path = Path(data_dir)
    loader = EducationalDataLoader(data_dir=str(data_path))
    loader.load_all_datasets()
    merged_df = loader.create_merged_dataset()
    
    preprocessor = EducationalDataPreprocessor()
    
    # Create student profiles
    profiles = preprocessor.create_student_profiles(merged_df)
    
    # Create labels from real outcomes
    y, _ = preprocessor.create_anomaly_labels(
        profiles,
        normal_labels=config.NORMAL_OUTCOME_LABELS,
        abnormal_labels=config.ABNORMAL_OUTCOME_LABELS
    )
    
    # Prepare features for autoencoder
    X, feature_names = preprocessor.prepare_for_autoencoder(profiles)
    
    # Split data using EXACT same method as original pipeline
    X_train, X_test, _, y_test, _, _ = preprocessor.split_normal_abnormal(
        X, y,
        normal_train_fraction=config.NORMAL_TRAIN_FRACTION,
        random_state=config.SEED
    )
    
    # Normalize features AFTER splitting (fit scaler on train data only)
    scaler, X_train_normalized = preprocessor.normalize_features(
        pd.DataFrame(X_train, columns=feature_names)
    )
    X_train = X_train_normalized
    
    # Apply same scaler to test data (prevent data leakage)
    X_test_normalized = scaler.transform(
        pd.DataFrame(X_test, columns=feature_names)
    )
    X_test = X_test_normalized
    
    print(f"  ✓ Training samples: {len(X_train)} (normal)")
    print(f"  ✓ Test samples: {len(X_test)} ({np.sum(y_test==1)} normal, {np.sum(y_test==-1)} abnormal)")
    print(f"  ✓ Features: {X_train.shape[1]}")
    
    # Dedicated split for threshold tuning vs final evaluation
    print("\n[2/10] Splitting threshold-tuning and holdout evaluation sets...")
    threshold_tune_fraction = 0.30
    unique_labels, label_counts = np.unique(y_test, return_counts=True)
    can_stratify = len(unique_labels) > 1 and np.all(label_counts >= 2)

    try:
        X_threshold, X_eval, y_threshold, y_eval = train_test_split(
            X_test,
            y_test,
            test_size=1.0 - threshold_tune_fraction,
            random_state=config.SEED,
            stratify=y_test if can_stratify else None
        )
    except ValueError:
        # Fallback when stratified split is not feasible with small class counts
        X_threshold, X_eval, y_threshold, y_eval = train_test_split(
            X_test,
            y_test,
            test_size=1.0 - threshold_tune_fraction,
            random_state=config.SEED,
            stratify=None
        )

    print(
        f"  [OK] Threshold-tuning set: {len(X_threshold)} "
        f"({np.sum(y_threshold==1)} normal, {np.sum(y_threshold==-1)} abnormal)"
    )
    print(
        f"  [OK] Holdout evaluation set: {len(X_eval)} "
        f"({np.sum(y_eval==1)} normal, {np.sum(y_eval==-1)} abnormal)"
    )

    # ==================== BASELINE ====================
    print("\n[3/10] Training baseline model (bottleneck=4, depth=3, dropout=0.1, relu)...")
    
    baseline_results = train_variant(
        X_train, X_threshold, y_threshold, X_eval, y_eval,
        bottleneck_dim=4,
        depth=3,
        dropout_rate=0.1,
        activation="relu",
        batch_size=16,
        epochs=80
    )
    baseline_results["config"] = {
        "bottleneck_dim": 4,
        "depth": 3,
        "dropout_rate": 0.1,
        "activation": "relu",
        "batch_size": 16,
        "epochs": 80,
        "feature_subset": 1.0,
    }
    print(f"  ✓ F1: {baseline_results['f1']:.4f}, ROC-AUC: {baseline_results['roc_auc']:.4f}")
    
    all_results = {
        "baseline": baseline_results,
        "variations": {},
        "timestamp": datetime.now().isoformat(),
        "data_summary": {
            "n_features": int(X_train.shape[1]),
            "n_train": int(len(X_train)),
            "n_threshold_tune": int(len(X_threshold)),
            "n_threshold_tune_abnormal": int(np.sum(y_threshold == -1)),
            "n_eval": int(len(X_eval)),
            "n_eval_abnormal": int(np.sum(y_eval == -1)),
        },
        "study_config": {
            "threshold_tune_fraction": threshold_tune_fraction,
            "threshold_metric": "f1",
            "random_state": int(config.SEED),
        },
    }
    
    # ==================== TEST BOTTLENECK DIMENSION ====================
    print("\n[4/10] Testing bottleneck dimensions [2, 4, 8, 16, 32]...")
    
    bottleneck_results = {}
    for bottleneck_dim in [2, 4, 8, 16, 32]:
        print(f"  Testing bottleneck_dim={bottleneck_dim}...", end=" ")
        results = train_variant(
            X_train, X_threshold, y_threshold, X_eval, y_eval,
            bottleneck_dim=bottleneck_dim,
            depth=3,
            dropout_rate=0.1,
            activation="relu",
            batch_size=16,
            epochs=80
        )
        results["config"] = {
            "bottleneck_dim": bottleneck_dim,
            "depth": 3,
            "dropout_rate": 0.1,
            "activation": "relu",
        }
        bottleneck_results[f"bottleneck_{bottleneck_dim}"] = results
        print(f"F1={results['f1']:.4f}")
    
    all_results["variations"]["bottleneck_dimension"] = bottleneck_results
    
    # ==================== TEST NETWORK DEPTH ====================
    print("\n[5/10] Testing network depths [1, 2, 3, 4]...")
    
    depth_results = {}
    for depth in [1, 2, 3, 4]:
        print(f"  Testing depth={depth}...", end=" ")
        results = train_variant(
            X_train, X_threshold, y_threshold, X_eval, y_eval,
            bottleneck_dim=4,
            depth=depth,
            dropout_rate=0.1,
            activation="relu",
            batch_size=16,
            epochs=80
        )
        results["config"] = {
            "bottleneck_dim": 4,
            "depth": depth,
            "dropout_rate": 0.1,
            "activation": "relu",
        }
        depth_results[f"depth_{depth}"] = results
        print(f"F1={results['f1']:.4f}")
    
    all_results["variations"]["network_depth"] = depth_results
    
    # ==================== TEST DROPOUT RATE ====================
    print("\n[6/10] Testing dropout rates [0.0, 0.1, 0.2, 0.3]...")
    
    dropout_results = {}
    for dropout_rate in [0.0, 0.1, 0.2, 0.3]:
        print(f"  Testing dropout_rate={dropout_rate}...", end=" ")
        results = train_variant(
            X_train, X_threshold, y_threshold, X_eval, y_eval,
            bottleneck_dim=4,
            depth=3,
            dropout_rate=dropout_rate,
            activation="relu",
            batch_size=16,
            epochs=80
        )
        results["config"] = {
            "bottleneck_dim": 4,
            "depth": 3,
            "dropout_rate": dropout_rate,
            "activation": "relu",
        }
        dropout_results[f"dropout_{dropout_rate:.1f}"] = results
        print(f"F1={results['f1']:.4f}")
    
    all_results["variations"]["dropout_rate"] = dropout_results
    
    # ==================== TEST ACTIVATION FUNCTIONS ====================
    print("\n[7/10] Testing activation functions [relu, sigmoid, tanh]...")
    
    activation_results = {}
    for activation in ["relu", "sigmoid", "tanh"]:
        print(f"  Testing activation={activation}...", end=" ")
        results = train_variant(
            X_train, X_threshold, y_threshold, X_eval, y_eval,
            bottleneck_dim=4,
            depth=3,
            dropout_rate=0.1,
            activation=activation,
            batch_size=16,
            epochs=80
        )
        results["config"] = {
            "bottleneck_dim": 4,
            "depth": 3,
            "dropout_rate": 0.1,
            "activation": activation,
        }
        activation_results[f"activation_{activation}"] = results
        print(f"F1={results['f1']:.4f}")
    
    all_results["variations"]["activation_function"] = activation_results
    
    # ==================== TEST FEATURE SUBSETS ====================
    print("\n[8/10] Testing feature subsets [100%, 50%, 25%]...")
    
    feature_results = {}
    for pct in [1.0, 0.5, 0.25]:
        print(f"  Testing feature_subset={int(pct*100)}%...", end=" ")
        
        # Select top features by variance (computed from training data)
        n_features = max(1, int(X_train.shape[1] * pct))
        feature_var = np.var(X_train, axis=0)
        feature_indices = np.argsort(feature_var)[-n_features:]
        
        X_train_subset = X_train[:, feature_indices]
        X_threshold_subset = X_threshold[:, feature_indices]
        X_eval_subset = X_eval[:, feature_indices]
        
        results = train_variant(
            X_train_subset, X_threshold_subset, y_threshold, X_eval_subset, y_eval,
            bottleneck_dim=4,
            depth=3,
            dropout_rate=0.1,
            activation="relu",
            batch_size=16,
            epochs=80
        )
        results["config"] = {
            "bottleneck_dim": 4,
            "depth": 3,
            "dropout_rate": 0.1,
            "activation": "relu",
            "n_features_used": n_features,
            "feature_subset_pct": float(pct),
        }
        feature_results[f"features_{int(pct*100)}pct"] = results
        print(f"F1={results['f1']:.4f}")
    
    all_results["variations"]["feature_subset"] = feature_results

    # ==================== TEST BATCH SIZE ====================
    print("\n[9/10] Testing batch sizes [8, 16, 32]...")

    batch_size_results = {}
    for batch_size in [8, 16, 32]:
        print(f"  Testing batch_size={batch_size}...", end=" ")
        results = train_variant(
            X_train, X_threshold, y_threshold, X_eval, y_eval,
            bottleneck_dim=4,
            depth=3,
            dropout_rate=0.1,
            activation="relu",
            batch_size=batch_size,
            epochs=80
        )
        results["config"] = {
            "bottleneck_dim": 4,
            "depth": 3,
            "dropout_rate": 0.1,
            "activation": "relu",
            "batch_size": batch_size,
            "epochs": 80,
        }
        batch_size_results[f"batch_size_{batch_size}"] = results
        print(f"F1={results['f1']:.4f}")

    all_results["variations"]["batch_size"] = batch_size_results

    # ==================== TEST EPOCH BUDGET ====================
    print("\n[10/10] Testing epoch budgets [40, 80, 120]...")

    epoch_results = {}
    for epochs in [40, 80, 120]:
        print(f"  Testing epochs={epochs}...", end=" ")
        results = train_variant(
            X_train, X_threshold, y_threshold, X_eval, y_eval,
            bottleneck_dim=4,
            depth=3,
            dropout_rate=0.1,
            activation="relu",
            batch_size=16,
            epochs=epochs
        )
        results["config"] = {
            "bottleneck_dim": 4,
            "depth": 3,
            "dropout_rate": 0.1,
            "activation": "relu",
            "batch_size": 16,
            "epochs": epochs,
        }
        epoch_results[f"epochs_{epochs}"] = results
        print(f"F1={results['f1']:.4f}")

    all_results["variations"]["epochs"] = epoch_results
    
    # ==================== SAVE RESULTS ====================
    print("\n" + "█" * 80)
    print("█ Saving results...")
    
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"✓ Results saved to {output_path}")
    
    # Print summary
    print("\n" + "█" * 80)
    print("█ SUMMARY")
    print("█" * 80)
    print(f"\nBaseline F1 Score: {baseline_results['f1']:.4f}")
    print(f"Baseline ROC-AUC: {baseline_results['roc_auc']:.4f}\n")
    
    # Find best variants
    print("Best variant by F1 in each category:")
    for category, variants in all_results["variations"].items():
        best_key = max(variants.keys(), key=lambda k: variants[k]["f1"])
        best_f1 = variants[best_key]["f1"]
        improvement = (best_f1 - baseline_results['f1']) / baseline_results['f1'] * 100
        print(f"  {category:25} {best_key:25} F1={best_f1:.4f} ({improvement:+.2f}%)")
    
    return output_path


if __name__ == "__main__":
    run_ablation_study()
