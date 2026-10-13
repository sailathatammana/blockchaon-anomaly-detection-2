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
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
)

# Set random seeds FIRST, before any TensorFlow imports
os.environ["PYTHONHASHSEED"] = "41"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import tensorflow as tf

# Import project modules
import config
from data_loader import EducationalDataLoader
from preprocessing import EducationalDataPreprocessor
from autoencoder import (
    train_detector,
    predict_anomalies,
    score_anomalies,
    optimize_threshold,
    visualize_training_metrics,
)
from poa_blockchain import PoABlockchain


# Set seeds after imports
random.seed(config.SEED)
np.random.seed(config.SEED)
tf.random.set_seed(config.SEED)


def load_and_preprocess_data(
    data_dir: str = "data",
    results_dir: str = "results",
) -> Tuple:
    """Load and preprocess data for anomaly detection."""

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

        merged_df.to_csv(
            results_path / "merged_dataset.csv",
            index=False,
        )
        print("✓ Merged dataset saved")

        # Preprocess data
        print("\n" + "█" * 70)
        print("█ STEP 2: DATA PREPROCESSING")
        print("█" * 70)

        preprocessor = EducationalDataPreprocessor()

        # Create student performance profiles
        profiles = preprocessor.create_student_profiles(merged_df)

        print("\n✓ Student profiles created")
        print(f"  Shape: {profiles.shape}")
        print(f"  Students: {profiles.shape[0]}")
        print(f"  Features: {profiles.shape[1]}")

        profiles.to_csv(
            results_path / "student_profiles.csv",
            index=False,
        )
        print("✓ Profiles saved")

        # Extract student IDs before preparing for autoencoder
        student_ids = (
            profiles["id_student"].values
            if "id_student" in profiles.columns
            else None
        )

        # Extract gender for subgroup fairness analysis
        genders = (
            profiles["gender"].values
            if "gender" in profiles.columns
            else None
        )

        # Prepare features for autoencoder
        X, feature_names = preprocessor.prepare_for_autoencoder(profiles)

        print("\n✓ Features prepared for autoencoder")
        print(f"  Input shape: {X.shape}")
        print(f"  Num features: {len(feature_names)}")

        # Save feature names
        with open(
            results_path / "feature_names.json",
            "w",
        ) as f:
            json.dump(feature_names, f, indent=2)

        print("✓ Feature names saved")

        # Split into train/validation/test
        print("\n" + "█" * 70)
        print("█ STEP 3: TRAIN/VALIDATION/TEST SPLIT")
        print("█" * 70)

        # First keep 25% for the final test set
        if student_ids is not None and genders is not None:

            (
                X_train_val,
                X_test,
                train_val_ids,
                test_ids,
                train_val_genders,
                test_genders,
            ) = train_test_split(
                X,
                student_ids,
                genders,
                test_size=config.TEST_SIZE,
                random_state=config.SEED,
            )

        else:

            X_train_val, X_test = train_test_split(
                X,
                test_size=config.TEST_SIZE,
                random_state=config.SEED,
            )

            train_val_ids = None
            test_ids = None
            train_val_genders = None
            test_genders = None

        # Split remaining 75% into 60% train and 15% validation
        val_fraction = (
            config.VAL_SIZE
            / (config.TRAIN_SIZE + config.VAL_SIZE)
        )

        if (
            train_val_ids is not None
            and train_val_genders is not None
        ):

            (
                X_train,
                X_val,
                train_ids,
                val_ids,
                train_genders,
                val_genders,
            ) = train_test_split(
                X_train_val,
                train_val_ids,
                train_val_genders,
                test_size=val_fraction,
                random_state=config.SEED,
            )

        else:

            X_train, X_val = train_test_split(
                X_train_val,
                test_size=val_fraction,
                random_state=config.SEED,
            )

            train_ids = None
            val_ids = None
            train_genders = None
            val_genders = None

        print(
            f"\n✓ Data split completed "
            f"({config.TRAIN_SIZE * 100:.0f}/"
            f"{config.VAL_SIZE * 100:.0f}/"
            f"{config.TEST_SIZE * 100:.0f})"
        )

        print(
            f"  Training set:   {X_train.shape[0]} samples "
            f"({X_train.shape[0] / len(X) * 100:.1f}%)"
        )

        print(
            f"  Validation set: {X_val.shape[0]} samples "
            f"({X_val.shape[0] / len(X) * 100:.1f}%)"
        )

        print(
            f"  Test set:       {X_test.shape[0]} samples "
            f"({X_test.shape[0] / len(X) * 100:.1f}%)"
        )

        # Normalize features
        print("\n" + "█" * 70)
        print("█ STEP 4: FEATURE NORMALIZATION")
        print("█" * 70)

        scaler, X_train_scaled = (
            preprocessor.normalize_features(X_train)
        )

        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        print("\n✓ Features normalized to [0, 1] range")
        print(
            f"  Training set shape:   "
            f"{X_train_scaled.shape}"
        )
        print(
            f"  Validation set shape: "
            f"{X_val_scaled.shape}"
        )
        print(
            f"  Test set shape:       "
            f"{X_test_scaled.shape}"
        )

        return (
            X_train_scaled,
            X_val_scaled,
            X_test_scaled,
            feature_names,
            scaler,
            val_ids,
            test_ids,
            test_genders,
        )

    except Exception as e:

        print("\n" + "█" * 70)
        print("█ DATA LOADING ERROR")
        print("█" * 70)
        print(f"\nError: {e}")

        import traceback
        traceback.print_exc()

        return (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def train_and_evaluate_model(
    X_train_scaled,
    X_val_scaled,
    X_test_scaled,
    feature_names,
    scaler,
    val_ids=None,
    test_ids=None,
    test_genders=None,
) -> None:
    """Train autoencoder, inject anomalies, evaluate performance,
    perform subgroup analysis, and log detections to blockchain.
    """

    if X_train_scaled is None:
        print("Data loading failed. Stopping.")
        return

    # Train autoencoder
    print("\n" + "█" * 70)
    print("█ STEP 5A: AUTOENCODER TRAINING")
    print("█" * 70)

    print(
        f"\nTraining on {len(X_train_scaled)} samples "
        f"(This may take a moment...)"
    )

    model, threshold, history = train_detector(
        X_train_scaled,
        contamination=config.INJECTION_FRAC,
        bottleneck_dim=config.BOTTLENECK_DIM,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        random_state=config.SEED,
    )

    print("\nTraining completed successfully!")

    # Visualize training metrics
    print("\n" + "█" * 70)
    print("█ STEP 5B: TRAINING METRICS VISUALIZATION")
    print("█" * 70)

    training_metrics = visualize_training_metrics(
        history,
        results_dir="results",
        filename_prefix="pipeline_training",
    )

    print("\n✓ Training metrics visualized and saved")

    with open(
        Path("results") / "training_metrics.json",
        "w",
    ) as f:
        json.dump(training_metrics, f, indent=2)

    print("✓ Training metrics saved to training_metrics.json")

    # Inject synthetic anomalies
    print("\n" + "█" * 70)
    print(
        f"█ STEP 6: ANOMALY INJECTION "
        f"({config.INJECTION_FRAC * 100:.0f}% Gaussian Noise)"
    )
    print("█" * 70)

    # Validation anomalies
    np.random.seed(config.SEED)

    n_val = len(X_val_scaled)
    n_val_anomalies = int(
        n_val * config.INJECTION_FRAC
    )

    y_val = np.ones(n_val)

    val_anomaly_indices = np.random.choice(
        n_val,
        size=n_val_anomalies,
        replace=False,
    )

    y_val[val_anomaly_indices] = -1

    X_val_anomaly = X_val_scaled.copy()

    val_noise = np.random.normal(
        loc=0.0,
        scale=config.NOISE_SCALE,
        size=(
            n_val_anomalies,
            X_val_scaled.shape[1],
        ),
    )

    X_val_anomaly[val_anomaly_indices] += val_noise

    print(
        f"✓ Successfully injected "
        f"{n_val_anomalies} anomalies "
        f"into the validation set."
    )

    # Test anomalies generated separately
    np.random.seed(config.SEED + 1)

    n_test = len(X_test_scaled)

    n_test_anomalies = int(
        n_test * config.INJECTION_FRAC
    )

    y_true = np.ones(n_test)

    anomaly_indices = np.random.choice(
        n_test,
        size=n_test_anomalies,
        replace=False,
    )

    y_true[anomaly_indices] = -1

    X_test_anomaly = X_test_scaled.copy()

    test_noise = np.random.normal(
        loc=0.0,
        scale=config.NOISE_SCALE,
        size=(
            n_test_anomalies,
            X_test_scaled.shape[1],
        ),
    )

    X_test_anomaly[anomaly_indices] += test_noise

    print(
        f"✓ Successfully injected "
        f"{n_test_anomalies} anomalies "
        f"into the test set."
    )

    # Threshold optimization
    print("\n" + "█" * 70)
    print("█ STEP 7: THRESHOLD OPTIMIZATION")
    print("█" * 70)

    if config.OPTIMIZE_THRESHOLD:

        print(
            f"\nOptimizing threshold for "
            f"{config.THRESHOLD_METRIC.upper()} "
            f"balance using validation data..."
        )

        optimal_threshold, best_score = optimize_threshold(
            model,
            X_val_anomaly,
            y_val,
            metric=config.THRESHOLD_METRIC,
        )

        print(
            f"✓ Optimal threshold found: "
            f"{optimal_threshold:.5f}"
        )

        print(
            f"  Validation score: "
            f"{best_score:.4f}"
        )

    else:

        print("\nUsing default threshold from training...")

        optimal_threshold = threshold
        best_score = None

        print(
            f"✓ Default threshold: "
            f"{optimal_threshold:.5f}"
        )

    # Evaluate model
    print("\n" + "█" * 70)
    print("█ STEP 8: MODEL EVALUATION")
    print("█" * 70)

    test_scores = score_anomalies(
        model,
        X_test_anomaly,
    )

    test_preds = predict_anomalies(
        model,
        X_test_anomaly,
        optimal_threshold,
    )

    accuracy = accuracy_score(
        y_true,
        test_preds,
    )

    precision = precision_score(
        y_true,
        test_preds,
        pos_label=-1,
    )

    recall = recall_score(
        y_true,
        test_preds,
        pos_label=-1,
    )

    f1 = f1_score(
        y_true,
        test_preds,
        pos_label=-1,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        test_preds,
        labels=[1, -1],
    ).ravel()

    # ROC AUC
    y_true_binary = (
        y_true == -1
    ).astype(int)

    roc_auc = roc_auc_score(
        y_true_binary,
        test_scores,
    )

    print("\n" + "=" * 50)
    print("DETECTION RESULTS:")
    print("=" * 50)

    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")

    print(
        f"Detected:  "
        f"{int(np.sum(test_preds == -1))}/"
        f"{len(test_preds)}"
    )

    print("\n--- Detection Breakdown ---")

    print(
        "True Positives "
        "(Correctly caught injected anomalies):  "
        f"{tp}"
    )

    print(
        "False Positives "
        "(Normal profiles flagged as anomaly):  "
        f"{fp}"
    )

    print(
        "False Negatives "
        "(Missed injected anomalies):           "
        f"{fn}"
    )

    print(
        "True Negatives "
        "(Correctly labeled normal profiles):    "
        f"{tn}"
    )

    # ==========================================================
    # GENDER SUBGROUP ERROR ANALYSIS
    # ==========================================================

    subgroup_results = {}

    if test_genders is not None:

        print("\n" + "=" * 50)
        print("GENDER SUBGROUP ERROR ANALYSIS")
        print("=" * 50)

        for gender in np.unique(test_genders):

            mask = test_genders == gender

            y_group = y_true[mask]
            pred_group = test_preds[mask]

            g_tn, g_fp, g_fn, g_tp = confusion_matrix(
                y_group,
                pred_group,
                labels=[1, -1],
            ).ravel()

            false_positive_rate = (
                g_fp / (g_fp + g_tn)
                if (g_fp + g_tn) > 0
                else 0.0
            )

            false_negative_rate = (
                g_fn / (g_fn + g_tp)
                if (g_fn + g_tp) > 0
                else 0.0
            )

            print(f"\nGender: {gender}")
            print(
                f"  Samples:             "
                f"{int(np.sum(mask))}"
            )
            print(
                f"  True Positives:      "
                f"{int(g_tp)}"
            )
            print(
                f"  False Positives:     "
                f"{int(g_fp)}"
            )
            print(
                f"  False Negatives:     "
                f"{int(g_fn)}"
            )
            print(
                f"  True Negatives:      "
                f"{int(g_tn)}"
            )
            print(
                f"  False Positive Rate: "
                f"{false_positive_rate:.4f}"
            )
            print(
                f"  False Negative Rate: "
                f"{false_negative_rate:.4f}"
            )

            subgroup_results[str(gender)] = {
                "samples": int(np.sum(mask)),
                "tp": int(g_tp),
                "fp": int(g_fp),
                "fn": int(g_fn),
                "tn": int(g_tn),
                "false_positive_rate": float(
                    false_positive_rate
                ),
                "false_negative_rate": float(
                    false_negative_rate
                ),
            }

    else:

        print(
            "\nGender subgroup analysis skipped: "
            "gender information unavailable."
        )

    # Log to permissioned PoA blockchain
    print("\n" + "█" * 70)
    print(
        "█ STEP 9: PERMISSIONED PoA "
        "BLOCKCHAIN LOGGING"
    )
    print("█" * 70)

    print(
        "\nLogging detected anomalies to "
        "permissioned PoA blockchain..."
    )

    poa_blockchain = PoABlockchain(
        ledger_dir="results/poa_ledgers"
    )

    detected_anomaly_indices = [
        int(i)
        for i, p in enumerate(test_preds)
        if p == -1
    ]

    for idx in detected_anomaly_indices:

        validator = (
            poa_blockchain.get_current_validator()
        )

        student_id = (
            int(test_ids[idx])
            if test_ids is not None
            else None
        )

        anomaly_data = {
            "sample_index": idx,
            "student_id": student_id,
            "score": float(test_scores[idx]),
            "detector": "autoencoder",
        }

        poa_blockchain.create_block(
            validator=validator,
            anomaly_data=anomaly_data,
        )

    print(
        f"✓ Logged "
        f"{len(detected_anomaly_indices)} anomalies "
        f"to permissioned PoA blockchain"
    )

    # Verify all validator ledgers
    is_valid = (
        poa_blockchain.verify_all_validators()
    )

    if is_valid:

        print(
            "✓ PoA blockchain verification passed"
        )

        print(
            "✓ 3/3 validator ledgers "
            "valid and identical"
        )

    else:

        print(
            "✗ PoA blockchain verification FAILED - "
            "possible tampering or replica "
            "inconsistency detected!"
        )

    # Save all results
    output = {
        "config": {
            "train_size": config.TRAIN_SIZE,
            "val_size": config.VAL_SIZE,
            "test_size": config.TEST_SIZE,
            "injection_frac": config.INJECTION_FRAC,
            "noise_scale": config.NOISE_SCALE,
            "bottleneck_dim": config.BOTTLENECK_DIM,
            "epochs": config.EPOCHS,
            "batch_size": config.BATCH_SIZE,
        },

        "metrics": {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        },

        "gender_subgroup_analysis": subgroup_results,

        "threshold": float(optimal_threshold),

        "validation_f1": (
            float(best_score)
            if best_score is not None
            else None
        ),

        "n_validation_anomalies": int(
            n_val_anomalies
        ),

        "n_test_anomalies": int(
            n_test_anomalies
        ),
    }

    out_path = (
        Path("results")
        / "pipeline_outputs.json"
    )

    with open(out_path, "w") as f:
        json.dump(
            output,
            f,
            indent=2,
        )

    print(
        f"\n✓ Results saved to {out_path}"
    )

    print("\n" + "█" * 70)
    print("█ PIPELINE COMPLETE ✓")
    print("█" * 70 + "\n")


def main():
    """Run the complete anomaly detection pipeline."""

    print("\n" + "█" * 70)
    print(
        "█ EDUCATIONAL ANOMALY "
        "DETECTION PIPELINE (UNIFIED)"
    )
    print("█" * 70)

    (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        feature_names,
        scaler,
        val_ids,
        test_ids,
        test_genders,
    ) = load_and_preprocess_data(
        data_dir="data",
        results_dir="results",
    )

    train_and_evaluate_model(
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        feature_names,
        scaler,
        val_ids,
        test_ids,
        test_genders,
    )


if __name__ == "__main__":
    main()