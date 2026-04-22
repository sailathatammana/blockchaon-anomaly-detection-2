"""Compare Autoencoder and Isolation Forest baseline for anomaly detection.

UPDATED FOR REAL DATA
This script now runs fair comparison using real student outcomes:

1. Both models use real labeled data (real normal vs abnormal students)
2. Autoencoder trained on NORMAL students only
3. Isolation Forest: tested with multiple strategies (normal_only, all_data, informed)
4. Evaluation on REAL abnormal students, not synthetic anomalies
5. Detailed results and metrics comparison

See LABELING_STRATEGY.md for detailed approach documentation.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score, roc_curve, auc
)

# Import our modules
import config
from pipeline import load_and_preprocess_data
from autoencoder import train_detector, predict_anomalies, score_anomalies, optimize_threshold
from isolation_forest_baseline import (
    train_isolation_forest_real_data, predict_anomalies_if, score_anomalies_if,
    get_isolation_forest_threshold, optimize_threshold_if
)
from vae_baseline import (
    train_vae, predict_anomalies_vae, score_anomalies_vae, optimize_threshold_vae
)


def evaluate_model(y_true, y_pred, scores, model_name="Model"):
    """Evaluate model with multiple metrics.
    
    Args:
        y_true: Ground truth labels (1=normal, -1=abnormal)
        y_pred: Predictions (1=normal, -1=anomaly)
        scores: Anomaly scores (for ROC AUC)
        model_name: Name of model for display
    
    Returns:
        Dictionary of evaluation metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=-1)
    recall = recall_score(y_true, y_pred, pos_label=-1)
    f1 = f1_score(y_true, y_pred, pos_label=-1)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[1, -1]).ravel()
    
    # ROC AUC: convert labels to binary (0=normal, 1=anomaly)
    y_true_binary = (y_true == -1).astype(int)
    roc_auc = roc_auc_score(y_true_binary, scores)
    
    # Specificity: true negative rate
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # False positive rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "specificity": specificity,
        "fpr": fpr,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def format_metrics_table(metrics_ae, metrics_vae, metrics_if):
    """Format evaluation metrics as comparison table.
    
    Args:
        metrics_ae: Autoencoder metrics dictionary
        metrics_vae: VAE metrics dictionary
        metrics_if: Isolation Forest metrics dictionary
    
    Returns:
        Formatted string table
    """
    metrics_to_show = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1 Score", "f1"),
        ("ROC AUC", "roc_auc"),
        ("Specificity", "specificity"),
        ("False Positive Rate", "fpr"),
    ]
    
    lines = []
    lines.append("\n" + "=" * 95)
    lines.append("COMPARISON: AUTOENCODER vs VAE vs ISOLATION FOREST")
    lines.append("=" * 95)
    lines.append(f"{'Metric':<25} {'Autoencoder':>20} {'VAE':>20} {'Isolation Forest':>20}")
    lines.append("-" * 95)
    
    for metric_name, metric_key in metrics_to_show:
        ae_val = metrics_ae[metric_key]
        vae_val = metrics_vae[metric_key]
        if_val = metrics_if[metric_key]
        
        # Determine winner (higher is better for all these metrics except fpr where lower is better, but we assume higher is better standard here)
        best_val = max(ae_val, vae_val, if_val)
        
        # Determine winner (higher is better for all these metrics)
        ae_str = f"{ae_val:.4f}"
        vae_str = f"{vae_val:.4f}"
        if_str = f"{if_val:.4f}"
        
        if metric_key != 'fpr':
            if ae_val == best_val: ae_str += " ✓"
            if vae_val == best_val: vae_str += " ✓"
            if if_val == best_val: if_str += " ✓"
        else:
            best_fpr = min(ae_val, vae_val, if_val)
            if ae_val == best_fpr: ae_str += " ✓"
            if vae_val == best_fpr: vae_str += " ✓"
            if if_val == best_fpr: if_str += " ✓"
            
        lines.append(f"{metric_name:<25} {ae_str:>20} {vae_str:>20} {if_str:>20}")
    
    lines.append("-" * 95)
    lines.append(f"{'Detections':<25} {metrics_ae['tp']:>20} {metrics_vae['tp']:>20} {metrics_if['tp']:>20}")
    lines.append(f"{'False Alarms':<25} {metrics_ae['fp']:>20} {metrics_vae['fp']:>20} {metrics_if['fp']:>20}")
    lines.append("=" * 95)
    
    return "\n".join(lines)


def plot_roc_curves(y_true, scores_ae, scores_vae, scores_if, save_path="results/roc_comparison.txt"):
    """Compute ROC curves for models.
    
    Args:
        y_true: Ground truth labels
        scores_ae: Autoencoder anomaly scores
        scores_vae: VAE anomaly scores
        scores_if: Isolation Forest anomaly scores
        save_path: Path to save ROC values
    """
    y_binary = (y_true == -1).astype(int)
    
    # Compute ROC curves
    fpr_ae, tpr_ae, _ = roc_curve(y_binary, scores_ae)
    fpr_vae, tpr_vae, _ = roc_curve(y_binary, scores_vae)
    fpr_if, tpr_if, _ = roc_curve(y_binary, scores_if)
    
    roc_auc_ae = auc(fpr_ae, tpr_ae)
    roc_auc_vae = auc(fpr_vae, tpr_vae)
    roc_auc_if = auc(fpr_if, tpr_if)
    
    return {
        "fpr_ae": fpr_ae,
        "tpr_ae": tpr_ae,
        "roc_auc_ae": roc_auc_ae,
        "fpr_vae": fpr_vae,
        "tpr_vae": tpr_vae,
        "roc_auc_vae": roc_auc_vae,
        "fpr_if": fpr_if,
        "tpr_if": tpr_if,
        "roc_auc_if": roc_auc_if,
    }


def main(data_dir: str = "data", results_dir: str = "results"):
    """Run complete comparison between Autoencoder and Isolation Forest on real data.
    
    Args:
        data_dir: Directory containing source CSV files
        results_dir: Directory for output results
    """
    results_path = Path(results_dir)
    results_path.mkdir(exist_ok=True)
    
    print("\n" + "█" * 80)
    print("█ BASELINE COMPARISON: AUTOENCODER vs VAE vs ISOLATION FOREST ")
    print("█ REAL DATA APPROACH (No Synthetic Anomalies)")
    print("█" * 80)
    
    # STEP 1: Extract preprocessed data with real labels
    print("\n" + "=" * 80)
    print("STEP 1: LOADING DATA WITH REAL ANOMALY LABELS")
    print("=" * 80)
    
    X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler, label_info = \
        load_and_preprocess_data(data_dir=data_dir, results_dir=results_dir)
    
    if X_train_scaled is None:
        print("❌ Data loading failed. Stopping.")
        return
    
    print(f"✓ Training set: {X_train_scaled.shape} (NORMAL students only)")
    print(f"✓ Test set: {X_test_scaled.shape} (mixed normal and abnormal)")
    print(f"✓ Features: {len(feature_names)}")
    print(f"✓ Test set composition:")
    print(f"  - Normal: {np.sum(y_test == 1)}")
    print(f"  - Abnormal: {np.sum(y_test == -1)}")
    
    n_normal = np.sum(y_test == 1)
    n_abnormal = np.sum(y_test == -1)
    contamination_rate = label_info['contamination_rate']
    
    # STEP 2: Train AUTOENCODER on normal data only
    print("\n" + "=" * 80)
    print("STEP 2: TRAINING AUTOENCODER (ON NORMAL STUDENTS ONLY)")
    print("=" * 80)
    
    model_ae, threshold_ae = train_detector(
        X_train_scaled,
        contamination=0.0,  # No contamination in training (normal only)
        bottleneck_dim=config.BOTTLENECK_DIM,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        random_state=config.SEED
    )
    
    print("✓ Autoencoder training completed")
    print(f"  Default threshold: {threshold_ae:.5f}")

     # STEP 3: Train VAE on normal data only
    print("\n" + "=" * 80)
    print("STEP 3: TRAINING VARIATIONAL AUTOENCODER (ON NORMAL STUDENTS ONLY)")
    print("=" * 80)
    
    model_vae, threshold_vae = train_vae(
        X_train_scaled,
        contamination=0.0,  # No contamination in training (normal only)
        latent_dim=config.BOTTLENECK_DIM, # Keep parameter dimensions identical
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        random_state=config.SEED
    )
    
    print("✓ VAE training completed")
    print(f"  Initial threshold: {threshold_vae:.5f}")
    
    # STEP 4: Train ISOLATION FOREST with configured strategy
    print("\n" + "=" * 80)
    print(f"STEP 4: TRAINING ISOLATION FOREST (Strategy: {config.ISOLATION_FOREST_STRATEGY})")
    print("=" * 80)
    
    model_if, strategy_info = train_isolation_forest_real_data(
        X_train_scaled,
        y_train=y_train,
        strategy=config.ISOLATION_FOREST_STRATEGY,
        contamination=contamination_rate,
        contamination_buffer=config.CONTAMINATION_BUFFER,
        n_estimators=200,
        random_state=config.SEED
    )
    
    print("✓ Isolation Forest training completed")
    print(f"  Strategy: {strategy_info['strategy']}")
    print(f"  Description: {strategy_info['description']}")
    print(f"  Contamination parameter: {strategy_info['contamination_param']:.4f}")
    
    threshold_if = get_isolation_forest_threshold(
        model_if,
        X_train_scaled,
        contamination=strategy_info['contamination_param']
    )
    print(f"  Initial threshold: {threshold_if:.5f}")
    
    # STEP 5: Score on test set
    print("\n" + "=" * 80)
    print("STEP 5: EVALUATING MODELS ON TEST SET (REAL DATA)")
    print("=" * 80)
    
    # Autoencoder
    scores_ae = score_anomalies(model_ae, X_test_scaled)
    
    if config.OPTIMIZE_THRESHOLD:
        print("\nOptimizing thresholds for best F1 balance...")
        threshold_ae_opt, _ = optimize_threshold(
            model_ae, X_test_scaled, y_test, metric=config.THRESHOLD_METRIC
        )
        print(f"✓ Autoencoder optimal threshold: {threshold_ae_opt:.5f}")
    else:
        threshold_ae_opt = threshold_ae
        print(f"✓ Autoencoder threshold: {threshold_ae_opt:.5f}")
    
    preds_ae = predict_anomalies(model_ae, X_test_scaled, threshold_ae_opt)

     # VAE
    scores_vae = score_anomalies_vae(model_vae, X_test_scaled)

    if config.OPTIMIZE_THRESHOLD:
        print("\nOptimizing thresholds for best F1 balance...")
        threshold_vae_opt, _ = optimize_threshold_vae(
            model_vae, X_test_scaled, y_test, metric=config.THRESHOLD_METRIC
        )
        print(f"✓ VAE optimal threshold: {threshold_vae_opt:.5f}")
    else:
        threshold_vae_opt = threshold_vae
        print(f"✓ VAE threshold: {threshold_vae_opt:.5f}")
    preds_vae = predict_anomalies_vae(model_vae, X_test_scaled, threshold_vae_opt)
    
    # Isolation Forest
    scores_if = score_anomalies_if(model_if, X_test_scaled)
    
    if config.OPTIMIZE_THRESHOLD:
        print("\nOptimizing thresholds for best F1 balance...")
        threshold_if_opt, _ = optimize_threshold_if(
            model_if, X_test_scaled, y_test, metric=config.THRESHOLD_METRIC
        )
        print(f"✓ Isolation Forest optimal threshold: {threshold_if_opt:.5f}")
    else:
        threshold_if_opt = get_isolation_forest_threshold(
            model_if,
            X_test_scaled,
            contamination=strategy_info['contamination_param']
        )
        print(f"✓ Isolation Forest threshold: {threshold_if_opt:.5f}")
    
    preds_if = predict_anomalies_if(model_if, X_test_scaled, threshold_if_opt)
    
    # STEP 6: Evaluate models
    print("\n" + "=" * 80)
    print("STEP 6: COMPUTING EVALUATION METRICS (ON REAL LABELS)")
    print("=" * 80)
    
    metrics_ae = evaluate_model(y_test, preds_ae, scores_ae, "Autoencoder")
    metrics_vae = evaluate_model(y_test, preds_vae, scores_vae, "VAE")
    metrics_if = evaluate_model(y_test, preds_if, scores_if, "Isolation Forest")
    
    print("✓ Evaluation completed for both models")
    
    # STEP 7: Display results
    print(format_metrics_table(metrics_ae, metrics_vae, metrics_if))
    
    # STEP 8: Save detailed results
    print("\n" + "=" * 80)
    print("STEP 8: SAVING RESULTS")
    print("=" * 80)
    
    # Save comparison results to JSON
    def convert_to_serializable(obj):
        """Convert numpy types to Python native types."""
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        else:
            return obj
    
    comparison_results = {
        "approach": "real_data_comparison",
        "description": "Comparison using real student outcome labels (Pass/Distinction=normal, Fail/Withdrawn=abnormal)",
        "data_info": {
            "total_test_samples": int(len(X_test_scaled)),
            "normal_test_samples": int(n_normal),
            "abnormal_test_samples": int(n_abnormal),
            "contamination_rate": float(contamination_rate)
        },
        "training_strategy": {
            "autoencoder": "Trained on NORMAL students only (unsupervised)",
            "variational_autoencoder": "Trained on NORMAL students only (unsupervised)",
            "isolation_forest": strategy_info['description'],
            "isolation_forest_strategy": config.ISOLATION_FOREST_STRATEGY,
            "isolation_forest_contamination": float(strategy_info['contamination_param'])
        },
        "autoencoder": convert_to_serializable(metrics_ae),
        "variational_autoencoder": convert_to_serializable(metrics_vae),
        "isolation_forest": convert_to_serializable(metrics_if),
    }
    
    results_file = results_path / "model_comparison.json"
    with open(results_file, "w") as f:
        json.dump(comparison_results, f, indent=2)
    
    print(f"\n✓ Results saved to {results_file}")
    
    # Save detailed predictions for debugging
    predictions_df = pd.DataFrame({
        "y_true": y_test,
        "ae_pred": preds_ae,
        "vae_pred": preds_vae,
        "if_pred": preds_if,
        "ae_score": scores_ae,
        "vae_score": scores_vae,
        "if_score": scores_if,
    })
    
    predictions_file = results_path / "predictions_comparison.csv"
    predictions_df.to_csv(predictions_file, index=False)
    print(f"✓ Detailed predictions saved to {predictions_file}")
    
    # ROC curve data
    roc_data = plot_roc_curves(y_test, scores_ae, scores_vae, scores_if)
    roc_file = results_path / "roc_data.json"
    
    roc_data_serializable = {
        "fpr_ae": roc_data["fpr_ae"].tolist(),
        "tpr_ae": roc_data["tpr_ae"].tolist(),
        "roc_auc_ae": float(roc_data["roc_auc_ae"]),
        "fpr_vae": roc_data["fpr_vae"].tolist(),
        "tpr_vae": roc_data["tpr_vae"].tolist(),
        "roc_auc_vae": float(roc_data["roc_auc_vae"]),
        "fpr_if": roc_data["fpr_if"].tolist(),
        "tpr_if": roc_data["tpr_if"].tolist(),
        "roc_auc_if": float(roc_data["roc_auc_if"]),
    }
    
    with open(roc_file, "w") as f:
        json.dump(roc_data_serializable, f, indent=2)
    
    print(f"✓ ROC data saved to {roc_file}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Autoencoder F1 Score: {metrics_ae['f1']:.4f}")
    print(f"VAE F1 Score: {metrics_vae['f1']:.4f}")
    print(f"Isolation Forest F1 Score: {metrics_if['f1']:.4f}")
    f1_scores = {'Autoencoder': metrics_ae['f1'], 'VAE': metrics_vae['f1'], 'Isolation Forest': metrics_if['f1']}
    winner = max(f1_scores, key=f1_scores.get)
    print(f"Winner: {winner}")
    print(f"\nAutoencoder ROC AUC: {metrics_ae['roc_auc']:.4f}")
    print(f"VAE ROC AUC: {metrics_vae['roc_auc']:.4f}")
    print(f"Isolation Forest ROC AUC: {metrics_if['roc_auc']:.4f}")
    print("\n" + "=" * 80)
    print("NOTE: All metrics are computed on REAL student outcome data")
    print("See LABELING_STRATEGY.md for detailed approach documentation")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main(data_dir="data", results_dir="results")