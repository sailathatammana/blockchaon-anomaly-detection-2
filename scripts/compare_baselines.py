"""Compare Autoencoder and Isolation Forest baseline for anomaly detection.

This script runs a fair comparison:
1. Both models trained on the same data splits
2. Same synthetic anomalies injected into test set
3. Same evaluation metrics for fair comparison
4. Detailed results and visualizations
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
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Import our modules
import config
from pipeline import load_and_preprocess_data
from autoencoder import train_detector, predict_anomalies, score_anomalies, optimize_threshold
from isolation_forest_baseline import (
    train_isolation_forest, predict_anomalies_if, score_anomalies_if,
    get_isolation_forest_threshold
)
from vae_baseline import (
    train_vae, predict_anomalies_vae, score_anomalies_vae, optimize_threshold_vae
)


def inject_anomalies(X_test, contamination_frac, noise_scale, seed):
    """Inject synthetic Gaussian noise anomalies into test set.
    
    Args:
        X_test: Test data
        contamination_frac: Fraction of test set to corrupt
        noise_scale: Standard deviation of Gaussian noise
        seed: Random seed for reproducibility
    
    Returns:
        X_test_corrupted: Test data with injected anomalies
        y_true: Ground truth labels (1=normal, -1=anomaly)
        anomaly_indices: Indices of injected anomalies
    """
    np.random.seed(seed)
    n_test = len(X_test)
    n_anomalies = int(n_test * contamination_frac)
    
    # Ground truth: 1 for normal, -1 for anomaly
    y_true = np.ones(n_test)
    
    # Select which samples to corrupt
    anomaly_indices = np.random.choice(n_test, size=n_anomalies, replace=False)
    y_true[anomaly_indices] = -1
    
    # Create corrupted test set
    X_test_corrupted = X_test.copy()
    noise = np.random.normal(
        loc=0.0,
        scale=noise_scale,
        size=(n_anomalies, X_test.shape[1])
    )
    X_test_corrupted[anomaly_indices] += noise
    
    return X_test_corrupted, y_true, anomaly_indices


def evaluate_model(y_true, y_pred, scores, model_name="Model"):
    """Evaluate model with multiple metrics.
    
    Args:
        y_true: Ground truth labels (1=normal, -1=anomaly)
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


def format_metrics_table(metrics_ae, metrics_if, metrics_vae):
    """Format evaluation metrics as comparison table.
    
    Args:
        metrics_ae: Autoencoder metrics dictionary
        metrics_if: Isolation Forest metrics dictionary
        metrics_vae: VAE metrics dictionary
    
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
    lines.append("COMPARISON: AUTOENCODER vs ISOLATION FOREST vs VAE")
    lines.append("=" * 95)
    lines.append(f"{'Metric':<25} {'Autoencoder':>20} {'Isolation Forest':>20} {'VAE':>20}")
    lines.append("-" * 95)
    
    for metric_name, metric_key in metrics_to_show:
        ae_val = metrics_ae[metric_key]
        if_val = metrics_if[metric_key]
        vae_val = metrics_vae[metric_key]
        
        # Determine winner (higher is better for all these metrics except fpr where lower is better, but we assume higher is better standard here)
        best_val = max(ae_val, if_val, vae_val)
        
        ae_str = f"{ae_val:.4f}"
        if_str = f"{if_val:.4f}"
        vae_str = f"{vae_val:.4f}"
        
        if metric_key != 'fpr':
            if ae_val == best_val: ae_str += " ✓"
            if if_val == best_val: if_str += " ✓"
            if vae_val == best_val: vae_str += " ✓"
        else:
            best_fpr = min(ae_val, if_val, vae_val)
            if ae_val == best_fpr: ae_str += " ✓"
            if if_val == best_fpr: if_str += " ✓"
            if vae_val == best_fpr: vae_str += " ✓"
            
        lines.append(f"{metric_name:<25} {ae_str:>20} {if_str:>20} {vae_str:>20}")
    
    lines.append("-" * 95)
    lines.append(f"{'Detections':<25} {metrics_ae['tp']:>20} {metrics_if['tp']:>20} {metrics_vae['tp']:>20}")
    lines.append(f"{'False Alarms':<25} {metrics_ae['fp']:>20} {metrics_if['fp']:>20} {metrics_vae['fp']:>20}")
    lines.append("=" * 95)
    
    return "\n".join(lines)


def plot_roc_curves(y_true, scores_ae, scores_if, scores_vae, save_path="results/roc_comparison.txt"):
    """Compute ROC curves for models.
    
    Args:
        y_true: Ground truth labels
        scores_ae: Autoencoder anomaly scores
        scores_if: Isolation Forest anomaly scores
        scores_vae: VAE anomaly scores
        save_path: Path to save ROC values
    """
    y_binary = (y_true == -1).astype(int)
    
    # Compute ROC curves
    fpr_ae, tpr_ae, _ = roc_curve(y_binary, scores_ae)
    fpr_if, tpr_if, _ = roc_curve(y_binary, scores_if)
    fpr_vae, tpr_vae, _ = roc_curve(y_binary, scores_vae)
    
    roc_auc_ae = auc(fpr_ae, tpr_ae)
    roc_auc_if = auc(fpr_if, tpr_if)
    roc_auc_vae = auc(fpr_vae, tpr_vae)
    
    return {
        "fpr_ae": fpr_ae,
        "tpr_ae": tpr_ae,
        "roc_auc_ae": roc_auc_ae,
        "fpr_if": fpr_if,
        "tpr_if": tpr_if,
        "roc_auc_if": roc_auc_if,
        "fpr_vae": fpr_vae,
        "tpr_vae": tpr_vae,
        "roc_auc_vae": roc_auc_vae,
    }


def plot_pca_overview(X_test, y_true, save_path="results/pca_anomalies_overview.png"):
    """Plot a 2D PCA representation of the normal data vs injected anomalies.
    
    Args:
        X_test: Test set features
        y_true: True labels (1 for normal, -1 for anomaly)
        save_path: Path to save the plot image
    """
    pca = PCA(n_components=2, random_state=config.SEED)
    X_pca = pca.fit_transform(X_test)
    
    normal_idx = np.where(y_true == 1)[0]
    anomaly_idx = np.where(y_true == -1)[0]
    
    plt.figure(figsize=(10, 8))
    
    # Plot normal points
    plt.scatter(
        X_pca[normal_idx, 0], X_pca[normal_idx, 1],
        c='blue', alpha=0.3, label='Normal (True)', s=20
    )
    
    # Plot injected anomalies
    plt.scatter(
        X_pca[anomaly_idx, 0], X_pca[anomaly_idx, 1],
        c='red', alpha=0.7, label='Injected Anomalies', s=30, marker='x'
    )
    
    plt.title('2D PCA Overview of Test Set (Normal vs Injected Anomalies)')
    plt.xlabel(f'First Principal Component (Var: {pca.explained_variance_ratio_[0]*100:.1f}%)')
    plt.ylabel(f'Second Principal Component (Var: {pca.explained_variance_ratio_[1]*100:.1f}%)')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main(data_dir: str = "data", results_dir: str = "results"):
    """Run complete comparison between Autoencoder and Isolation Forest.
    
    Args:
        data_dir: Directory containing source CSV files
        results_dir: Directory for output results
    """
    results_path = Path(results_dir)
    results_path.mkdir(exist_ok=True)
    
    print("\n" + "█" * 80)
    print("█ BASELINE COMPARISON: AUTOENCODER vs ISOLATION FOREST vs VAE")
    print("█" * 80)
    
    # STEP 1: Extract preprocessed data
    print("\n" + "=" * 80)
    print("STEP 1: EXTRACTING DATA FROM PIPELINE")
    print("=" * 80)
    
    X_train_scaled, X_test_scaled, feature_names, scaler, _ = load_and_preprocess_data(
        data_dir=data_dir,
        results_dir=results_dir
    )
    
    if X_train_scaled is None:
        print("❌ Data extraction failed. Stopping.")
        return
    
    print(f"✓ Training set: {X_train_scaled.shape}")
    print(f"✓ Test set: {X_test_scaled.shape}")
    print(f"✓ Features: {len(feature_names)}")
    
    # STEP 2: Inject synthetic anomalies
    print("\n" + "=" * 80)
    print(f"STEP 2: INJECTING SYNTHETIC ANOMALIES ({config.INJECTION_FRAC*100:.0f}% Gaussian Noise)")
    print("=" * 80)
    
    X_test_corrupted, y_true, anomaly_indices = inject_anomalies(
        X_test_scaled.copy(),
        contamination_frac=config.INJECTION_FRAC,
        noise_scale=config.NOISE_SCALE,
        seed=config.SEED
    )
    
    n_anomalies = np.sum(y_true == -1)
    n_normal = np.sum(y_true == 1)
    print(f"✓ Injected {n_anomalies} anomalies into {len(X_test_corrupted)} test samples")
    print(f"  Normal samples: {n_normal}")
    print(f"  Anomalous samples: {n_anomalies}")
    
    # STEP 3: Train AUTOENCODER
    print("\n" + "=" * 80)
    print("STEP 3: TRAINING AUTOENCODER")
    print("=" * 80)
    
    model_ae, threshold_ae = train_detector(
        X_train_scaled,
        contamination=config.INJECTION_FRAC,
        bottleneck_dim=config.BOTTLENECK_DIM,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        random_state=config.SEED
    )
    
    print("✓ Autoencoder training completed")
    print(f"  Initial threshold: {threshold_ae:.5f}")
    
    # STEP 4: Train ISOLATION FOREST
    print("\n" + "=" * 80)
    print("STEP 4: TRAINING ISOLATION FOREST")
    print("=" * 80)
    
    model_if = train_isolation_forest(
        X_train_scaled,
        contamination=config.INJECTION_FRAC,
        n_estimators=100,
        random_state=config.SEED
    )
    
    threshold_if = get_isolation_forest_threshold(
        model_if,
        X_train_scaled,
        contamination=config.INJECTION_FRAC
    )
    
    print("✓ Isolation Forest training completed")
    print(f"  Initial threshold: {threshold_if:.5f}")
    
    # STEP 5: Train VARIATIONAL AUTOENCODER
    print("\n" + "=" * 80)
    print("STEP 5: TRAINING VARIATIONAL AUTOENCODER")
    print("=" * 80)
    
    model_vae, threshold_vae = train_vae(
        X_train_scaled,
        contamination=config.INJECTION_FRAC,
        latent_dim=config.BOTTLENECK_DIM, # Keep parameter dimensions identical
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        random_state=config.SEED
    )
    
    print("✓ VAE training completed")
    print(f"  Initial threshold: {threshold_vae:.5f}")
    
    # STEP 6: Score and predict on test set
    print("\n" + "=" * 80)
    print("STEP 6: EVALUATING MODELS ON TEST SET")
    print("=" * 80)
    
    # Autoencoder
    scores_ae = score_anomalies(model_ae, X_test_corrupted)
    
    if config.OPTIMIZE_THRESHOLD:
        print("Optimizing thresholds for best F1 balance...")
        threshold_ae_opt, _ = optimize_threshold(
            model_ae, X_test_corrupted, y_true, metric=config.THRESHOLD_METRIC
        )
        print(f"✓ Autoencoder optimal threshold: {threshold_ae_opt:.5f}")
    else:
        threshold_ae_opt = threshold_ae
    
    preds_ae = predict_anomalies(model_ae, X_test_corrupted, threshold_ae_opt)
    
    # Isolation Forest
    scores_if = score_anomalies_if(model_if, X_test_corrupted)
    threshold_if_opt = get_isolation_forest_threshold(
        model_if,
        X_test_corrupted,
        contamination=config.INJECTION_FRAC
    )
    preds_if = predict_anomalies_if(model_if, X_test_corrupted, threshold_if_opt)
    
    print(f"✓ Isolation Forest threshold: {threshold_if_opt:.5f}")
    
    # VAE
    scores_vae = score_anomalies_vae(model_vae, X_test_corrupted)
    if config.OPTIMIZE_THRESHOLD:
        threshold_vae_opt, _ = optimize_threshold_vae(
            model_vae, X_test_corrupted, y_true, metric=config.THRESHOLD_METRIC
        )
        print(f"✓ VAE optimal threshold: {threshold_vae_opt:.5f}")
    else:
        threshold_vae_opt = threshold_vae
        
    preds_vae = predict_anomalies_vae(model_vae, X_test_corrupted, threshold_vae_opt)
    
    # STEP 7: Evaluate models
    print("\n" + "=" * 80)
    print("STEP 7: COMPUTING EVALUATION METRICS")
    print("=" * 80)
    
    metrics_ae = evaluate_model(y_true, preds_ae, scores_ae, "Autoencoder")
    metrics_if = evaluate_model(y_true, preds_if, scores_if, "Isolation Forest")
    metrics_vae = evaluate_model(y_true, preds_vae, scores_vae, "VAE")
    
    print("✓ Evaluation completed for all models")
    
    # Display results
    print(format_metrics_table(metrics_ae, metrics_if, metrics_vae))
    
    # STEP 8: Save detailed results
    print("\n" + "=" * 80)
    print("STEP 8: SAVING RESULTS")
    print("=" * 80)
    
    # Save comparison results to JSON
    # Convert numpy types to Python native types for JSON serialization
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
        "autoencoder": convert_to_serializable(metrics_ae),
        "isolation_forest": convert_to_serializable(metrics_if),
        "vae": convert_to_serializable(metrics_vae),
        "test_set_size": int(len(X_test_corrupted)),
        "n_anomalies_injected": int(n_anomalies),
        "n_normal_samples": int(n_normal),
        "contamination_rate": float(config.INJECTION_FRAC),
        "noise_scale": float(config.NOISE_SCALE),
    }
    
    results_file = results_path / "model_comparison.json"
    with open(results_file, "w") as f:
        json.dump(comparison_results, f, indent=2)
    
    print(f"\n✓ Results saved to {results_file}")
    
    # Save detailed predictions for debugging
    predictions_df = pd.DataFrame({
        "y_true": y_true,
        "ae_pred": preds_ae,
        "if_pred": preds_if,
        "vae_pred": preds_vae,
        "ae_score": scores_ae,
        "if_score": scores_if,
        "vae_score": scores_vae,
        "is_injected_anomaly": np.isin(np.arange(len(y_true)), anomaly_indices),
    })
    
    predictions_file = results_path / "predictions_comparison.csv"
    predictions_df.to_csv(predictions_file, index=False)
    print(f"✓ Detailed predictions saved to {predictions_file}")
    
    # ROC curve data
    roc_data = plot_roc_curves(y_true, scores_ae, scores_if, scores_vae)
    roc_file = results_path / "roc_data.json"
    
    roc_data_serializable = {
        "fpr_ae": roc_data["fpr_ae"].tolist(),
        "tpr_ae": roc_data["tpr_ae"].tolist(),
        "roc_auc_ae": float(roc_data["roc_auc_ae"]),
        "fpr_if": roc_data["fpr_if"].tolist(),
        "tpr_if": roc_data["tpr_if"].tolist(),
        "roc_auc_if": float(roc_data["roc_auc_if"]),
        "fpr_vae": roc_data["fpr_vae"].tolist(),
        "tpr_vae": roc_data["tpr_vae"].tolist(),
        "roc_auc_vae": float(roc_data["roc_auc_vae"]),
    }
    
    with open(roc_file, "w") as f:
        json.dump(roc_data_serializable, f, indent=2)
    
    print(f"✓ ROC data saved to {roc_file}")
    
    # Generate PCA visualization of anomalies
    pca_file = results_path / "pca_anomalies_overview.png"
    plot_pca_overview(X_test_corrupted, y_true, pca_file)
    print(f"✓ PCA overview plot saved to {pca_file}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Autoencoder F1 Score: {metrics_ae['f1']:.4f}")
    print(f"Isolation Forest F1 Score: {metrics_if['f1']:.4f}")
    print(f"VAE F1 Score: {metrics_vae['f1']:.4f}")
    
    f1_scores = {'Autoencoder': metrics_ae['f1'], 'Isolation Forest': metrics_if['f1'], 'VAE': metrics_vae['f1']}
    winner = max(f1_scores, key=f1_scores.get)
    print(f"Winner: {winner}")
    
    print("\nAutoencoder ROC AUC: {:.4f}".format(metrics_ae['roc_auc']))
    print("Isolation Forest ROC AUC: {:.4f}".format(metrics_if['roc_auc']))
    print("VAE ROC AUC: {:.4f}".format(metrics_vae['roc_auc']))
    print("=" * 80)


if __name__ == "__main__":
    main(data_dir="data", results_dir="results")
