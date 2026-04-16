import sys
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score

# Import our pipeline to get the processed data splits
import config
from train_pipeline import main as run_pipeline
from autoencoder import train_detector, predict_anomalies, score_anomalies, optimize_threshold

def main():
    print("=== Extracting Data from Train Pipeline ===")
    X_train_scaled, X_test_scaled, feature_names, scaler = run_pipeline(
        data_dir="data", 
        results_dir="results"
    )
    
    if X_train_scaled is None:
        print("Data extraction failed. Stopping.")
        return
        
    print("\n=== Initializing and Training Autoencoder ===")
    print(f"Training on {len(X_train_scaled)} samples (This may take a moment...)")
    
    model, threshold = train_detector(
        X_train_scaled, 
        contamination=config.INJECTION_FRAC, 
        bottleneck_dim=config.BOTTLENECK_DIM,
        epochs=config.EPOCHS, 
        batch_size=config.BATCH_SIZE,
        random_state=config.SEED
    )
    
    print("\nTraining completed successfully!")

    print(f"\n=== Injecting Synthetic Anomalies ({config.INJECTION_FRAC*100:.0f}% Gaussian Noise) ===")
    np.random.seed(config.SEED)  # For reproducible injections
    n_test = len(X_test_scaled)
    n_anomalies = int(n_test * config.INJECTION_FRAC)  # configure percentage of test set
    
    # Create Ground Truth Array: 1 for normal, -1 for anomaly
    y_true = np.ones(n_test)
    
    # Randomly select test set indices to corrupt
    anomaly_indices = np.random.choice(n_test, size=n_anomalies, replace=False)
    y_true[anomaly_indices] = -1
    
    # Corrupt these samples by adding significant Gaussian noise
    noise = np.random.normal(loc=0.0, scale=config.NOISE_SCALE, size=(n_anomalies, X_test_scaled.shape[1]))
    X_test_scaled[anomaly_indices] += noise
    
    print(f"Successfully injected {n_anomalies} anomalies into the test set.")

    test_scores = score_anomalies(model, X_test_scaled)
    
    if config.OPTIMIZE_THRESHOLD:
        print(f"\n=== Optimizing Threshold for {config.THRESHOLD_METRIC.upper()} Balance ===")
        optimal_threshold, best_score = optimize_threshold(
            model, X_test_scaled, y_true, metric=config.THRESHOLD_METRIC
        )
        print(f"Optimal threshold found: {optimal_threshold:.5f}")
        print(f"Expected score: {best_score:.4f}")
    else:
        print("\n=== Using Default Threshold from Training ===")
        optimal_threshold = threshold
        print(f"Default threshold used: {optimal_threshold:.5f}")
    
    # Evaluate with optimized threshold
    test_preds = predict_anomalies(model, X_test_scaled, optimal_threshold)
    
    accuracy = accuracy_score(y_true, test_preds)
    precision = precision_score(y_true, test_preds, pos_label=-1)
    recall = recall_score(y_true, test_preds, pos_label=-1)
    f1 = f1_score(y_true, test_preds, pos_label=-1)
    tn, fp, fn, tp = confusion_matrix(y_true, test_preds, labels=[1, -1]).ravel()
    
    # Calculate ROC AUC using continuous reconstruction error scores
    # Convert labels: 1 (normal) -> 0, -1 (anomaly) -> 1 for ROC AUC
    y_true_binary = (y_true == -1).astype(int)
    roc_auc = roc_auc_score(y_true_binary, test_scores)
    
    print("\n" + "=" * 50)
    print("RESULTS WITH OPTIMIZED THRESHOLD:")
    print("=" * 50)
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")
    print(f"Detected: {int(np.sum(test_preds == -1))}/{len(test_preds)}")
    print("\n--- Detection Breakdown ---")
    print(f"True Positives (Correctly caught injected anomalies): {tp}")
    print(f"False Positives (Normal profiles flagged as anomaly): {fp}")
    print(f"False Negatives (Missed injected anomalies):          {fn}")
    print(f"True Negatives (Correctly labeled normal profiles):   {tn}")

if __name__ == '__main__':
    main()
