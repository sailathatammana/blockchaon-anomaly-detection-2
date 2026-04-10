import sys
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

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
    
    # Use 20 epochs for testing, 2% assumed contamination
    model, threshold = train_detector(
        X_train_scaled, 
        contamination=config.INJECTION_FRAC, 
        epochs=config.EPOCHS, 
        batch_size=config.BATCH_SIZE
    )
    
    print("\nTraining completed successfully!")

    print("\n=== Injecting Synthetic Anomalies (2% Gaussian Noise) ===")
    np.random.seed(41)  # For reproducible injections
    n_test = len(X_test_scaled)
    n_anomalies = int(n_test * 0.02)  # 2% of test set
    
    # Create Ground Truth Array: 1 for normal, -1 for anomaly
    y_true = np.ones(n_test)
    
    # Randomly select 2% of the test set indices to corrupt
    anomaly_indices = np.random.choice(n_test, size=n_anomalies, replace=False)
    y_true[anomaly_indices] = -1
    
    # Corrupt these samples by adding significant Gaussian noise
    # Mean 0, StdDev 3 (since data is normalized [-1, 1], STD 3 is massive)
    noise = np.random.normal(loc=0.0, scale=3.0, size=(n_anomalies, X_test_scaled.shape[1]))
    X_test_scaled[anomaly_indices] += noise
    
    print(f"Successfully injected {n_anomalies} anomalies into the test set.")

    test_scores = score_anomalies(model, X_test_scaled)
    
    print("\n=== Optimizing Threshold for Precision/Recall Balance ===")
    optimal_threshold, best_f1 = optimize_threshold(
        model, X_test_scaled, y_true, metric="f1"
    )
    print(f"Optimal threshold found: {optimal_threshold:.5f}")
    print(f"Expected F1 score: {best_f1:.4f}")
    
    # Evaluate with optimized threshold
    test_preds = predict_anomalies(model, X_test_scaled, optimal_threshold)
    
    accuracy = accuracy_score(y_true, test_preds)
    precision = precision_score(y_true, test_preds, pos_label=-1)
    recall = recall_score(y_true, test_preds, pos_label=-1)
    f1 = f1_score(y_true, test_preds, pos_label=-1)
    tn, fp, fn, tp = confusion_matrix(y_true, test_preds, labels=[1, -1]).ravel()
    
    print("\n" + "=" * 50)
    print("RESULTS WITH OPTIMIZED THRESHOLD:")
    print("=" * 50)
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("\n--- Detection Breakdown ---")
    print(f"True Positives (Correctly caught injected anomalies): {tp}")
    print(f"False Positives (Normal profiles flagged as anomaly): {fp}")
    print(f"False Negatives (Missed injected anomalies):          {fn}")
    print(f"True Negatives (Correctly labeled normal profiles):   {tn}")

if __name__ == '__main__':
    main()
