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
    print(f"Calculated Anomaly Threshold: {threshold:.5f} (using 2% contamination)")
    
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

    print("\n=== Evaluating Test Set ===")
    print(f"Scoring {len(X_test_scaled)} hold-out test samples...")
    test_scores = score_anomalies(model, X_test_scaled)
    test_preds = predict_anomalies(model, X_test_scaled, threshold)
    
    # Evaluate Performance against our Ground Truth
    acc = accuracy_score(y_true, test_preds)
    precision_anom = precision_score(y_true, test_preds, pos_label=-1)
    recall_anom = recall_score(y_true, test_preds, pos_label=-1)
    f1_anom = f1_score(y_true, test_preds, pos_label=-1)
    
    # Calculate True Positives, False Positives, False Negatives, True Negatives
    # Setting labels=[1, -1] returns flat array [TN, FP, FN, TP]
    tn, fp, fn, tp = confusion_matrix(y_true, test_preds, labels=[1, -1]).ravel()
    
    print("\n" + "=" * 50)
    print("RESULTS ON TEST DATA:")
    print("=" * 50)
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {precision_anom:.4f}")
    print(f"Recall:    {recall_anom:.4f}")
    print(f"F1 Score:  {f1_anom:.4f}")
    print("\n--- Detection Breakdown ---")
    print(f"True Positives (Correctly caught injected anomalies): {tp}")
    print(f"False Positives (Normal profiles flagged as anomaly): {fp}")
    print(f"False Negatives (Missed injected anomalies):          {fn}")
    print(f"True Negatives (Correctly labeled normal profiles):   {tn}")

if __name__ == '__main__':
    main()
