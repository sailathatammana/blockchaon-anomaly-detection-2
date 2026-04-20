"""Isolation Forest baseline for anomaly detection comparison.

UPDATED FOR REAL DATA
Supports multiple training strategies for real outcome data:

Strategy A: "normal_only" - Train on normal students only (conservative)
            Similar to autoencoder: learns patterns of normal behavior
            
Strategy B: "all_data" - Train on all students with known contamination rate
            Uses the true proportion of abnormal students
            
Strategy C: "informed" - Train on all students with buffer
            Adds buffer to contamination estimate for robustness
            Recommended approach - balances both strategies

See LABELING_STRATEGY.md for detailed comparison.
"""
import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Tuple, Optional


def train_isolation_forest_real_data(
    X_train: np.ndarray,
    y_train: Optional[np.ndarray] = None,
    strategy: str = "informed",
    contamination: float = 0.02,
    contamination_buffer: float = 0.05,
    n_estimators: int = 100,
    random_state: int = 41
) -> Tuple[IsolationForest, dict]:
    """Train Isolation Forest for real anomaly detection.
    
    REAL DATA APPROACH:
    Supports multiple strategies for leveraging outcome data.
    
    Args:
        X_train: Training data (shape: n_samples × n_features)
        y_train: Optional labels (1=normal, -1=abnormal)
                Required for "all_data" and "informed" strategies
        strategy: Training approach ("normal_only", "all_data", "informed")
        contamination: Expected proportion of anomalies (0.0-0.5)
                      Only used for strategy="all_data"
        contamination_buffer: Buffer to add for "informed" strategy
        n_estimators: Number of isolation trees
        random_state: Seed for reproducibility
    
    Returns:
        Tuple of:
            - model: Trained IsolationForest model
            - strategy_info: Dictionary with strategy details
    """
    strategy_info = {"strategy": strategy}
    
    if strategy == "normal_only":
        # Strategy A: Train on normal data only (if labels provided)
        if y_train is not None:
            X_normal = X_train[y_train == 1]
            strategy_info["description"] = "Trained on normal students only"
            strategy_info["training_samples"] = len(X_normal)
            strategy_info["contamination_param"] = 0.0
            X_to_train = X_normal
            contam_param = 0.0
        else:
            # No labels: train on all data with low contamination
            strategy_info["description"] = "No labels provided, training on all data (conservative)"
            strategy_info["training_samples"] = len(X_train)
            strategy_info["contamination_param"] = 0.02
            X_to_train = X_train
            contam_param = 0.02
            
    elif strategy == "all_data":
        # Strategy B: Train on all data with true contamination
        if y_train is not None:
            true_contamination = np.sum(y_train == -1) / len(y_train)
            contam_param = true_contamination
            strategy_info["description"] = f"Trained on all students with true contamination rate"
            strategy_info["training_samples"] = len(X_train)
            strategy_info["true_contamination"] = float(true_contamination)
            strategy_info["contamination_param"] = float(true_contamination)
        else:
            contam_param = contamination
            strategy_info["description"] = "Trained on all data with provided contamination"
            strategy_info["training_samples"] = len(X_train)
            strategy_info["contamination_param"] = float(contamination)
        X_to_train = X_train
        
    elif strategy == "informed":
        # Strategy C: Train on all data with contamination buffer
        if y_train is not None:
            true_contamination = np.sum(y_train == -1) / len(y_train)
            contam_param = min(0.5, true_contamination + contamination_buffer)
            strategy_info["description"] = f"Trained on all students with contamination buffer"
            strategy_info["training_samples"] = len(X_train)
            strategy_info["true_contamination"] = float(true_contamination)
            strategy_info["buffer"] = float(contamination_buffer)
            strategy_info["contamination_param"] = float(contam_param)
        else:
            contam_param = contamination
            strategy_info["description"] = "Trained on all data with provided contamination + buffer"
            strategy_info["training_samples"] = len(X_train)
            strategy_info["contamination_param"] = float(contam_param)
        X_to_train = X_train
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Train model
    model = IsolationForest(
        contamination=contam_param,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,  # Use all available processors
        verbose=0
    )
    
    model.fit(X_to_train)
    
    return model, strategy_info


def train_isolation_forest(
    X_train: np.ndarray,
    contamination: float = 0.02,
    n_estimators: int = 100,
    random_state: int = 41
) -> IsolationForest:
    """Train Isolation Forest on training data (legacy interface).
    
    Isolation Forest works by:
    1. Randomly selecting features and split values
    2. Building isolation trees that separate data points
    3. Computing anomaly scores based on path lengths
    4. Shorter paths = anomalies (they isolate more easily)
    
    Args:
        X_train: Training data (shape: n_samples × n_features)
        contamination: Expected proportion of anomalies (0.0-0.5)
                      Used to determine decision threshold
        n_estimators: Number of isolation trees to build
        random_state: Seed for reproducibility
    
    Returns:
        Trained IsolationForest model ready for prediction/scoring
    """
    model = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,  # Use all available processors
        verbose=0
    )
    
    model.fit(X_train)
    return model


def score_anomalies_if(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """Compute anomaly scores using Isolation Forest.
    
    The Isolation Forest provides anomaly scores via score_samples():
    - Negative scores = anomalous (short isolation path)
    - Positive scores = normal (long isolation path)
    - Scores are approximately in range [-1, 0.5]
    
    For comparison with autoencoder (where higher = more anomalous),
    we negate the scores so that positive = anomalous.
    
    Args:
        model: Trained IsolationForest model
        X: Data to score (shape: n_samples × n_features)
    
    Returns:
        Array of anomaly scores (positive = anomalous, like autoencoder)
    """
    # Get negative scores (Isolation Forest convention: negative = anomalous)
    raw_scores = model.score_samples(X)
    
    # Negate to match autoencoder convention (positive = anomalous)
    anomaly_scores = -raw_scores
    
    return anomaly_scores


def predict_anomalies_if(
    model: IsolationForest,
    X: np.ndarray,
    threshold: float = None
) -> np.ndarray:
    """Predict anomalies using Isolation Forest.
    
    Args:
        model: Trained IsolationForest model
        X: Data to predict (shape: n_samples × n_features)
        threshold: Optional custom anomaly threshold
                  If None, uses model's default decision boundary
    
    Returns:
        Predictions array:
        - 1 for normal samples
        - -1 for anomalies
    """
    if threshold is None:
        # Use model's built-in threshold (based on contamination)
        predictions = model.predict(X)
    else:
        # Use custom threshold on anomaly scores
        scores = score_anomalies_if(model, X)
        predictions = np.where(scores > threshold, -1, 1)
    
    return predictions


def get_isolation_forest_threshold(
    model: IsolationForest,
    X: np.ndarray,
    contamination: float = 0.02
) -> float:
    """Get the anomaly threshold for Isolation Forest predictions.
    
    Args:
        model: Trained IsolationForest model
        X: Data to compute threshold on
        contamination: Contamination parameter (fraction of anomalies)
    
    Returns:
        Threshold value for anomaly scores
    """
    scores = score_anomalies_if(model, X)
    # Use quantile corresponding to contamination
    q = max(0.0, min(1.0, 1.0 - contamination))
    threshold = float(np.quantile(scores, q))
    
    return threshold


def predict_anomalies_if(model: IsolationForest,
    X: np.ndarray,
    threshold: float = None
) -> np.ndarray:
    """Predict anomalies using Isolation Forest.
    
    Args:
        model: Trained IsolationForest model
        X: Data to predict (shape: n_samples × n_features)
        threshold: Optional custom anomaly threshold
                  If None, uses model's default decision boundary
    
    Returns:
        Predictions array:
        - 1 for normal samples
        - -1 for anomalies
    """
    if threshold is None:
        # Use model's built-in threshold (based on contamination)
        predictions = model.predict(X)
    else:
        # Use custom threshold on anomaly scores
        scores = score_anomalies_if(model, X)
        predictions = np.where(scores > threshold, -1, 1)
    
    return predictions


def get_isolation_forest_threshold(
    model: IsolationForest,
    X: np.ndarray,
    contamination: float = 0.02
) -> float:
    """Get the anomaly threshold for Isolation Forest predictions.
    
    Args:
        model: Trained IsolationForest model
        X: Data to compute threshold on
        contamination: Contamination parameter (fraction of anomalies)
    
    Returns:
        Threshold value for anomaly scores
    """
    scores = score_anomalies_if(model, X)
    # Use quantile corresponding to contamination
    q = max(0.0, min(1.0, 1.0 - contamination))
    threshold = float(np.quantile(scores, q))
    
    return threshold
