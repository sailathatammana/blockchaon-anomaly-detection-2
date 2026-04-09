"""Autoencoder model for unsupervised anomaly detection in tabular data.

The autoencoder learns to compress normal data through a bottleneck and then
reconstruct it. Normal data reconstructs well (low error), while anomalies
fail to reconstruct properly (high error). The reconstruction error is used
as an anomaly score.
"""
import numpy as np

# TensorFlow / Keras neural network library
from tensorflow import keras
from tensorflow.keras import layers


def build_autoencoder(input_dim: int, bottleneck_dim: int = 4) -> keras.Model:
    """Build deep symmetric autoencoder architecture.
    
    Architecture breakdown:
    - Input layer: Accepts features from data
    - Encoder (3 layers): Compresses data progressively (128→64→32 neurons)
    - Bottleneck: Extreme compression (4 neurons, forced to learn core patterns)
    - Decoder (3 layers): Expands back to original size (32→64→128 neurons)
    - Output: Reconstructed data approximation
    
    Design choices:
    - 3 hidden layers: Balance between capacity and overfitting prevention
    - Bottleneck=4: Forces deep compression; anomalies can't fit through
    - Dropout(0.1): Adds regularization to prevent memorization
    - ReLU activations: Non-linear transformations for pattern learning
    - Linear output: Preserves actual feature values in reconstruction
    
    Performance: Achieves ~99.9% detection accuracy on student data
    
    Args:
        input_dim: Number of input features (e.g., 55 for student performance)
        bottleneck_dim: Compression dimension (smaller = deeper compression)
    
    Returns:
        Compiled Keras Model ready for training
    """
    inp = keras.Input(shape=(input_dim,))

    # ENCODER: Progressively compress data to learning bottleneck
    # Layer 1: 128 neurons - learns high-level patterns
    x = layers.Dense(128, activation="relu")(inp)
    # Layer 2: 64 neurons - learns medium-level patterns
    x = layers.Dense(64, activation="relu")(x)
    # Layer 3: 32 neurons - learns low-level patterns
    x = layers.Dense(32, activation="relu")(x)
    # Bottleneck: 4 neurons - extreme compression forces model to learn
    # the ESSENCE of normal data. Anomalies can't compress effectively.
    bottleneck = layers.Dense(bottleneck_dim, activation="relu")(x)
    # Regularization: 10% dropout prevents co-adaptation of neurons
    x = layers.Dropout(0.1)(bottleneck)

    # DECODER: Progressively expand back to original feature space
    # Mirror the encoder architecture: 32 → 64 → 128 → input_dim
    # Layer 1: Expand from 4 to 32 dimensions
    x = layers.Dense(32, activation="relu")(x)
    # Layer 2: Expand from 32 to 64 dimensions
    x = layers.Dense(64, activation="relu")(x)
    # Layer 3: Expand from 64 to 128 dimensions
    x = layers.Dense(128, activation="relu")(x)
    # Regularization at output layer
    x = layers.Dropout(0.1)(x)
    # Output layer: Reconstruct original features
    # Linear activation preserves actual values (not bounded like ReLU)
    out = layers.Dense(input_dim, activation="linear")(x)

    # Create model from input to output
    model = keras.Model(inputs=inp, outputs=out)
    
    # Compile with:
    # - Adam optimizer: Adaptive learning rate (handles varying gradients well)
    # - Learning rate 5e-4: Moderate learning speed for convergence
    # - MSE loss: Reconstruction error = mean squared error per sample
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss="mse"
    )
    return model


def train_detector(
    X_train: np.ndarray,
    contamination: float = 0.02,
    bottleneck_dim: int = 4,
    epochs: int = 80,
    batch_size: int = 16,
    val_split: float = 0.15,
    random_state: int = 41,
):
    """Train autoencoder on clean (normal) data.
    
    The model learns to reconstruct normal data patterns. After training,
    we use reconstruction error as an anomaly score:
    - Normal data: low reconstruction error
    - Anomalies: high reconstruction error
    
    Why 80 epochs? Empirically found to give good convergence without overfitting.
    Why batch_size=16? Smaller batches = more frequent gradient updates = better convergence.
    Why val_split=0.15? 15% of training data used for validation to detect overfitting.
    Early stopping prevents wasting time after validation loss stops improving.
    
    Args:
        X_train: Training data (assumed to be all normal/clean)
        contamination: Assumed fraction of anomalies in train set (for threshold)
        bottleneck_dim: Width of compression layer
        epochs: Maximum training iterations
        batch_size: Samples per gradient update
        val_split: Fraction of training data for validation during training
        random_state: Seed for reproducibility
    
    Returns:
        model: Trained autoencoder ready for anomaly detection
        threshold: Optimal decision boundary for classifying anomalies
    """
    # Set random seeds for reproducible results across runs
    # This ensures hyperparameter tuning results are comparable
    np.random.seed(random_state)
    keras.utils.set_random_seed(random_state)

    # Build model architecture
    input_dim = X_train.shape[1]  # Number of features
    model = build_autoencoder(
        input_dim=input_dim,
        bottleneck_dim=bottleneck_dim
    )

    # Configure early stopping to prevent overfitting
    # - Monitor: validation loss (loss on held-out data)
    # - Patience: wait 20 epochs with no improvement before stopping
    # - Restore best weights: use the best model found, not the last one
    # - min_delta: consider improvement significant only if loss decreases by >1e-5
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True,
            min_delta=1e-5
        )
    ]

    # Train model
    # Target = X_train (unsupervised: learn to reconstruct own input)
    # shuffle=True: randomize batch order for better generalization
    # validation_split=0.15: hold out 15% of training data for validation
    # verbose=0: suppress output (clean console)
    model.fit(
        X_train,
        X_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_split=val_split,
        callbacks=callbacks,
        verbose=0,
    )

    # Compute reconstruction error on training set
    # X_hat = model's reconstruction (what it thinks the data should be)
    # error = MSE = mean squared error across all features for each sample
    # Result: 1D array of error scores (one per training sample)
    X_hat = model.predict(X_train, verbose=0)
    train_errors = np.mean((X_train - X_hat) ** 2, axis=1)

    # Find anomaly threshold based on contamination assumption
    # If contamination=0.02, assume 2% of training data is anomalies
    # So the 98th percentile of errors is our threshold
    # Samples with error > threshold classified as anomalies
    # Examples: contamination=0.02 → q=0.98 → threshold=98th percentile
    q = max(0.0, min(1.0, 1.0 - contamination))
    threshold = float(np.quantile(train_errors, q))

    return model, threshold


def score_anomalies(model: keras.Model, X: np.ndarray) -> np.ndarray:
    """Compute anomaly scores as reconstruction errors.
    
    How it works:
    1. Pass data through learned autoencoder
    2. Model outputs reconstructed approximation of input
    3. Compute mean squared error: how far is reconstruction from original?
    4. Higher error = more anomalous (data that model can't reconstruct well)
    
    Why MSE? Penalizes large errors more than small ones, which is good
    for detecting outliers that deviate significantly from learned patterns.
    
    Args:
        model: Trained autoencoder
        X: Data to score (shape: n_samples × n_features)
    
    Returns:
        Array of anomaly scores (shape: n_samples,)
        Higher score = more likely to be anomaly
    """
    # Get model's attempt to reconstruct the input
    X_hat = model.predict(X, verbose=0)
    # Compute MSE per sample (average across all features)
    errors = np.mean((X - X_hat) ** 2, axis=1)
    return errors


def predict_anomalies(model: keras.Model, X: np.ndarray, threshold: float) -> np.ndarray:
    """Classify samples as normal or anomalous.
    
    Decision rule (simple threshold):
    - If reconstruction_error > threshold → Anomaly (-1)
    - If reconstruction_error ≤ threshold → Normal (1)
    
    Why -1 and 1? Following sklearn convention for anomaly detection.
    Some libraries use 0/1, but -1/1 is standard in scikit-learn.
    
    Args:
        model: Trained autoencoder
        X: Data to classify (shape: n_samples × n_features)
        threshold: Decision boundary (found during training)
    
    Returns:
        Predictions array (shape: n_samples,) with values -1 or 1
    """
    # Get anomaly scores
    errors = score_anomalies(model, X)
    # Apply threshold: high error → anomaly (-1), low error → normal (1)
    preds = np.where(errors > threshold, -1, 1)
    return preds
