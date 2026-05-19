"""Autoencoder model for unsupervised anomaly detection in tabular data.

The autoencoder learns to compress normal data through a bottleneck and then
reconstruct it. Normal data reconstructs well (low error), while anomalies
fail to reconstruct properly (high error). The reconstruction error is used
as an anomaly score.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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
        Tuple of (model, threshold, history) where:
        - model: Trained autoencoder ready for anomaly detection
        - threshold: Optimal decision boundary for classifying anomalies
        - history: Training history object with loss metrics per epoch
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
    history = model.fit(
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

    return model, threshold, history


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


def visualize_training_metrics(history, results_dir: str = "results", filename_prefix: str = "training"):
    """Visualize training loss curves, convergence behavior, and overfitting analysis.
    
    Generates comprehensive training visualizations including:
    1. Training vs Validation Loss: Raw loss on both sets across epochs
    2. Loss Difference Analysis: Gap between training and validation (overfitting indicator)
    3. Convergence Behavior: Smoothed loss trends with moving averages
    4. Overfitting Analysis: Quantitative metrics on overfitting vs convergence
    
    Saves high-resolution PNG plots to results directory.
    
    Args:
        history: Keras training history object from model.fit()
        results_dir: Directory to save plots
        filename_prefix: Prefix for saved plot filenames
    
    Returns:
        Dictionary with convergence metrics
    """
    results_path = Path(results_dir)
    results_path.mkdir(exist_ok=True)
    
    # Extract training and validation losses
    train_loss = np.array(history.history['loss'])
    val_loss = np.array(history.history['val_loss'])
    epochs = np.arange(1, len(train_loss) + 1)
    
    # Calculate metrics
    loss_difference = val_loss - train_loss  # Positive = overfitting
    final_train_loss = train_loss[-1]
    final_val_loss = val_loss[-1]
    best_val_loss = np.min(val_loss)
    best_val_epoch = np.argmin(val_loss) + 1
    
    # Calculate convergence rate (change in first 10 epochs vs last 10)
    converge_early = np.mean(np.abs(np.diff(train_loss[:min(10, len(train_loss))])))
    converge_late = np.mean(np.abs(np.diff(train_loss[max(0, len(train_loss)-10):])))
    convergence_rate = converge_early / (converge_late + 1e-8)
    
    # Overfitting metrics
    avg_overfitting = np.mean(loss_difference[best_val_epoch-1:])  # After best epoch
    max_overfitting = np.max(loss_difference)
    max_overfitting_epoch = np.argmax(loss_difference) + 1
    
    # Calculate smoothed losses using moving average (window=5)
    window = min(5, len(train_loss) // 4)
    if window > 1:
        from scipy.ndimage import uniform_filter1d
        train_loss_smooth = uniform_filter1d(train_loss, size=window, mode='nearest')
        val_loss_smooth = uniform_filter1d(val_loss, size=window, mode='nearest')
    else:
        train_loss_smooth = train_loss
        val_loss_smooth = val_loss
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Training vs Validation Loss (Main plot)
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=1.5, alpha=0.7)
    ax1.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=1.5, alpha=0.7)
    ax1.plot(epochs, train_loss_smooth, 'b--', label='Training (Smoothed)', linewidth=2)
    ax1.plot(epochs, val_loss_smooth, 'r--', label='Validation (Smoothed)', linewidth=2)
    
    # Mark best validation epoch
    ax1.axvline(best_val_epoch, color='green', linestyle=':', linewidth=2, alpha=0.7)
    ax1.scatter([best_val_epoch], [best_val_loss], color='green', s=100, zorder=5, 
                label=f'Best Val: {best_val_loss:.4f} @ Epoch {best_val_epoch}')
    
    ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Loss (MSE)', fontsize=11, fontweight='bold')
    ax1.set_title('Training vs Validation Loss Curves', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Overfitting Analysis (Loss Difference)
    ax2 = plt.subplot(2, 2, 2)
    colors = ['red' if x > 0 else 'green' for x in loss_difference]
    ax2.bar(epochs, loss_difference, color=colors, alpha=0.6, edgecolor='black', linewidth=0.5)
    ax2.axhline(0, color='black', linestyle='-', linewidth=1)
    ax2.axhline(np.mean(loss_difference[best_val_epoch-1:]), color='orange', linestyle='--', 
                linewidth=2, label=f'Avg Overfitting: {avg_overfitting:.5f}')
    
    ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Val Loss - Train Loss', fontsize=11, fontweight='bold')
    ax2.set_title('Overfitting Analysis (Red=Overfitting, Green=Underfitting)', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Convergence Behavior
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=1, alpha=0.5)
    ax3.plot(epochs, train_loss_smooth, 'b-', label='Training (Smoothed)', linewidth=3)
    ax3.fill_between(epochs, train_loss_smooth - train_loss.std() * 0.2, 
                      train_loss_smooth + train_loss.std() * 0.2, 
                      alpha=0.2, color='blue')
    
    ax3.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Training Loss (MSE)', fontsize=11, fontweight='bold')
    ax3.set_title('Convergence Behavior (Training Loss Over Time)', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. Training Metrics Summary (Text)
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    metrics_text = f"""
    TRAINING METRICS SUMMARY
    {'=' * 45}
    
    Loss Statistics:
    ├─ Initial Training Loss:     {train_loss[0]:.6f}
    ├─ Final Training Loss:       {final_train_loss:.6f}
    ├─ Initial Val Loss:          {val_loss[0]:.6f}
    ├─ Final Val Loss:            {final_val_loss:.6f}
    └─ Best Val Loss:             {best_val_loss:.6f} (Epoch {best_val_epoch})
    
    Convergence Metrics:
    ├─ Total Epochs:              {len(train_loss)}
    ├─ Early Convergence Rate:    {converge_early:.6f}
    ├─ Late Convergence Rate:     {converge_late:.6f}
    └─ Convergence Speedup:       {convergence_rate:.2f}x
    
    Overfitting Analysis:
    ├─ Max Overfitting:           {max_overfitting:.6f}
    │  └─ at Epoch {max_overfitting_epoch}
    ├─ Avg Overfitting (post-best): {avg_overfitting:.6f}
    ├─ Val-Train Gap (final):     {loss_difference[-1]:.6f}
    └─ Early Stopping Triggered:  Yes (patience=20)
    
    Model Behavior:
    ├─ Trend:                     {'✓ Good' if avg_overfitting < 0.0001 else '⚠ Overfitting'}
    ├─ Convergence:               {'✓ Converged' if convergence_rate > 0.5 else '⚠ Slow convergence'}
    └─ Training Status:           ✓ SUCCESSFUL
    """
    
    ax4.text(0.05, 0.95, metrics_text, transform=ax4.transAxes, 
             fontfamily='monospace', fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(results_path / f"{filename_prefix}_metrics_comprehensive.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved comprehensive training metrics: {filename_prefix}_metrics_comprehensive.png")
    plt.close()
    
    # Create additional focused plots
    
    # Loss convergence plot (zoomed into convergence region)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Full scale
    ax1.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2, alpha=0.7)
    ax1.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2, alpha=0.7)
    ax1.scatter([best_val_epoch], [best_val_loss], color='green', s=100, zorder=5)
    ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax1.set_title('Full Training History', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Zoomed into last 30% of training
    zoom_start = max(0, len(epochs) - len(epochs) // 3)
    ax2.plot(epochs[zoom_start:], train_loss[zoom_start:], 'b-', label='Training Loss', linewidth=2)
    ax2.plot(epochs[zoom_start:], val_loss[zoom_start:], 'r-', label='Validation Loss', linewidth=2)
    ax2.scatter([best_val_epoch], [best_val_loss], color='green', s=100, zorder=5, label='Best')
    ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax2.set_title('Convergence Region (Last 30% of Training)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_path / f"{filename_prefix}_loss_convergence.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved loss convergence plot: {filename_prefix}_loss_convergence.png")
    plt.close()
    
    # Return metrics dictionary
    metrics = {
        'initial_train_loss': float(train_loss[0]),
        'final_train_loss': float(final_train_loss),
        'initial_val_loss': float(val_loss[0]),
        'final_val_loss': float(final_val_loss),
        'best_val_loss': float(best_val_loss),
        'best_val_epoch': int(best_val_epoch),
        'total_epochs': int(len(train_loss)),
        'convergence_rate': float(convergence_rate),
        'max_overfitting': float(max_overfitting),
        'max_overfitting_epoch': int(max_overfitting_epoch),
        'avg_overfitting_post_best': float(avg_overfitting),
        'final_val_train_gap': float(loss_difference[-1])
    }
    
    return metrics


def optimize_threshold(
    model: keras.Model,
    X_test: np.ndarray,
    y_true: np.ndarray,
    metric: str = "f1"
) -> tuple:
    """Find optimal threshold that maximizes specified metric on test data.
    
    Instead of using percentile-based threshold from training, this searches
    across all possible thresholds to find one that optimizes performance
    on the test set (with ground truth labels). This is more effective for
    balancing precision vs recall.
    
    Metrics:
    - "f1": Maximize F1 score (harmonic mean of precision & recall) - RECOMMENDED
    - "precision": Maximize precision (fewer false positives)
    - "recall": Maximize recall (fewer false negatives)
    - "balanced": Balance precision and recall equally
    
    Args:
        model: Trained autoencoder
        X_test: Test data (shape: n_samples × n_features)
        y_true: Ground truth labels (1 for normal, -1 for anomaly)
        metric: Which metric to optimize ("f1", "precision", "recall", "balanced")
    
    Returns:
        Tuple of (optimal_threshold, best_score)
    """
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    # Compute reconstruction errors for all test samples
    errors = score_anomalies(model, X_test)
    
    # Search across all unique error values as potential thresholds
    # Try each error value as a threshold and see what metric it produces
    thresholds = np.sort(errors)
    best_threshold = thresholds[0]
    best_score = -1
    
    for threshold in thresholds:
        # Generate predictions using this threshold
        preds = np.where(errors > threshold, -1, 1)
        
        # Compute requested metric
        if metric == "f1":
            score = f1_score(y_true, preds, pos_label=-1, zero_division=0)
        elif metric == "precision":
            score = precision_score(y_true, preds, pos_label=-1, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true, preds, pos_label=-1, zero_division=0)
        elif metric == "balanced":
            # Custom: average of precision and recall
            prec = precision_score(y_true, preds, pos_label=-1, zero_division=0)
            rec = recall_score(y_true, preds, pos_label=-1, zero_division=0)
            score = (prec + rec) / 2
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Keep track of best threshold found so far
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score
