"""Variational Autoencoder (VAE) baseline for unsupervised anomaly detection.

The VAE is a generative model that learns the probability distribution of the normal data.
It maps inputs to a normal distribution in a latent space, enforcing it via KL divergence.
Anomalies fail to reconstruct properly since they don't follow the learned normal distribution.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import config


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        
        # Sample normal distribution epsilon
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        
        # Reparameterization trick
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(keras.Model):
    """Variational Autoencoder custom Model class.
    
    Includes a custom train_step to process both the reconstruction loss
    and the KL divergence.
    """

    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        # We only need X, if a tuple is passed, grab only the input X
        if isinstance(data, tuple):
            inputs = data[0]
        else:
            inputs = data
            
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(inputs)
            reconstruction = self.decoder(z)
            
            # Reconstruction Loss (using MSE to remain comparable to standard Autoencoder)
            # Reduce across features, take mean across batch
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_squared_error(inputs, reconstruction), 
                    axis=-1
                )
            )
            
            # KL Divergence Loss
            # Formulate: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            
            # Note: We weight KL loss lower sometimes to let reconstruction drive,
            # but standard formulation is total_loss = reconstruction_loss + kl_loss
            total_loss = reconstruction_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
        
    def call(self, inputs):
        # For prediction, we just pass the mean representation through the decoder
        # to get deterministic reconstruction, instead of randomly sampling.
        z_mean, _, _ = self.encoder(inputs)
        return self.decoder(z_mean)


def build_vae(input_dim: int, latent_dim: int = 4) -> tuple:
    """Build VAE architecture (Encoder, Decoder, and VAE Model).
    
    Architecture aligns with the standard deterministic Autoencoder.
    
    Args:
        input_dim: Number of input features
        latent_dim: Number of dimensions for the latent distribution
    
    Returns:
        Built and compiled VAE Model.
    """
    
    # ---------------- ENCODER ----------------
    encoder_inputs = keras.Input(shape=(input_dim,))
    
    x = layers.Dense(128, activation="relu")(encoder_inputs)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)
    
    # Rather than a single bottleneck, we output a mean and variance
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    
    # Sample from distribution
    z = Sampling()([z_mean, z_log_var])
    
    encoder = keras.Model(
        encoder_inputs, [z_mean, z_log_var, z], name="encoder"
    )
    
    # ---------------- DECODER ----------------
    latent_inputs = keras.Input(shape=(latent_dim,))
    
    x = layers.Dense(32, activation="relu")(latent_inputs)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    
    # Output matches original features using linear activation
    decoder_outputs = layers.Dense(input_dim, activation="linear")(x)
    
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    
    # ---------------- VAE MODEL ----------------
    vae = VAE(encoder, decoder)
    
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-4))
    
    return vae


def train_vae(
    X_train: np.ndarray,
    contamination: float = 0.02,
    latent_dim: int = config.BOTTLENECK_DIM,
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    val_split: float = 0.15,
    random_state: int = config.SEED,
) -> tuple:
    """Train the Variational Autoencoder on clean data.
    
    Args:
        X_train: Normal training data
        contamination: Expected anomaly fraction in training set
        latent_dim: Dimensionality of latent space distribution
        epochs: Number of training epochs
        batch_size: Batch size
        val_split: Hold out set for validation
        random_state: Random state
        
    Returns:
        vae: Trained VAE model
        threshold: Recommended reconstruction error threshold
    """
    np.random.seed(random_state)
    tf.random.set_seed(random_state)
    
    vae = build_vae(input_dim=X_train.shape[1], latent_dim=latent_dim)
    
    callbacks = [
        # Note: Since our custom training loop only properly exposes loss on training data,
        # we aren't monitoring val_loss out of the box in this simplistic custom step without 
        # further overriding test_step. We will monitor simply `loss` or utilize epochs.
        # Alternatively, we just let it run for epochs because VAEs are less prone to naive overfitting.
        keras.callbacks.EarlyStopping(
            monitor="loss", 
            patience=10,
            restore_best_weights=True,
            min_delta=1e-4
        )
    ]
    
    # Train model
    vae.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=0
    )
    
    # Get anomaly scores (Mean Squared Reconstruction Error)
    X_hat = vae.predict(X_train, verbose=0)
    train_errors = np.mean((X_train - X_hat) ** 2, axis=1)
    
    q = max(0.0, min(1.0, 1.0 - contamination))
    threshold = float(np.quantile(train_errors, q))
    print(f"✓ VAE trained. Recommended threshold (based on {contamination*100:.2f}% contamination): {threshold:.4f}")
    
    return vae, threshold


def score_anomalies_vae(model: VAE, X: np.ndarray) -> np.ndarray:
    """Compute anomaly severity scores for given data.
    
    Similar to the deterministic autoencoder, we use Mean Squared Error
    of the reconstruction as our primary anomaly score to remain perfectly 
    baseline comparable.
    
    Args:
        model: Trained VAE
        X: Data to score
        
    Returns:
        Array of scores (higher = more anomalous)
    """
    X_hat = model.predict(X, verbose=0)
    errors = np.mean((X - X_hat) ** 2, axis=1)
    return errors


def predict_anomalies_vae(model: VAE, X: np.ndarray, threshold: float) -> np.ndarray:
    """Predict anomalous (-1) or normal (1) labels based on a threshold.
    
    Args:
        model: Trained VAE
        X: Data to classify
        threshold: Calculated threshold value
        
    Returns:
        Predictions array with -1 or 1
    """
    errors = score_anomalies_vae(model, X)
    preds = np.where(errors > threshold, -1, 1)
    return preds


def optimize_threshold_vae(
    model: VAE, 
    X_test: np.ndarray, 
    y_true: np.ndarray, 
    metric: str = "f1"
) -> tuple:
    """Find the threshold maximizing the target evaluation metric.
    
    Args:
        model: Trained VAE
        X_test: Evaluation data
        y_true: Ground truth labels (1=normal, -1=anomaly)
        metric: Optimization metric ('f1', 'precision', 'recall', 'balanced')
        
    Returns:
        Tuple of (best_threshold, best_score)
    """
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    errors = score_anomalies_vae(model, X_test)
    thresholds = np.sort(errors)
    best_threshold = thresholds[0]
    best_score = -1
    
    for threshold in thresholds:
        preds = np.where(errors > threshold, -1, 1)
        
        if metric == "f1":
            score = f1_score(y_true, preds, pos_label=-1, zero_division=0)
        elif metric == "precision":
            score = precision_score(y_true, preds, pos_label=-1, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true, preds, pos_label=-1, zero_division=0)
        elif metric == "balanced":
            prec = precision_score(y_true, preds, pos_label=-1, zero_division=0)
            rec = recall_score(y_true, preds, pos_label=-1, zero_division=0)
            score = (prec + rec) / 2
        else:
            raise ValueError(f"Unknown metric: {metric}")
            
        if score > best_score:
            best_score = score
            best_threshold = threshold
            
    return best_threshold, best_score