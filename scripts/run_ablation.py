"""Ablation Study Runner — Main Autoencoder Only.

Runs two categories of controlled ablation experiments:

  1. Architectural Ablation
     a) Bottleneck Dimension  — sweeps the compression width [2, 4, 8, 16, 32]
     b) Network Depth         — sweeps the number of hidden encoder/decoder layers [1, 2, 3, 4]

  2. Data Ablation
     Anomaly Severity (Noise Scale) — sweeps the Gaussian noise injected into
     synthetic anomalies [0.5, 1.0, 1.8, 2.5, 4.0]

Each condition is evaluated across five random seeds using Precision, Recall,
F1, and ROC-AUC. Results are printed to the console and saved as CSV files
in results/ablation/.
"""

import os
import sys
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

# ── Environment flags must be set before TensorFlow is imported ──────────────
os.environ["PYTHONHASHSEED"] = "41"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
)
from sklearn.model_selection import train_test_split

# ── Project modules ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
import config
from data_loader import EducationalDataLoader
from preprocessing import EducationalDataPreprocessor
from autoencoder import score_anomalies

# ── Output directory ──────────────────────────────────────────────────────────
ABLATION_DIR = Path("results") / "ablation"
ABLATION_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = [41, 42, 43, 44, 45]


# ═══════════════════════════════════════════════════════════════════════════════
# Model builders
# ═══════════════════════════════════════════════════════════════════════════════

def build_autoencoder_variable(
    input_dim: int,
    bottleneck_dim: int = 4,
    n_hidden_layers: int = 3,
) -> keras.Model:
    """Build a symmetric autoencoder with configurable depth and bottleneck.

    The hidden layer widths are fixed at [128, 64, 32] (first 3 slots).
    When n_hidden_layers < 3 only the first n layers are used; when it is
    4 a fourth layer of 16 neurons is appended on each side.
    """
    candidate_widths = [128, 64, 32, 16]
    widths = candidate_widths[:n_hidden_layers]

    inp = keras.Input(shape=(input_dim,))

    # ── Encoder ───────────────────────────────────────────────────────────────
    x = inp
    for w in widths:
        x = layers.Dense(w, activation="relu")(x)

    bottleneck = layers.Dense(bottleneck_dim, activation="relu")(x)
    x = layers.Dropout(0.1)(bottleneck)

    # ── Decoder (mirror of encoder) ───────────────────────────────────────────
    for w in reversed(widths):
        x = layers.Dense(w, activation="relu")(x)

    x = layers.Dropout(0.1)(x)
    out = layers.Dense(input_dim, activation="linear")(x)

    model = keras.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss="mse",
    )
    return model


def train_autoencoder(
    X_train: np.ndarray,
    bottleneck_dim: int = 4,
    n_hidden_layers: int = 3,
    epochs: int = 80,
    batch_size: int = 16,
    val_split: float = 0.15,
    random_state: int = 41,
) -> tuple:
    """Train the autoencoder."""

    np.random.seed(random_state)
    keras.utils.set_random_seed(random_state)

    model = build_autoencoder_variable(
        input_dim=X_train.shape[1],
        bottleneck_dim=bottleneck_dim,
        n_hidden_layers=n_hidden_layers,
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True,
        min_delta=1e-5,
    )

    model.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_split=val_split,
        callbacks=[early_stop],
        verbose=0,
    )

    return model


# ═══════════════════════════════════════════════════════════════════════════════
# Data helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_data(data_dir: str = "data", results_dir: str = "results") -> tuple:
    """Load and preprocess the full dataset."""

    data_path = Path(data_dir)
    results_path = Path(results_dir)
    results_path.mkdir(exist_ok=True)

    try:
        loader = EducationalDataLoader(data_dir=str(data_path))
        loader.load_all_datasets()
        merged_df = loader.create_merged_dataset()

        preprocessor = EducationalDataPreprocessor()
        profiles = preprocessor.create_student_profiles(merged_df)
        X, _ = preprocessor.prepare_for_autoencoder(profiles)

        return X, preprocessor

    except Exception as exc:
        print(f"\n[ERROR] Data loading failed: {exc}")
        import traceback
        traceback.print_exc()
        return None, None


def split_data(X, preprocessor, random_state):
    """Create a 60/15/25 split and normalize using training data only."""

    X_train_val, X_test = train_test_split(
        X,
        test_size=config.TEST_SIZE,
        random_state=random_state,
    )

    val_fraction = config.VAL_SIZE / (config.TRAIN_SIZE + config.VAL_SIZE)

    X_train, X_val = train_test_split(
        X_train_val,
        test_size=val_fraction,
        random_state=random_state,
    )

    scaler, X_train_scaled = preprocessor.normalize_features(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled


def inject_anomalies(
    X_data: np.ndarray,
    injection_frac: float,
    noise_scale: float,
    random_state: int = 41,
) -> tuple:
    """Inject Gaussian noise anomalies into a copy of the data."""

    rng = np.random.RandomState(random_state)
    X_corrupted = X_data.copy()
    n_samples = len(X_data)
    n_anomalies = int(n_samples * injection_frac)

    y_true = np.ones(n_samples, dtype=int)
    anomaly_indices = rng.choice(
        n_samples,
        size=n_anomalies,
        replace=False
    )
    y_true[anomaly_indices] = -1

    noise = rng.normal(
        loc=0.0,
        scale=noise_scale,
        size=(n_anomalies, X_data.shape[1])
    )
    X_corrupted[anomaly_indices] += noise

    return X_corrupted, y_true


def find_best_threshold(scores, y_true) -> float:
    """Find the threshold that maximizes F1 on validation data."""

    thresholds = np.quantile(
        scores,
        np.linspace(0.05, 0.999, 400)
    )

    best_threshold = thresholds[0]
    best_f1 = 0.0

    for threshold in thresholds:
        preds = np.where(scores > threshold, -1, 1)

        current_f1 = f1_score(
            y_true,
            preds,
            pos_label=-1,
            zero_division=0
        )

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold

    return float(best_threshold)


def evaluate(model, X_val_corrupted, y_val,
             X_test_corrupted, y_test) -> dict:
    """Select threshold on validation data and evaluate on test data."""

    val_scores = score_anomalies(model, X_val_corrupted)
    opt_threshold = find_best_threshold(val_scores, y_val)

    test_scores = score_anomalies(model, X_test_corrupted)
    preds = np.where(test_scores > opt_threshold, -1, 1)

    y_binary = (y_test == -1).astype(int)

    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(
            y_test, preds, pos_label=-1, zero_division=0
        )),
        "recall": float(recall_score(
            y_test, preds, pos_label=-1, zero_division=0
        )),
        "f1": float(f1_score(
            y_test, preds, pos_label=-1, zero_division=0
        )),
        "roc_auc": float(roc_auc_score(y_binary, test_scores)),
        "threshold": float(opt_threshold),
    }


def summarise_results(df: pd.DataFrame, varied_col: str) -> pd.DataFrame:
    """Calculate mean and standard deviation across seeds."""

    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    summary = df.groupby(varied_col)[metrics].agg(["mean", "std"])
    summary.columns = [
        f"{metric}_{stat}"
        for metric, stat in summary.columns
    ]
    summary = summary.reset_index()

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# Pretty printing helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _banner(title: str) -> None:
    print("\n" + "█" * 70)
    print(f"█  {title}")
    print("█" * 70)


def _print_results_table(df: pd.DataFrame, varied_col: str) -> None:
    col_order = [
        varied_col,
        "precision_mean", "precision_std",
        "recall_mean", "recall_std",
        "f1_mean", "f1_std",
        "roc_auc_mean", "roc_auc_std",
    ]
    col_order = [c for c in col_order if c in df.columns]
    print(df[col_order].to_string(index=False))


# ═══════════════════════════════════════════════════════════════════════════════
# Study 1a — Bottleneck Dimension
# ═══════════════════════════════════════════════════════════════════════════════

BOTTLENECK_SWEEP = [2, 4, 8, 16, 32]


def study_bottleneck(X, preprocessor) -> pd.DataFrame:
    """Sweep bottleneck dimension; all other settings match config defaults."""

    _banner("ARCHITECTURAL ABLATION — Bottleneck Dimension")
    print(f"  Sweep: {BOTTLENECK_SWEEP}")
    print(f"  Network depth (hidden layers per side): {3}  [fixed]")
    print(f"  Noise scale: {config.NOISE_SCALE}  [fixed]\n")

    records = []

    for seed in SEEDS:
        print(f"\n  Seed {seed}")

        X_train, X_val, X_test = split_data(
            X, preprocessor, seed
        )

        X_val_corrupted, y_val = inject_anomalies(
            X_val,
            injection_frac=config.INJECTION_FRAC,
            noise_scale=config.NOISE_SCALE,
            random_state=seed,
        )

        X_test_corrupted, y_test = inject_anomalies(
            X_test,
            injection_frac=config.INJECTION_FRAC,
            noise_scale=config.NOISE_SCALE,
            random_state=seed + 1,
        )

        for bd in BOTTLENECK_SWEEP:
            print(
                f"  → bottleneck_dim = {bd:>2d}  ...",
                end="",
                flush=True
            )

            model = train_autoencoder(
                X_train,
                bottleneck_dim=bd,
                n_hidden_layers=3,
                epochs=config.EPOCHS,
                batch_size=config.BATCH_SIZE,
                random_state=seed,
            )

            metrics = evaluate(
                model,
                X_val_corrupted,
                y_val,
                X_test_corrupted,
                y_test
            )

            records.append({
                "seed": seed,
                "bottleneck_dim": bd,
                **metrics
            })

            print(
                f"  Prec={metrics['precision']:.3f}  "
                f"Rec={metrics['recall']:.3f}  "
                f"F1={metrics['f1']:.3f}  "
                f"AUC={metrics['roc_auc']:.3f}"
            )

            keras.backend.clear_session()

    individual_df = pd.DataFrame(records)
    individual_path = ABLATION_DIR / "bottleneck_ablation_individual.csv"
    individual_df.to_csv(individual_path, index=False)

    df = summarise_results(
        individual_df,
        "bottleneck_dim"
    )

    out_path = ABLATION_DIR / "bottleneck_ablation.csv"
    df.to_csv(out_path, index=False)

    print(f"\n  Results saved → {out_path}")
    print()
    _print_results_table(df, "bottleneck_dim")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Study 1b — Network Depth
# ═══════════════════════════════════════════════════════════════════════════════

DEPTH_SWEEP = [1, 2, 3, 4]


def study_depth(X, preprocessor) -> pd.DataFrame:
    """Sweep number of hidden encoder/decoder layers; bottleneck kept at default."""

    _banner("ARCHITECTURAL ABLATION — Network Depth")
    print(f"  Sweep (hidden layers per side): {DEPTH_SWEEP}")
    print(f"  Bottleneck dim: {config.BOTTLENECK_DIM}  [fixed]")
    print(f"  Noise scale: {config.NOISE_SCALE}  [fixed]\n")

    records = []

    for seed in SEEDS:
        print(f"\n  Seed {seed}")

        X_train, X_val, X_test = split_data(
            X, preprocessor, seed
        )

        X_val_corrupted, y_val = inject_anomalies(
            X_val,
            injection_frac=config.INJECTION_FRAC,
            noise_scale=config.NOISE_SCALE,
            random_state=seed,
        )

        X_test_corrupted, y_test = inject_anomalies(
            X_test,
            injection_frac=config.INJECTION_FRAC,
            noise_scale=config.NOISE_SCALE,
            random_state=seed + 1,
        )

        for depth in DEPTH_SWEEP:
            print(
                f"  → n_hidden_layers = {depth}  ...",
                end="",
                flush=True
            )

            model = train_autoencoder(
                X_train,
                bottleneck_dim=config.BOTTLENECK_DIM,
                n_hidden_layers=depth,
                epochs=config.EPOCHS,
                batch_size=config.BATCH_SIZE,
                random_state=seed,
            )

            metrics = evaluate(
                model,
                X_val_corrupted,
                y_val,
                X_test_corrupted,
                y_test
            )

            records.append({
                "seed": seed,
                "n_hidden_layers": depth,
                **metrics
            })

            print(
                f"  Prec={metrics['precision']:.3f}  "
                f"Rec={metrics['recall']:.3f}  "
                f"F1={metrics['f1']:.3f}  "
                f"AUC={metrics['roc_auc']:.3f}"
            )

            keras.backend.clear_session()

    individual_df = pd.DataFrame(records)
    individual_path = ABLATION_DIR / "depth_ablation_individual.csv"
    individual_df.to_csv(individual_path, index=False)

    df = summarise_results(
        individual_df,
        "n_hidden_layers"
    )

    out_path = ABLATION_DIR / "depth_ablation.csv"
    df.to_csv(out_path, index=False)

    print(f"\n  Results saved → {out_path}")
    print()
    _print_results_table(df, "n_hidden_layers")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Study 2 — Anomaly Severity (Noise Scale)
# ═══════════════════════════════════════════════════════════════════════════════

NOISE_SWEEP = [0.5, 1.0, 1.8, 2.5, 4.0]


def study_noise_scale(X, preprocessor) -> pd.DataFrame:
    """Sweep the noise scale used during anomaly injection.

    The model is retrained once per seed with default settings; only the
    corruption changes across conditions. This isolates the effect of anomaly
    severity from any training-side confound.
    """

    _banner("DATA ABLATION — Anomaly Severity (Noise Scale)")
    print(f"  Sweep: {NOISE_SWEEP}")
    print(f"  Bottleneck dim: {config.BOTTLENECK_DIM}  [fixed]")
    print(f"  Network depth:  3  [fixed]\n")

    records = []

    for seed in SEEDS:
        print(f"\n  Seed {seed}")

        X_train, X_val, X_test = split_data(
            X, preprocessor, seed
        )

        # Train model ONCE for this seed with default architecture
        print(
            "  Training baseline autoencoder "
            "(used for all noise conditions) ...",
            end="",
            flush=True
        )

        model = train_autoencoder(
            X_train,
            bottleneck_dim=config.BOTTLENECK_DIM,
            n_hidden_layers=3,
            epochs=config.EPOCHS,
            batch_size=config.BATCH_SIZE,
            random_state=seed,
        )

        print("  done.\n")

        for ns in NOISE_SWEEP:
            print(
                f"  → noise_scale = {ns:.1f}  ...",
                end="",
                flush=True
            )

            X_val_corrupted, y_val = inject_anomalies(
                X_val,
                injection_frac=config.INJECTION_FRAC,
                noise_scale=ns,
                random_state=seed,
            )

            X_test_corrupted, y_test = inject_anomalies(
                X_test,
                injection_frac=config.INJECTION_FRAC,
                noise_scale=ns,
                random_state=seed + 1,
            )

            metrics = evaluate(
                model,
                X_val_corrupted,
                y_val,
                X_test_corrupted,
                y_test
            )

            records.append({
                "seed": seed,
                "noise_scale": ns,
                **metrics
            })

            print(
                f"  Prec={metrics['precision']:.3f}  "
                f"Rec={metrics['recall']:.3f}  "
                f"F1={metrics['f1']:.3f}  "
                f"AUC={metrics['roc_auc']:.3f}"
            )

        keras.backend.clear_session()

    individual_df = pd.DataFrame(records)
    individual_path = ABLATION_DIR / "noise_scale_ablation_individual.csv"
    individual_df.to_csv(individual_path, index=False)

    df = summarise_results(
        individual_df,
        "noise_scale"
    )

    out_path = ABLATION_DIR / "noise_scale_ablation.csv"
    df.to_csv(out_path, index=False)

    print(f"\n  Results saved → {out_path}")
    print()
    _print_results_table(df, "noise_scale")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Summary helper
# ═══════════════════════════════════════════════════════════════════════════════

def _save_summary(df_bottleneck, df_depth, df_noise) -> None:
    """Concatenate all results into a single summary CSV."""

    def _tag(df, study, varied):
        out = df.copy()
        out.insert(0, "study", study)
        out.insert(1, "varied_param", varied)
        out = out.rename(columns={varied: "param_value"})
        return out

    summary = pd.concat([
        _tag(
            df_bottleneck,
            "architectural_bottleneck",
            "bottleneck_dim"
        ),
        _tag(
            df_depth,
            "architectural_depth",
            "n_hidden_layers"
        ),
        _tag(
            df_noise,
            "data_noise_scale",
            "noise_scale"
        ),
    ], ignore_index=True)

    out_path = ABLATION_DIR / "ablation_summary.csv"
    summary.to_csv(out_path, index=False)

    print(f"\n  Full summary saved → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Run all ablation studies and save results."""

    print("\n" + "█" * 70)
    print("█  AUTOENCODER ABLATION STUDY")
    print("█  Studies: Bottleneck Dim | Network Depth | Noise Scale")
    print("█  Random Seeds: 41, 42, 43, 44, 45")
    print("█" * 70)

    # Fix global seeds
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    tf.random.set_seed(config.SEED)

    # ── Load data once ────────────────────────────────────────────────────────
    print("\n[Step 1/4]  Loading and preprocessing data ...")

    X, preprocessor = load_data(
        data_dir="data",
        results_dir="results"
    )

    if X is None:
        print(
            "\n[ABORT]  Data loading failed — "
            "cannot run ablation studies."
        )
        sys.exit(1)

    print(f"  Total samples : {X.shape[0]}")
    print(f"  Features      : {X.shape[1]}")
    print("  Split         : 60/15/25")
    print(f"  Seeds         : {SEEDS}")

    # ── Run studies ───────────────────────────────────────────────────────────
    print(
        "\n[Step 2/4]  "
        "Architectural Ablation — Bottleneck Dimension"
    )
    df_bottleneck = study_bottleneck(
        X,
        preprocessor
    )

    print(
        "\n[Step 3/4]  "
        "Architectural Ablation — Network Depth"
    )
    df_depth = study_depth(
        X,
        preprocessor
    )

    print(
        "\n[Step 4/4]  "
        "Data Ablation — Anomaly Severity (Noise Scale)"
    )
    df_noise = study_noise_scale(
        X,
        preprocessor
    )

    # ── Save combined summary ─────────────────────────────────────────────────
    _save_summary(
        df_bottleneck,
        df_depth,
        df_noise
    )

    print("\n" + "█" * 70)
    print("█  ALL ABLATION STUDIES COMPLETE ✓")
    print(f"█  Output directory: {ABLATION_DIR.resolve()}")
    print("█" * 70 + "\n")


if __name__ == "__main__":
    main()