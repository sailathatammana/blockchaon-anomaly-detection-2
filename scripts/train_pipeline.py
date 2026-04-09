"""Main training and evaluation pipeline for educational anomaly detection.

Orchestrates data loading, preprocessing, model training, and evaluation.
Implements reproducible results through deterministic random seeding and
environment configuration.
"""
import os
import sys

# Set random seeds FIRST, before any TensorFlow imports
# These environment variables ensure reproducible results across runs:
# - PYTHONHASHSEED: Makes Python's hash function deterministic
# - TF_CPP_MIN_LOG_LEVEL: Reduces TensorFlow logging noise (2 = warnings only)
# - TF_DETERMINISTIC_OPS: Forces TensorFlow to use deterministic algorithms
os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import json
import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from sklearn.model_selection import train_test_split

import tensorflow as tf

# Import project modules
import config
from data_loader import EducationalDataLoader
from preprocessing import EducationalDataPreprocessor

# Set seeds after imports for all libraries
# This ensures reproducible random behavior in data splitting, model initialization, etc.
random.seed(config.SEED)
np.random.seed(config.SEED)
tf.random.set_seed(config.SEED)


def main(data_dir: str = "data", results_dir: str = "results"):
    """Run complete anomaly detection pipeline.
    
    Orchestrates the full workflow:
    1. Load and explore data from multiple CSV sources
    2. Preprocess and create student performance profiles
    3. Prepare features for anomaly detection model
    4. Split data into train/test sets
    
    Args:
        data_dir: Directory containing source CSV files
        results_dir: Directory for output results and profiles
    
    Returns:
        Tuple of:
            - X_train: Training set features (70%)
            - X_test: Test set features (30%)
            - feature_names: List of feature names for interpretation
    """
    # Initialize paths
    data_path = Path(data_dir)
    results_path = Path(results_dir)
    results_path.mkdir(exist_ok=True)
    
    print("\n" + "█" * 70)
    print("█ EDUCATIONAL ANOMALY DETECTION PIPELINE")
    print("█" * 70)
    
    try:
        # STEP 1: LOAD DATA
        # Read raw data, handle missing values, encode categorical features
        print("\n" + "=" * 70)
        print("STEP 1: DATA LOADING")
        print("=" * 70)
        
        loader = EducationalDataLoader(data_dir=str(data_path))
        loader.load_all_datasets()
        loader.get_summary_statistics()
        
        merged_df = loader.create_merged_dataset()
        print(f"\n✓ Merged dataset shape: {merged_df.shape}")
        print(f"  Rows:    {merged_df.shape[0]} samples")
        print(f"  Columns: {merged_df.shape[1]} features")
        
        # STEP 2: PREPROCESS DATA
        # Create student profiles and prepare features for autoencoder
        print("\n" + "=" * 70)
        print("STEP 2: DATA PREPROCESSING")
        print("=" * 70)
        
        preprocessor = EducationalDataPreprocessor()
        
        # Create student performance profiles (aggregated metrics per student)
        profiles = preprocessor.create_student_profiles(merged_df)
        print(f"\n✓ Student profiles created")
        print(f"  Shape: {profiles.shape}")
        print(f"  Students: {profiles.shape[0]}")
        print(f"  Features: {profiles.shape[1]}")
        
        # Save profiles to CSV for analysis and reproducibility
        profiles.to_csv(results_path / "student_profiles.csv", index=False)
        print(f"\n✓ Profiles saved to {results_path / 'student_profiles.csv'}")
        
        # Prepare features for autoencoder: handle missing values
        X, feature_names = preprocessor.prepare_for_autoencoder(profiles)
        print(f"\n✓ Features prepared for autoencoder")
        print(f"  Input shape: {X.shape}")
        print(f"  Num features: {len(feature_names)}")

        # Save feature names for model interpretability
        with open(results_path / "feature_names.json", 'w') as f:
            json.dump(feature_names, f, indent=2)
        print(f"✓ Feature names saved to {results_path / 'feature_names.json'}")
        
        # STEP 3: SPLIT INTO TRAIN/TEST
        # Separate data: 70% for training the autoencoder, 30% for evaluation.
        # We'll inject anomalies into the test set to simulate real-world scenarios.
        print("\n" + "=" * 70)
        print("STEP 3: TRAIN/TEST SPLIT")
        print("=" * 70)
        
        X_train, X_test = train_test_split(
            X, test_size=config.TEST_SIZE, random_state=config.SEED
        )
        
        print(f"\n✓ Data split completed ({(1-config.TEST_SIZE)*100:.0f}/{config.TEST_SIZE*100:.0f})")
        print(f"  Training set:   {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
        print(f"  Test set:       {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
        
        # STEP 4: NORMALIZE FEATURES
        # Normalize features to [0, 1] range. Fit scaler ONLY on training data
        # to prevent data leakage, then apply to test data.
        print("\n" + "=" * 70)
        print("STEP 4: FEATURE NORMALIZATION")
        print("=" * 70)
        
        scaler, X_train_scaled = preprocessor.normalize_features(X_train)
        X_test_scaled = scaler.transform(X_test)
        print(pd.DataFrame(X_train_scaled, columns=feature_names).head())
        
        print(f"\n✓ Features normalized to [0, 1] range")
        print(f"  Training set shape:   {X_train_scaled.shape}")
        print(f"  Test set shape:       {X_test_scaled.shape}")
        
        print("\n" + "=" * 70)
        print("PIPELINE STATUS: ✓ COMPLETE")
        print("=" * 70)
        print(f"Next step: Train autoencoder on {X_train_scaled.shape[0]} student profiles")
        print(f"Evaluate on {X_test_scaled.shape[0]} normalized samples\n")

        return X_train_scaled, X_test_scaled, feature_names, scaler
            
    except Exception as e:
        print("\n" + "=" * 70)
        print("PIPELINE STATUS: ✗ ERROR")
        print("=" * 70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


if __name__ == "__main__":
    # Run the complete pipeline
    X_train_scaled, X_test_scaled, feature_names, scaler = main(
        data_dir="data",
        results_dir="results"
    )
