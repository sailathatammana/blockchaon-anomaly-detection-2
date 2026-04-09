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


class AnomalyDetectionPipeline:
    """Complete pipeline for educational anomaly detection.
    
    This pipeline orchestrates the full workflow:
    1. Load and merge multiple educational datasets
    2. Preprocess data and create student performance profiles
    4. Trains an autoencoder model on clean data
    5. Injects synthetic anomalies to evaluate detection capability
    6. Finds optimal detection threshold to maximize F1 score
    7. Predicts which injected samples are anomalies
    8. Logs results to a permissioned blockchain for audit trail
    9. Saves performance metrics to JSON
    
    Attributes:
        data_dir: Path to input data directory
        results_dir: Path to save results and artifacts
        loader: EducationalDataLoader instance
        preprocessor: EducationalDataPreprocessor instance
    """
    
    def __init__(self, data_dir: str = "data", results_dir: str = "results"):
        """Initialize pipeline with directory paths.
        
        Args:
            data_dir: Directory containing source CSV files
            results_dir: Directory for output results and profiles
        """
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        
        # Create directories if they don't exist
        self.results_dir.mkdir(exist_ok=True)
        
        self.loader = None
        self.preprocessor = None
        self.detector = None
        self.blockchain = None
        self.results = {}
    
    def load_data(self) -> pd.DataFrame:
        """Load and merge multiple educational datasets.

        Prints summary statistics and merges datasets into a unified structure.
        
        Returns:
            DataFrame: Merged dataset combining all sources
        """
        print("\n" + "=" * 70)
        print("STEP 1: DATA LOADING")
        print("=" * 70)
        
        # Initialize loader and load all datasets from data directory
        self.loader = EducationalDataLoader(data_dir=str(self.data_dir))
        self.loader.load_all_datasets()
        self.loader.get_summary_statistics()
        
        # Create merged dataset combining all data sources
        merged_df = self.loader.create_merged_dataset()
        print(f"\n✓ Merged dataset shape: {merged_df.shape}")
        print(f"  Rows:    {merged_df.shape[0]} samples")
        print(f"  Columns: {merged_df.shape[1]} features")
        
        return merged_df
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, list]:
        """Preprocess data and create student profiles.
        
        Steps:
        1. Create individual student performance profiles from raw data
        2. prepare features for neural network input
        3. Save transformed data and metadata for reproducibility
        
        Args:
            df: Merged dataset from load_data()
        
        Returns:
            Tuple of:
                - X: Feature array ready for autoencoder (shape: n_students x n_features)
                - feature_names: List of feature names for interpretation
        """
        print("\n" + "=" * 70)
        print("STEP 2: DATA PREPROCESSING")
        print("=" * 70)
        
        # Initialize preprocessor with educational domain knowledge
        self.preprocessor = EducationalDataPreprocessor()
        
        # Create student performance profiles (aggregated metrics per student)
        profiles = self.preprocessor.create_student_profiles(df)
        print(f"\n✓ Student profiles created")
        print(f"  Shape: {profiles.shape}")
        print(f"  Students: {profiles.shape[0]}")
        print(f"  Features: {profiles.shape[1]}")
        
        # Save profiles to CSV for analysis and reproducibility
        profiles.to_csv(self.results_dir / "student_profiles.csv", index=False)
        print(f"\n✓ Profiles saved to {self.results_dir / 'student_profiles.csv'}")
        
        # Prepare features for autoencoder: handle missing values
        X, feature_names = self.preprocessor.prepare_for_autoencoder(profiles)
        print(f"\n✓ Features prepared for autoencoder")
        print(f"  Input shape: {X.shape}")
        print(f"  Num features: {len(feature_names)}")

        # Save feature names for model interpretability
        with open(self.results_dir / "feature_names.json", 'w') as f:
            json.dump(feature_names, f, indent=2)
        print(f"✓ Feature names saved to {self.results_dir / 'feature_names.json'}")

        return X, feature_names
    
    def split_data(self, X: np.ndarray, test_size: float = config.TEST_SIZE, 
                   random_state: int = config.SEED) -> Tuple[np.ndarray, np.ndarray]:
        """Split data into train and test sets.
        
        Separate data: 70% for training the autoencoder, 30% for evaluation.
        We'll inject anomalies into the test set to simulate real-world scenarios.
        
        Args:
            X: Feature array from preprocessing
            test_size: Proportion of data for testing (default 0.30 = 30%)
            random_state: Random seed for reproducibility
        
        Returns:
            Tuple of:
                - X_train: Training set features (70%)
                - X_test: Test set features (30%)
        """
        print("\n" + "=" * 70)
        print("STEP 3: TRAIN/TEST SPLIT")
        print("=" * 70)
        
        # Split into train/test with stratification for balanced sets
        X_train, X_test = train_test_split(
            X, test_size=test_size, random_state=random_state
        )
        
        print(f"\n✓ Data split completed (70/30)")
        print(f"  Training set:   {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
        print(f"  Test set:       {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
        
        return X_train, X_test
    
    def run_pipeline(self):
        """Run complete anomaly detection pipeline.
        
        Orchestrates the full workflow:
        1. Load and explore data from multiple CSV sources
        2. Preprocess and create student performance profiles
        3. Prepare features for anomaly detection model
        
        Handles errors gracefully with detailed logging.
        """
        print("\n" + "█" * 70)
        print("█ EDUCATIONAL ANOMALY DETECTION PIPELINE")
        print("█" * 70)
        
        try:
            # Load data from CSV files
            df = self.load_data()
            
            # Preprocess and create profiles
            X, feature_names = self.preprocess_data(df)
            
            # Split into train/test sets
            X_train, X_test = self.split_data(X, test_size=config.TEST_SIZE, random_state=config.SEED)
            
            print("\n" + "=" * 70)
            print("PIPELINE STATUS: ✓ COMPLETE")
            print("=" * 70)
            print(f"Next step: Train autoencoder on {X_train.shape[0]} student profiles")
            print(f"Evaluate on {X_test.shape[0]} samples\n")

            return X_train, X_test, feature_names
            
        except Exception as e:
            print("\n" + "=" * 70)
            print("PIPELINE STATUS: ✗ ERROR")
            print("=" * 70)
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            return None


if __name__ == "__main__":
    # Initialize and run pipeline
    # This will load educational data, create student profiles,
    # and prepare features for anomaly detection training
    pipeline = AnomalyDetectionPipeline(
        data_dir="data",
        results_dir="results"
    )    
    X_train, X_test, feature_names = pipeline.run_pipeline()
