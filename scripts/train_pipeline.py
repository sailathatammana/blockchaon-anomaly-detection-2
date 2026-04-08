"""
Main training and evaluation pipeline for educational anomaly detection
Orchestrates data loading, preprocessing, model training, and evaluation
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
import json

# Import project modules
from data_loader import EducationalDataLoader
from preprocessing import EducationalDataPreprocessor


class AnomalyDetectionPipeline:
    """Complete pipeline for anomaly detection"""
    
    def __init__(self, data_dir: str = "data", results_dir: str = "results",
                 models_dir: str = "models"):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.models_dir = Path(models_dir)
        
        # Create directories
        self.results_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
        self.loader = None
        self.preprocessor = None
        self.detector = None
        self.blockchain = None
        self.results = {}
    
    def load_data(self) -> pd.DataFrame:
        """Load and merge datasets"""
        print("\n" + "="*60)
        print("STEP 1: DATA LOADING")
        print("="*60)
        
        self.loader = EducationalDataLoader(data_dir=str(self.data_dir))
        self.loader.load_all_datasets()
        self.loader.get_summary_statistics()
        
        merged_df = self.loader.create_merged_dataset()
        print(f"\nMerged dataset shape: {merged_df.shape}")
        
        return merged_df
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, list]:
        """Preprocess and prepare data"""
        print("\n" + "="*60)
        print("STEP 2: DATA PREPROCESSING")
        print("="*60)
        
        self.preprocessor = EducationalDataPreprocessor()
        
        # Create student profiles
        profiles = self.preprocessor.create_student_profiles(df)
        print(f"Student profiles shape: {profiles.shape}")
        
        # Save profiles
        profiles.to_csv(self.results_dir / "student_profiles.csv", index=False)
        print(f"Profiles saved to {self.results_dir / 'student_profiles.csv'}")
        
        # Prepare for autoencoder
        X, feature_names = self.preprocessor.prepare_for_autoencoder(profiles)
        print(f"Input features shape: {X.shape}")
        print(f"Feature names: {feature_names}")
        
        # Save feature names
        with open(self.results_dir / "feature_names.json", 'w') as f:
            json.dump(feature_names, f)

        return X, feature_names, profiles
    
    
    def run_pipeline(self):
        """Run complete pipeline"""
        print("\n" + "█"*60)
        print("█ EDUCATIONAL ANOMALY DETECTION PIPELINE")
        print("█"*60)
        
        try:
            # Load data
            df = self.load_data()
            
            # Preprocess
            self.preprocess_data(df)
            
        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()
            return None, None


if __name__ == "__main__":    
    # Run pipeline
    pipeline = AnomalyDetectionPipeline(
        data_dir="data",
        results_dir="results",
        models_dir="models"
    )
    
    pipeline.run_pipeline()
