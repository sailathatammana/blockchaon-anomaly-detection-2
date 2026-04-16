"""
Data Preprocessing Module for Educational Anomaly Detection
Handles normalization, feature engineering, and data preparation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class EducationalDataPreprocessor:
    """Preprocess educational data for autoencoder training"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_names = []
        self.numeric_features = []
        self.categorical_features = []
        
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features"""
        df_encoded = df.copy()
        
        # Find all text/object columns
        cat_cols = df_encoded.select_dtypes(include=["object"]).columns.tolist()
        # Expand each categorical to binary columns, drop first to avoid redundancy
        df_encoded = pd.get_dummies(df_encoded, columns=cat_cols, drop_first=True)
        
        return df_encoded, cat_cols
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in numeric columns"""
        df_filled = df.copy()
        # Identify column types
        num_cols = df_filled.select_dtypes(include=["int64", "float64"]).columns
        cat_cols = df_filled.select_dtypes(include=["object"]).columns
        
        for col in num_cols:
            df_filled[col] = df_filled[col].fillna(df_filled[col].median())

        # Fill categorical columns with mode (most common value) of that column
        for col in cat_cols:
            df_filled[col] = df_filled[col].fillna(df_filled[col].mode()[0])
        
        return df_filled
    
    def normalize_features(self, X: pd.DataFrame) -> Tuple[object, pd.DataFrame]:
        """Standardize features to zero mean, unit variance.
        
        Applies StandardScaler transformation: x_scaled = (x - mean) / std_dev
        
        Why normalize?
        - Neural networks converge much faster on normalized data
        - Prevents features with large ranges from dominating (e.g., age vs. income)
        - Makes learning rate more stable across all features
        
        Example:
        Before: GPA ∈ [0,4], Study_Hours ∈ [0,200]
        → Study_Hours has 50× larger range, dominates learning
        After: GPA ∈ [-1.5, 2.5], Study_Hours ∈ [-1.2, 2.8]
        → Both features have similar scale, equal importance
        
        CRITICAL: fit_transform(X_train) then transform(X_test)
        - Never fit on test set (prevents data leakage)
        - Both use same mean/std from training distribution
        
        Args:
            X: Features to normalize (typically training data)
        
        Returns:
            Tuple of:
                - scaler: Fitted StandardScaler (save to apply transformation to test data)
                - X_scaled: Normalized features as numpy array (mean≈0, std≈1)
        """
        # Create new scaler instance
        self.scaler = StandardScaler()
        # fit_transform: learn mean/std from X, then apply transformation
        X_scaled = self.scaler.fit_transform(X)
        return self.scaler, X_scaled

    def create_student_profiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create student engagement and performance profiles
        Extract meaningful features for anomaly detection
        """
        profiles = pd.DataFrame()
        
        # Handle ID columns
        if 'id_student' in df.columns:
            profiles['id_student'] = df['id_student']
        
        # Demographics (categorical)
        if 'gender' in df.columns:
            profiles['gender'] = df['gender']
        if 'disability' in df.columns:
            profiles['disability'] = df['disability']
        if 'highest_education' in df.columns:
            profiles['highest_education'] = df['highest_education']
        if 'region' in df.columns:
            profiles['region'] = df['region']
        
        # Engagement metrics (numeric)
        if 'studied_credits' in df.columns:
            profiles['studied_credits'] = df['studied_credits']
        if 'num_of_prev_attempts' in df.columns:
            profiles['prev_attempts'] = df['num_of_prev_attempts']
        if 'score_mean' in df.columns:
            profiles['avg_score'] = df['score_mean']
            profiles['score_std'] = df['score_std'].fillna(0)
            profiles['assessment_count'] = df['assessment_count']
        
        # Age group (encode if present)
        if 'age_band' in df.columns:
            age_order = ['0-35', '35-55', '55<=']
            age_numeric = df['age_band'].map({
                '0-35': 1,
                '35-55': 2,
                '55<=': 3
            }).fillna(1)
            profiles['age_group'] = age_numeric
        
        # VLE engagement metrics (if available)
        if 'total_clicks' in df.columns:
            profiles['total_clicks'] = df['total_clicks'].fillna(0)
        if 'avg_clicks_per_activity' in df.columns:
            profiles['avg_clicks_per_activity'] = df['avg_clicks_per_activity'].fillna(0)
        if 'std_clicks' in df.columns:
            profiles['std_clicks'] = df['std_clicks'].fillna(0)
        if 'num_interactions' in df.columns:
            profiles['num_interactions'] = df['num_interactions'].fillna(0)
        if 'unique_sites' in df.columns:
            profiles['unique_sites'] = df['unique_sites'].fillna(0)
        if 'first_access_day' in df.columns:
            profiles['first_access_day'] = df['first_access_day'].fillna(0)
        if 'last_access_day' in df.columns:
            profiles['last_access_day'] = df['last_access_day'].fillna(0)
        
        # Target variable
        if 'final_result' in df.columns:
            result_map = {
                'Pass': 0,
                'Fail': 1,
                'Withdrawn': 2,
                'Distinction': 0
            }
            profiles['final_result'] = df['final_result'].map(result_map).fillna(0).astype(int)
        
        return profiles
    
    def prepare_for_autoencoder(self, df: pd.DataFrame, 
                                exclude_cols: list = None) -> Tuple[np.ndarray, list]:
        """
        Prepare data for autoencoder training
        Returns feature matrix and feature names
        """
        df_features = df.copy()
        
        # Encode categorical features
        df_encoded, cat_cols = self.encode_categorical_features(df_features, fit=True)
        
        # Handle missing values
        df_filled = self.handle_missing_values(df_encoded)

        if exclude_cols is None:
            exclude_cols = ['id_student', 'code_module', 'code_presentation', 
                           'final_result']
        
        # Select features
        feature_cols = [col for col in df_filled.columns if col not in exclude_cols]
        df_features = df_filled[feature_cols].copy()
                
        self.feature_names = feature_cols
        
        return  df_features,feature_cols
    

if __name__ == "__main__":
    from data_loader import EducationalDataLoader
    
    # Load data
    loader = EducationalDataLoader(data_dir="data")
    merged_df = loader.create_merged_dataset()
    
    # Preprocess
    preprocessor = EducationalDataPreprocessor()
    profiles = preprocessor.create_student_profiles(merged_df)
    
    print("\nStudent Profiles Shape:", profiles.shape)
    print("\nProfile Columns:", list(profiles.columns))
    print("\nFirst few profiles:")
    print(profiles.head())
    
    # Prepare for autoencoder
    X, feature_names = preprocessor.prepare_for_autoencoder(profiles)
    print(f"\nAutoencoder input shape: {X.shape}")
    print(f"Feature names: {feature_names}")
