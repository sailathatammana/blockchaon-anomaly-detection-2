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
        
        # Target variable - KEEP AS STRING for labeling function
        if 'final_result' in df.columns:
            profiles['final_result'] = df['final_result'].astype(str)
        
        return profiles
    
    def prepare_for_autoencoder(self, df: pd.DataFrame, 
                                exclude_cols: list = None) -> Tuple[np.ndarray, list]:
        """
        Prepare data for autoencoder training
        Returns feature matrix (as numpy array) and feature names
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
        
        # Convert to numpy array for compatibility with sklearn operations
        X = df_features.values.astype(np.float32)
        
        return X, feature_cols
    
    @staticmethod
    def create_anomaly_labels(profiles: pd.DataFrame, 
                            normal_labels: set = None,
                            abnormal_labels: set = None) -> Tuple[np.ndarray, dict]:
        """Create binary anomaly labels from student outcomes.
        
        REAL DATA LABELING APPROACH:
        Uses actual student outcomes (final_result) instead of synthetic anomalies.
        
        Args:
            profiles: DataFrame with 'final_result' column
            normal_labels: Set of outcomes considered normal (e.g., {'Pass', 'Distinction'})
            abnormal_labels: Set of outcomes considered abnormal (e.g., {'Fail', 'Withdrawn'})
        
        Returns:
            y: Array where 1 = normal, -1 = abnormal, 0 = unknown
            label_info: Dictionary with statistics about label distribution
        """
        if normal_labels is None:
            normal_labels = {'Pass', 'Distinction'}
        if abnormal_labels is None:
            abnormal_labels = {'Fail', 'Withdrawn'}
        
        def label_outcome(outcome):
            """Map outcome to label"""
            if outcome in normal_labels:
                return 1
            elif outcome in abnormal_labels:
                return -1
            else:
                return 0  # Unknown/other
        
        y = profiles['final_result'].apply(label_outcome).values
        
        # Generate label statistics (convert numpy ints to Python ints for JSON serialization)
        n_normal = int(np.sum(y == 1))
        n_abnormal = int(np.sum(y == -1))
        n_unknown = int(np.sum(y == 0))
        
        label_info = {
            'n_normal': n_normal,
            'n_abnormal': n_abnormal,
            'n_unknown': n_unknown,
            'pct_normal': float(100 * n_normal / (n_normal + n_abnormal + n_unknown) if (n_normal + n_abnormal + n_unknown) > 0 else 0),
            'pct_abnormal': float(100 * n_abnormal / (n_normal + n_abnormal + n_unknown) if (n_normal + n_abnormal + n_unknown) > 0 else 0),
            'contamination_rate': float(n_abnormal / (n_normal + n_abnormal) if (n_normal + n_abnormal) > 0 else 0)
        }
        
        return y, label_info
    
    @staticmethod
    def split_normal_abnormal(X: np.ndarray, y: np.ndarray, 
                            normal_train_fraction: float = 0.75,
                            random_state: int = 41) -> Tuple[np.ndarray, np.ndarray, 
                                                             np.ndarray, np.ndarray,
                                                             np.ndarray, np.ndarray]:
        """Split data into training (normal only) and test (normal + abnormal).
        
        CRITICAL: This implements the core strategy for real data anomaly detection.
        
        Strategy:
        1. Separate normal (y=1) and abnormal (y=-1) students
        2. Train autoencoder on: normal_train_fraction of NORMAL students
        3. Test autoencoder on: remaining normal students + ALL abnormal students
        
        This ensures autoencoder only learns "normal" patterns during training,
        then faces true abnormal cases during evaluation.
        
        Args:
            X: Feature matrix
            y: Label array (1=normal, -1=abnormal)
            normal_train_fraction: Fraction of normal students to use for training (e.g., 0.75)
            random_state: Random seed for reproducibility
        
        Returns:
            X_train: Training features (normal students only)
            X_test: Test features (normal + abnormal students)
            y_train: Training labels (all 1, since only normal)
            y_test: Test labels (1 for normal, -1 for abnormal in test set)
            train_indices: Indices of training samples in original data
            test_indices: Indices of test samples in original data
        """
        np.random.seed(random_state)
        
        # Separate normal and abnormal indices
        normal_idx = np.where(y == 1)[0]
        abnormal_idx = np.where(y == -1)[0]
        
        # Split normal students: train/test
        n_normal_train = int(len(normal_idx) * normal_train_fraction)
        np.random.shuffle(normal_idx)
        
        train_normal_idx = normal_idx[:n_normal_train]
        test_normal_idx = normal_idx[n_normal_train:]
        
        # Combine test sets: normal + abnormal
        test_all_idx = np.concatenate([test_normal_idx, abnormal_idx])
        
        # Extract features and labels
        X_train = X[train_normal_idx]
        X_test = X[test_all_idx]
        
        # Ensure proper data types for labels
        y_train = np.ones(len(train_normal_idx), dtype=int)  # All normal in training (+1)
        y_test = y[test_all_idx].astype(int)  # Mix of normal (1) and abnormal (-1) in test
        
        return X_train, X_test, y_train, y_test, train_normal_idx, test_all_idx
    

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
