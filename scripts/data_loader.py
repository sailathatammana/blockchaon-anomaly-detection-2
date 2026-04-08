"""
Data Loading Module for Educational Anomaly Detection Project
Loads and merges multiple educational datasets
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

class EducationalDataLoader:
    """Load and merge educational datasets"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.datasets: Dict[str, pd.DataFrame] = {}
        
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """Load all CSV datasets"""
        dataset_names = [
            'studentInfo',
            'studentRegistration',
            'studentAssessment',
            'studentVle',
            'assessments',
            'vle',
            'courses'
        ]
        
        for name in dataset_names:
            file_path = self.data_dir / f"{name}.csv"
            if file_path.exists():
                print(f"Loading {name}...")
                self.datasets[name] = pd.read_csv(file_path)
                print(f"  Shape: {self.datasets[name].shape}")
            else:
                print(f"Warning: {file_path} not found")
                
        return self.datasets
    
    def create_merged_dataset(self) -> pd.DataFrame:
        """
        Create a comprehensive merged dataset for analysis
        Combines student info, assessment, and engagement data
        """
        if not self.datasets:
            self.load_all_datasets()
        
        # Start with student info
        merged = self.datasets['studentInfo'].copy()
        
        # Add registration info
        if 'studentRegistration' in self.datasets:
            reg = self.datasets['studentRegistration'].copy()
            reg = reg.groupby('id_student').agg({
                'date_registration': 'first',
                'date_unregistration': 'first'
            }).reset_index()
            merged = merged.merge(reg, on='id_student', how='left')
        
        # Add assessment scores (aggregated)
        if 'studentAssessment' in self.datasets:
            assessments = self.datasets['studentAssessment'].copy()
            
            # Merge with assessment metadata
            if 'assessments' in self.datasets:
                assessments = assessments.merge(
                    self.datasets['assessments'][['id_assessment', 'assessment_type']],
                    on='id_assessment',
                    how='left'
                )
            
            # Aggregate scores by student
            score_agg = assessments.groupby('id_student').agg({
                'score': ['mean', 'std', 'min', 'max', 'count'],
                'is_banked': 'sum'
            }).reset_index()
            
            score_agg.columns = ['id_student', 'score_mean', 'score_std', 
                                 'score_min', 'score_max', 'assessment_count', 'banked_count']
            
            merged = merged.merge(score_agg, on='id_student', how='left')
        
        # Add VLE engagement metrics (if available)
        if 'studentVle' in self.datasets:
            vle_data = self.datasets['studentVle'].copy()
            
            # Aggregate VLE engagement by student
            vle_agg = vle_data.groupby('id_student').agg({
                'sum_click': ['sum', 'mean', 'std', 'count'],
                'id_site': 'nunique',
                'date': ['min', 'max']
            }).reset_index()
            
            vle_agg.columns = ['id_student', 'total_clicks', 'avg_clicks_per_activity',
                              'std_clicks', 'num_interactions', 'unique_sites',
                              'first_access_day', 'last_access_day']
            
            merged = merged.merge(vle_agg, on='id_student', how='left')
        
        # Fill NaN values
        merged = merged.fillna(0)
        
        return merged
    
    def get_time_series_data(self) -> pd.DataFrame:
        """
        Create time-series data for anomaly detection
        Combines studentVle engagement over time
        """
        if not self.datasets:
            self.load_all_datasets()
        
        if 'studentVle' not in self.datasets:
            return None
            
        student_vle = self.datasets['studentVle'].copy()
        
        # Group by student and date for time-series
        ts_data = student_vle.groupby(['id_student', 'date']).agg({
            'sum_click': 'sum',
            'id_site': 'count'
        }).reset_index()
        
        ts_data.columns = ['id_student', 'date', 'total_clicks', 'sites_accessed']
        
        return ts_data
    
    def get_summary_statistics(self):
        """Print summary statistics of all datasets"""
        if not self.datasets:
            self.load_all_datasets()
        
        print("\n" + "="*60)
        print("DATASET SUMMARY STATISTICS")
        print("="*60)
        
        for name, df in self.datasets.items():
            print(f"\n{name.upper()}:")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            print(f"  Missing values: {df.isnull().sum().sum()}")
            
            if name == 'studentInfo':
                print(f"  Unique modules: {df['code_module'].nunique()}")
                print(f"  Unique presentations: {df['code_presentation'].nunique()}")
                print(f"  Total students: {df['id_student'].nunique()}")
                

if __name__ == "__main__":
    loader = EducationalDataLoader(data_dir="data")
    datasets = loader.load_all_datasets()
    loader.get_summary_statistics()
    
    merged = loader.create_merged_dataset()
    print(f"\nMerged dataset shape: {merged.shape}")
    print(f"\nFirst few rows:\n{merged.head()}")
