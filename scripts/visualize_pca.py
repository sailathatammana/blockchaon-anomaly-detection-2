"""2D PCA visualization of normal vs abnormal students.

This script creates a 2D PCA representation of students colored by their 
anomaly status (normal vs abnormal) based on real student outcomes.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Import our modules
import config
from pipeline import load_and_preprocess_data


def create_pca_visualization(X_scaled, y_labels, feature_names, output_dir="results", title="PCA: Normal vs Abnormal Students"):
    """Create and save 2D PCA visualization.
    
    Args:
        X_scaled: Scaled feature matrix (n_samples, n_features)
        y_labels: Anomaly labels (+1=normal, -1=abnormal)
        feature_names: List of feature names
        output_dir: Directory to save visualizations
        title: Title for the plot
    
    Returns:
        Dictionary with PCA results and variance explained
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("\n" + "=" * 80)
    print("PCA VISUALIZATION: NORMAL vs ABNORMAL STUDENTS")
    print("=" * 80)
    
    # Apply PCA to reduce to 2D
    print("\n✓ Applying PCA to reduce to 2 dimensions...")
    pca = PCA(n_components=2, random_state=config.SEED)
    X_pca = pca.fit_transform(X_scaled)
    
    variance_explained = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_explained)
    
    print(f"  PC1 variance explained: {variance_explained[0]:.4f} ({variance_explained[0]*100:.2f}%)")
    print(f"  PC2 variance explained: {variance_explained[1]:.4f} ({variance_explained[1]*100:.2f}%)")
    print(f"  Total variance explained: {cumulative_variance[1]:.4f} ({cumulative_variance[1]*100:.2f}%)")
    
    # Prepare data for plotting
    df_pca = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Label': y_labels
    })
    
    # Count normal and abnormal
    n_normal = np.sum(y_labels == 1)
    n_abnormal = np.sum(y_labels == -1)
    
    print(f"\n✓ Data distribution:")
    print(f"  Normal students: {n_normal}")
    print(f"  Abnormal students: {n_abnormal}")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Plot normal and abnormal students with different colors
    normal_mask = y_labels == 1
    abnormal_mask = y_labels == -1
    
    ax.scatter(X_pca[normal_mask, 0], X_pca[normal_mask, 1], 
               c='#2ecc71', label=f'Normal Students (n={n_normal})', 
               alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    ax.scatter(X_pca[abnormal_mask, 0], X_pca[abnormal_mask, 1],
               c='#e74c3c', label=f'Abnormal Students (n={n_abnormal})',
               alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    
    # Customize plot
    ax.set_xlabel(f'First Principal Component ({variance_explained[0]*100:.2f}% variance)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Second Principal Component ({variance_explained[1]*100:.2f}% variance)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add background color
    ax.set_facecolor('#f8f9fa')
    fig.patch.set_facecolor('white')
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    output_file = output_path / "pca_visualization.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to {output_file}")
    
    plt.show()
    


def main(data_dir: str = "data", results_dir: str = "results"):
    """Main execution function.
    
    Args:
        data_dir: Directory containing source CSV files
        results_dir: Directory for output results
    """
    print("\n" + "█" * 80)
    print("█ 2D PCA VISUALIZATION OF STUDENTS")
    print("█" * 80)
    
    # Load and preprocess data
    print("\n✓ Loading and preprocessing data...")
    X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler, label_info, profiles_labeled, test_idx = \
        load_and_preprocess_data(data_dir=data_dir, results_dir=results_dir)
    
    if X_train_scaled is None:
        print("❌ Data loading failed. Stopping.")
        return
    
    # Also create separate visualizations for test set only
    print("\n" + "=" * 80)
    print("CREATING PCA VISUALIZATION (TEST SET ONLY)")
    print("=" * 80)
    
    print(f"\nTest dataset size: {X_test_scaled.shape}")
    print(f"  Normal: {np.sum(y_test == 1)}")
    print(f"  Abnormal: {np.sum(y_test == -1)}")
    
    pca_results_test = create_pca_visualization(
        X_test_scaled,
        y_test,
        feature_names,
        output_dir=results_dir,
        title="2D PCA: Test Set\n(Normal vs Abnormal Students)"
    )
    
    print("\n" + "=" * 80)
    print("✓ PCA VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\nOutputs saved to: {results_dir}/")
    print("  - pca_visualization_test.png (test set only)")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main(data_dir="data", results_dir="results")
