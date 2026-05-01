"""Analyze and visualize ablation study results.

This script loads the ablation study results and generates:
1. Comparison charts for each component
2. Heatmap showing metric relationships
3. Detailed analysis report
4. Recommendations based on findings
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 10)
plt.rcParams["font.size"] = 10


def load_results(results_path: Path) -> Dict[str, Any]:
    """Load ablation study results from JSON file."""
    with open(results_path, "r") as f:
        return json.load(f)


def extract_comparison_data(results: Dict[str, Any]) -> pd.DataFrame:
    """Extract all variants into a single DataFrame for analysis."""
    rows = []
    
    # Add baseline
    baseline = results["baseline"]
    baseline_config = baseline["config"]
    row = {
        "variant": "baseline",
        "category": "baseline",
        **baseline_config,
        **{k: v for k, v in baseline.items() if k not in ["config"]}
    }
    rows.append(row)
    
    # Add all variations
    for category, variants in results["variations"].items():
        for variant_name, variant_data in variants.items():
            config = variant_data["config"]
            row = {
                "variant": variant_name,
                "category": category,
                **config,
                **{k: v for k, v in variant_data.items() if k not in ["config"]}
            }
            rows.append(row)
    
    return pd.DataFrame(rows)


def plot_component_comparison(df: pd.DataFrame, results_dir: Path):
    """Create comparison charts for each ablated component."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Ablation Study: Component Impact on Performance", 
                 fontsize=16, fontweight="bold", y=1.00)
    
    # 1. Bottleneck Dimension
    ax = axes[0, 0]
    bottleneck_df = df[df["category"] == "bottleneck_dimension"].copy()
    bottleneck_df = bottleneck_df.sort_values("bottleneck_dim")
    ax.plot(bottleneck_df["bottleneck_dim"], bottleneck_df["f1"], 
            marker="o", linewidth=2, markersize=8, label="F1 Score", color="#1f77b4")
    ax.plot(bottleneck_df["bottleneck_dim"], bottleneck_df["roc_auc"], 
            marker="s", linewidth=2, markersize=8, label="ROC-AUC", color="#ff7f0e")
    ax.set_xlabel("Bottleneck Dimension")
    ax.set_ylabel("Score")
    ax.set_title("Bottleneck Dimension Impact")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(bottleneck_df["bottleneck_dim"].unique())
    
    # 2. Network Depth
    ax = axes[0, 1]
    depth_df = df[df["category"] == "network_depth"].copy()
    depth_df = depth_df.sort_values("depth")
    ax.plot(depth_df["depth"], depth_df["f1"], 
            marker="o", linewidth=2, markersize=8, label="F1 Score", color="#1f77b4")
    ax.plot(depth_df["depth"], depth_df["roc_auc"], 
            marker="s", linewidth=2, markersize=8, label="ROC-AUC", color="#ff7f0e")
    ax.set_xlabel("Network Depth (# Hidden Layers)")
    ax.set_ylabel("Score")
    ax.set_title("Network Depth Impact")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(depth_df["depth"].unique())
    
    # 3. Dropout Rate
    ax = axes[0, 2]
    dropout_df = df[df["category"] == "dropout_rate"].copy()
    dropout_df = dropout_df.sort_values("dropout_rate")
    ax.plot(dropout_df["dropout_rate"], dropout_df["f1"], 
            marker="o", linewidth=2, markersize=8, label="F1 Score", color="#1f77b4")
    ax.plot(dropout_df["dropout_rate"], dropout_df["roc_auc"], 
            marker="s", linewidth=2, markersize=8, label="ROC-AUC", color="#ff7f0e")
    ax.set_xlabel("Dropout Rate")
    ax.set_ylabel("Score")
    ax.set_title("Dropout Rate Impact")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Activation Function
    ax = axes[1, 0]
    activation_df = df[df["category"] == "activation_function"].copy()
    activations = activation_df["activation"].values
    x_pos = np.arange(len(activations))
    width = 0.35
    ax.bar(x_pos - width/2, activation_df["f1"], width, label="F1 Score", alpha=0.8)
    ax.bar(x_pos + width/2, activation_df["roc_auc"], width, label="ROC-AUC", alpha=0.8)
    ax.set_xlabel("Activation Function")
    ax.set_ylabel("Score")
    ax.set_title("Activation Function Impact")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(activations)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim([0.8, 1.0])
    
    # 5. Feature Subset
    ax = axes[1, 1]
    feature_df = df[df["category"] == "feature_subset"].copy()
    feature_df = feature_df.sort_values("feature_subset_pct", ascending=False)
    x_pos = np.arange(len(feature_df))
    width = 0.35
    labels = [f"{int(pct*100)}%" for pct in feature_df["feature_subset_pct"]]
    ax.bar(x_pos - width/2, feature_df["f1"], width, label="F1 Score", alpha=0.8)
    ax.bar(x_pos + width/2, feature_df["roc_auc"], width, label="ROC-AUC", alpha=0.8)
    ax.set_xlabel("Feature Subset Percentage")
    ax.set_ylabel("Score")
    ax.set_title("Feature Subset Impact")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim([0.8, 1.0])
    
    # 6. Key Metrics Comparison
    ax = axes[1, 2]
    baseline_row = df[df["variant"] == "baseline"].iloc[0]
    metrics = ["f1", "precision", "recall", "roc_auc"]
    values = [baseline_row[m] for m in metrics]
    colors = plt.cm.Set3(np.linspace(0, 1, len(metrics)))
    bars = ax.barh(metrics, values, color=colors)
    ax.set_xlabel("Score")
    ax.set_title("Baseline Performance (All Metrics)")
    ax.set_xlim([0.8, 1.0])
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val - 0.02, i, f"{val:.4f}", ha="right", va="center", 
                fontweight="bold", color="white")
    ax.grid(True, alpha=0.3, axis="x")
    
    plt.tight_layout()
    output_path = results_dir / "ablation_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Comparison chart saved: {output_path}")
    plt.close()


def plot_metric_heatmap(df: pd.DataFrame, results_dir: Path):
    """Create heatmaps showing relationships between components and metrics."""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Ablation Study: Metric Heatmaps", 
                 fontsize=14, fontweight="bold")
    
    # Prepare data for heatmaps
    # We'll create a matrix of F1 vs ROC-AUC for different components
    
    # 1. Bottleneck dimension heatmap
    ax = axes[0]
    bottleneck_df = df[df["category"] == "bottleneck_dimension"].copy()
    bottleneck_df = bottleneck_df.sort_values("bottleneck_dim")
    
    heatmap_data = bottleneck_df[["bottleneck_dim", "f1", "precision", "recall", "roc_auc"]].set_index("bottleneck_dim")
    sns.heatmap(heatmap_data.T, annot=True, fmt=".4f", cmap="RdYlGn", 
                vmin=0.8, vmax=1.0, ax=ax, cbar_kws={"label": "Score"})
    ax.set_title("Metrics by Bottleneck Dimension")
    ax.set_xlabel("Bottleneck Dimension")
    ax.set_ylabel("Metric")
    
    # 2. Depth heatmap
    ax = axes[1]
    depth_df = df[df["category"] == "network_depth"].copy()
    depth_df = depth_df.sort_values("depth")
    
    heatmap_data = depth_df[["depth", "f1", "precision", "recall", "roc_auc"]].set_index("depth")
    sns.heatmap(heatmap_data.T, annot=True, fmt=".4f", cmap="RdYlGn", 
                vmin=0.8, vmax=1.0, ax=ax, cbar_kws={"label": "Score"})
    ax.set_title("Metrics by Network Depth")
    ax.set_xlabel("Network Depth (# Hidden Layers)")
    ax.set_ylabel("Metric")
    
    plt.tight_layout()
    output_path = results_dir / "ablation_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Heatmap saved: {output_path}")
    plt.close()


def generate_analysis_report(results: Dict[str, Any], df: pd.DataFrame, 
                            results_dir: Path):
    """Generate a detailed text analysis report."""
    
    report = []
    report.append("=" * 80)
    report.append("ABLATION STUDY ANALYSIS REPORT")
    report.append("=" * 80)
    report.append(f"Timestamp: {results.get('timestamp', 'Unknown')}\n")
    
    # Data Summary
    report.append("DATA SUMMARY")
    report.append("-" * 80)
    data_summary = results.get("data_summary", {})
    report.append(f"  Total features: {data_summary.get('n_features', 'N/A')}")
    report.append(f"  Training samples: {data_summary.get('n_train', 'N/A')} (normal only)")
    report.append(f"  Validation samples: {data_summary.get('n_val', 'N/A')} (normal only)")
    report.append(f"  Test samples: {data_summary.get('n_test', 'N/A')}")
    report.append(f"  Abnormal test samples: {data_summary.get('n_abnormal', 'N/A')}\n")
    
    # Baseline Performance
    report.append("BASELINE PERFORMANCE")
    report.append("-" * 80)
    baseline = results["baseline"]
    report.append(f"  Configuration:")
    for key, val in baseline["config"].items():
        report.append(f"    {key}: {val}")
    report.append(f"\n  Metrics:")
    report.append(f"    F1 Score:  {baseline['f1']:.4f}")
    report.append(f"    ROC-AUC:   {baseline['roc_auc']:.4f}")
    report.append(f"    Accuracy:  {baseline['accuracy']:.4f}")
    report.append(f"    Precision: {baseline['precision']:.4f}")
    report.append(f"    Recall:    {baseline['recall']:.4f}")
    report.append(f"    Specificity: {baseline['specificity']:.4f}\n")
    
    # Component Analysis
    report.append("COMPONENT IMPACT ANALYSIS")
    report.append("-" * 80)
    
    for category, variants in results["variations"].items():
        report.append(f"\n{category.upper().replace('_', ' ')}")
        report.append("  " + "-" * 76)
        
        # Create comparison table
        variant_data = []
        for variant_name, variant_info in variants.items():
            variant_data.append({
                "name": variant_name,
                "f1": variant_info["f1"],
                "roc_auc": variant_info["roc_auc"],
                "accuracy": variant_info["accuracy"],
            })
        
        variant_df = pd.DataFrame(variant_data).sort_values("f1", ascending=False)
        
        for idx, row in variant_df.iterrows():
            f1_delta = (row["f1"] - baseline["f1"]) / baseline["f1"] * 100
            report.append(f"  {row['name']:30} F1={row['f1']:.4f} ({f1_delta:+.2f}%) " +
                        f"ROC-AUC={row['roc_auc']:.4f}")
        
        # Highlight best
        best = variant_df.iloc[0]
        best_f1_delta = (best["f1"] - baseline["f1"]) / baseline["f1"] * 100
        report.append(f"  ✓ Best: {best['name']:30} F1={best['f1']:.4f} ({best_f1_delta:+.2f}%)")
    
    # Key Findings
    report.append("\n\n" + "=" * 80)
    report.append("KEY FINDINGS & RECOMMENDATIONS")
    report.append("=" * 80)
    
    # Best overall variant
    non_baseline = df[df["variant"] != "baseline"].copy()
    best_variant_idx = non_baseline["f1"].idxmax()
    best_variant = non_baseline.loc[best_variant_idx]
    
    report.append(f"\nBest Overall Configuration:")
    report.append(f"  Variant: {best_variant['variant']}")
    report.append(f"  Category: {best_variant['category']}")
    for key in best_variant.index:
        if key not in ["variant", "category"] and not key.startswith(("f1", "roc_auc", "accuracy", "precision", "recall", "specificity", "tp", "fp", "fn", "tn", "epochs_run", "final_")):
            report.append(f"  {key}: {best_variant[key]}")
    report.append(f"  F1 Score: {best_variant['f1']:.4f}")
    report.append(f"  ROC-AUC: {best_variant['roc_auc']:.4f}")
    improvement = (best_variant['f1'] - baseline['f1']) / baseline['f1'] * 100
    report.append(f"  Improvement over baseline: {improvement:+.2f}%\n")
    
    # Component rankings
    report.append("Component Sensitivity Ranking (by max F1 improvement):")
    for category in results["variations"].keys():
        cat_df = df[df["category"] == category]
        max_f1 = cat_df["f1"].max()
        improvement = (max_f1 - baseline["f1"]) / baseline["f1"] * 100
        report.append(f"  {category:25} {improvement:+.2f}%")
    
    # Recommendations
    report.append("\n\nRecommendations:")
    report.append("  1. Consider configurations that significantly improve F1 (>2% improvement)")
    report.append("  2. Balance F1, Precision, and Recall based on use case requirements")
    report.append("  3. Monitor for overfitting when using reduced feature sets")
    report.append("  4. Test recommended configurations on hold-out test set before deployment")
    
    # Write report with UTF-8 encoding to handle special characters (✓, etc.)
    report_text = "\n".join(report)
    output_path = results_dir / "ablation_analysis_report.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    print(f"✓ Analysis report saved: {output_path}")
    
    # Also print to console
    print("\n" + report_text)


def analyze_ablation_study(
    results_path: Path = None,
    results_dir: Path = None
):
    """Main analysis function."""
    
    if results_dir is None:
        results_dir = Path("results")
    
    if results_path is None:
        results_path = results_dir / "ablation_study.json"
    
    print("\n" + "█" * 80)
    print("█ ABLATION STUDY ANALYSIS")
    print("█" * 80)
    
    if not results_path.exists():
        print(f"\n✗ Results file not found: {results_path}")
        print("  Run ablation_study.py first to generate results.")
        return
    
    print(f"\n[1/4] Loading results from {results_path}...")
    results = load_results(results_path)
    
    print("[2/4] Extracting data...")
    df = extract_comparison_data(results)
    print(f"  ✓ Loaded {len(df)} variants (including baseline)")
    
    print("[3/4] Generating visualizations...")
    plot_component_comparison(df, results_dir)
    plot_metric_heatmap(df, results_dir)
    
    print("[4/4] Generating analysis report...")
    generate_analysis_report(results, df, results_dir)
    
    print("\n" + "█" * 80)
    print("█ Analysis complete!")
    print("█" * 80)
    
    return df, results


if __name__ == "__main__":
    analyze_ablation_study()
