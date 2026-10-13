"""Analyze and summarize OULAD ablation study results.

This script loads the ablation study CSV results and generates:
1. Comparison charts for each component
2. Detailed analysis report
3. Summary of the main findings
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


# Random seeds used in the ablation study
SEEDS = [41, 42, 43, 44, 45]
SEEDS_STR = ", ".join(map(str, SEEDS))


# Results directories
RESULTS_DIR = Path("results")
ABLATION_DIR = RESULTS_DIR / "ablation"


def load_results():
    """Load the ablation study results."""

    bottleneck = pd.read_csv(
        ABLATION_DIR / "bottleneck_ablation.csv"
    )

    depth = pd.read_csv(
        ABLATION_DIR / "depth_ablation.csv"
    )

    noise = pd.read_csv(
        ABLATION_DIR / "noise_scale_ablation.csv"
    )

    bottleneck_individual = pd.read_csv(
        ABLATION_DIR / "bottleneck_ablation_individual.csv"
    )

    depth_individual = pd.read_csv(
        ABLATION_DIR / "depth_ablation_individual.csv"
    )

    noise_individual = pd.read_csv(
        ABLATION_DIR / "noise_scale_ablation_individual.csv"
    )

    return (
        bottleneck,
        depth,
        noise,
        bottleneck_individual,
        depth_individual,
        noise_individual,
    )


def plot_component_comparison(
    bottleneck,
    depth,
    noise
):
    """Create comparison charts for the ablation results."""

    # Bottleneck dimension
    plt.figure(figsize=(8, 5))

    plt.plot(
        bottleneck["bottleneck_dim"],
        bottleneck["f1_mean"],
        marker="o",
        label="F1 Score"
    )

    plt.plot(
        bottleneck["bottleneck_dim"],
        bottleneck["roc_auc_mean"],
        marker="s",
        label="ROC-AUC"
    )

    plt.xlabel("Bottleneck Dimension")
    plt.ylabel("Score")
    plt.title(
        "OULAD Ablation: Bottleneck Dimension "
        "(Mean Across 5 Seeds)"
    )
    plt.xticks(bottleneck["bottleneck_dim"])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = ABLATION_DIR / "bottleneck_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Bottleneck chart saved: {output_path}")

    # Network depth
    plt.figure(figsize=(8, 5))

    plt.plot(
        depth["n_hidden_layers"],
        depth["f1_mean"],
        marker="o",
        label="F1 Score"
    )

    plt.plot(
        depth["n_hidden_layers"],
        depth["roc_auc_mean"],
        marker="s",
        label="ROC-AUC"
    )

    plt.xlabel("Network Depth (# Hidden Layers)")
    plt.ylabel("Score")
    plt.title(
        "OULAD Ablation: Network Depth "
        "(Mean Across 5 Seeds)"
    )
    plt.xticks(depth["n_hidden_layers"])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = ABLATION_DIR / "depth_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Depth chart saved: {output_path}")

    # Noise scale sensitivity
    plt.figure(figsize=(8, 5))

    plt.plot(
        noise["noise_scale"],
        noise["f1_mean"],
        marker="o",
        label="F1 Score"
    )

    plt.plot(
        noise["noise_scale"],
        noise["roc_auc_mean"],
        marker="s",
        label="ROC-AUC"
    )

    plt.xlabel("Noise Scale")
    plt.ylabel("Score")
    plt.title(
        "OULAD Ablation: Noise Scale Sensitivity "
        "(Mean Across 5 Seeds)"
    )
    plt.xticks(noise["noise_scale"])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = ABLATION_DIR / "noise_scale_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✓ Noise scale chart saved: {output_path}")


def generate_analysis_report(
    bottleneck,
    depth,
    noise,
    bottleneck_individual,
    depth_individual,
    noise_individual,
):
    """Generate a detailed text analysis report."""

    report = []

    report.append("=" * 80)
    report.append("OULAD ABLATION STUDY ANALYSIS REPORT")
    report.append("=" * 80)
    report.append(f"Random Seeds: {SEEDS_STR}\n")

    # Experimental setup
    report.append("EXPERIMENTAL SETUP")
    report.append("-" * 80)
    report.append(f"  Random Seeds: {SEEDS_STR}")
    report.append("  Split: 60/15/25")
    report.append("  Features: 31")
    report.append("  Injection fraction: 0.12")
    report.append("  Baseline bottleneck dimension: 4")
    report.append("  Baseline network depth: 3")
    report.append("  Baseline noise scale: 1.8")
    report.append("")

    # Baseline performance
    baseline = bottleneck[
        bottleneck["bottleneck_dim"] == 4
    ].iloc[0]

    report.append("BASELINE PERFORMANCE")
    report.append("-" * 80)

    report.append(
        f"  F1 Score: "
        f"{baseline['f1_mean']:.4f} ± "
        f"{baseline['f1_std']:.4f}"
    )

    report.append(
        f"  ROC-AUC:  "
        f"{baseline['roc_auc_mean']:.4f} ± "
        f"{baseline['roc_auc_std']:.4f}"
    )

    report.append(
        f"  Precision: "
        f"{baseline['precision_mean']:.4f} ± "
        f"{baseline['precision_std']:.4f}"
    )

    report.append(
        f"  Recall:    "
        f"{baseline['recall_mean']:.4f} ± "
        f"{baseline['recall_std']:.4f}"
    )

    report.append("")

    # Bottleneck analysis
    report.append("COMPONENT IMPACT ANALYSIS")
    report.append("-" * 80)

    report.append("\nBOTTLENECK DIMENSION")
    report.append("  " + "-" * 76)

    for dim in bottleneck["bottleneck_dim"]:

        runs = bottleneck_individual[
            bottleneck_individual["bottleneck_dim"] == dim
        ]

        report.append(f"  bottleneck_{int(dim)}")

        for _, run in runs.iterrows():
            report.append(
                f"    Seed {int(run['seed'])}: "
                f"F1={run['f1']:.4f} "
                f"ROC-AUC={run['roc_auc']:.4f}"
            )

        summary = bottleneck[
            bottleneck["bottleneck_dim"] == dim
        ].iloc[0]

        report.append(
            f"    Mean: "
            f"F1={summary['f1_mean']:.4f} ± "
            f"{summary['f1_std']:.4f} "
            f"ROC-AUC={summary['roc_auc_mean']:.4f} ± "
            f"{summary['roc_auc_std']:.4f}"
        )

    best_bottleneck = bottleneck.loc[
        bottleneck["f1_mean"].idxmax()
    ]

    baseline_f1 = baseline["f1_mean"]

    improvement = (
        (best_bottleneck["f1_mean"] - baseline_f1)
        / baseline_f1
        * 100
    )

    report.append(
        f"  ✓ Best: bottleneck_"
        f"{int(best_bottleneck['bottleneck_dim'])} "
        f"Mean F1={best_bottleneck['f1_mean']:.4f} "
        f"({improvement:+.2f}%)"
    )

    # Depth analysis
    report.append("\nNETWORK DEPTH")
    report.append("  " + "-" * 76)

    for depth_value in depth["n_hidden_layers"]:

        runs = depth_individual[
            depth_individual["n_hidden_layers"] == depth_value
        ]

        report.append(f"  depth_{int(depth_value)}")

        for _, run in runs.iterrows():
            report.append(
                f"    Seed {int(run['seed'])}: "
                f"F1={run['f1']:.4f} "
                f"ROC-AUC={run['roc_auc']:.4f}"
            )

        summary = depth[
            depth["n_hidden_layers"] == depth_value
        ].iloc[0]

        report.append(
            f"    Mean: "
            f"F1={summary['f1_mean']:.4f} ± "
            f"{summary['f1_std']:.4f} "
            f"ROC-AUC={summary['roc_auc_mean']:.4f} ± "
            f"{summary['roc_auc_std']:.4f}"
        )

    best_depth = depth.loc[
        depth["f1_mean"].idxmax()
    ]

    improvement = (
        (best_depth["f1_mean"] - baseline_f1)
        / baseline_f1
        * 100
    )

    report.append(
        f"  ✓ Best: depth_"
        f"{int(best_depth['n_hidden_layers'])} "
        f"Mean F1={best_depth['f1_mean']:.4f} "
        f"({improvement:+.2f}%)"
    )

    # Noise scale sensitivity
    report.append("\nNOISE SCALE SENSITIVITY")
    report.append("  " + "-" * 76)

    for noise_value in noise["noise_scale"]:

        runs = noise_individual[
            noise_individual["noise_scale"] == noise_value
        ]

        report.append(f"  noise_scale_{noise_value}")

        for _, run in runs.iterrows():
            report.append(
                f"    Seed {int(run['seed'])}: "
                f"F1={run['f1']:.4f} "
                f"ROC-AUC={run['roc_auc']:.4f}"
            )

        summary = noise[
            noise["noise_scale"] == noise_value
        ].iloc[0]

        report.append(
            f"    Mean: "
            f"F1={summary['f1_mean']:.4f} ± "
            f"{summary['f1_std']:.4f} "
            f"ROC-AUC={summary['roc_auc_mean']:.4f} ± "
            f"{summary['roc_auc_std']:.4f}"
        )

    highest_noise = noise.loc[
        noise["f1_mean"].idxmax()
    ]

    report.append(
        f"  ✓ Highest tested severity: "
        f"noise_scale_{highest_noise['noise_scale']} "
        f"Mean F1={highest_noise['f1_mean']:.4f}"
    )

    # Key findings
    report.append("\n\n" + "=" * 80)
    report.append("KEY FINDINGS")
    report.append("=" * 80)

    report.append("\nBest Architectural Variant:")

    report.append(
        f"  Bottleneck dimension: "
        f"{int(best_bottleneck['bottleneck_dim'])}"
    )

    report.append(
        f"  Mean F1 Score: "
        f"{best_bottleneck['f1_mean']:.4f} ± "
        f"{best_bottleneck['f1_std']:.4f}"
    )

    report.append(
        f"  Mean ROC-AUC: "
        f"{best_bottleneck['roc_auc_mean']:.4f} ± "
        f"{best_bottleneck['roc_auc_std']:.4f}"
    )

    report.append("\nBest Network Depth:")

    report.append(
        f"  Depth: "
        f"{int(best_depth['n_hidden_layers'])}"
    )

    report.append(
        f"  Mean F1 Score: "
        f"{best_depth['f1_mean']:.4f} ± "
        f"{best_depth['f1_std']:.4f}"
    )

    report.append("\nNoise Scale Sensitivity:")

    report.append(
        f"  Highest tested severity: "
        f"{highest_noise['noise_scale']}"
    )

    report.append(
        f"  Mean F1 Score: "
        f"{highest_noise['f1_mean']:.4f} ± "
        f"{highest_noise['f1_std']:.4f}"
    )

    report.append(
        f"  Mean ROC-AUC: "
        f"{highest_noise['roc_auc_mean']:.4f} ± "
        f"{highest_noise['roc_auc_std']:.4f}"
    )

    report.append(
        "\n  Higher noise scales represent stronger synthetic "
        "anomalies and are"
    )

    report.append(
        "  treated as a sensitivity analysis rather than "
        "model selection."
    )

    # Recommendations
    report.append("\n\nRecommendations:")

    report.append(
        "  1. Compare configurations using mean performance "
        "across all seeds"
    )

    report.append(
        "  2. Keep noise scale separate from model "
        "hyperparameter selection"
    )

    report.append(
        "  3. Report the exact baseline configuration used "
        "for the main experiments"
    )

    report.append(
        "  4. Avoid selecting configurations based on a "
        "single random seed"
    )

    # Save report
    report_text = "\n".join(report)

    output_path = ABLATION_DIR / "ablation_analysis_report.txt"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"✓ Analysis report saved: {output_path}")

    print("\n" + report_text)


def analyze_ablation_study():
    """Main analysis function."""

    print("\n" + "█" * 80)
    print("█ OULAD ABLATION STUDY ANALYSIS")
    print("█" * 80)

    print("\n[1/3] Loading ablation results...")

    try:
        (
            bottleneck,
            depth,
            noise,
            bottleneck_individual,
            depth_individual,
            noise_individual,
        ) = load_results()

    except FileNotFoundError as exc:
        print(f"\n✗ Results file not found: {exc}")
        print(
            "  Run run_ablation_study.py first "
            "to generate the results."
        )
        return

    print("  ✓ Ablation results loaded")

    print("\n[2/3] Generating comparison charts...")

    plot_component_comparison(
        bottleneck,
        depth,
        noise
    )

    print("\n[3/3] Generating analysis report...")

    generate_analysis_report(
        bottleneck,
        depth,
        noise,
        bottleneck_individual,
        depth_individual,
        noise_individual,
    )

    print("\n" + "█" * 80)
    print("█ Analysis complete!")
    print("█" * 80)


if __name__ == "__main__":
    analyze_ablation_study()