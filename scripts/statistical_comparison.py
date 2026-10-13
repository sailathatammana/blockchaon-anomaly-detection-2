"""Compare the models statistically using the results from the 5 seeds."""

import pandas as pd
from scipy.stats import wilcoxon


# Load results from the multi-seed experiment
results = pd.read_csv("results/multiseed_individual_results.csv")


# Models that are being compared
comparisons = [
    ("Autoencoder", "VAE"),
    ("Autoencoder", "Isolation Forest"),
    ("Autoencoder", "PCA"),
    ("VAE", "Isolation Forest")
]


# Metrics to compare
metrics = ["f1", "roc_auc"]


print("=" * 60)
print("STATISTICAL COMPARISON")
print("=" * 60)


for metric in metrics:

    print(f"\n{metric.upper()}")
    print("-" * 60)

    for model1, model2 in comparisons:

        # Get results for each model and keep seeds in the same order
        values1 = results[
            results["model"] == model1
        ].sort_values("seed")[metric].values

        values2 = results[
            results["model"] == model2
        ].sort_values("seed")[metric].values

        # Wilcoxon test because the results are paired by seed
        statistic, p_value = wilcoxon(
            values1,
            values2
        )

        print(f"\n{model1} vs {model2}")
        print(f"{model1} mean: {values1.mean():.4f}")
        print(f"{model2} mean: {values2.mean():.4f}")
        print(f"p-value: {p_value:.4f}")

        if p_value < 0.05:
            print("Significant difference")
        else:
            print("No significant difference")


print("\nDone.")