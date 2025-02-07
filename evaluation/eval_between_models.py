import os
import json
import numpy as np
import matplotlib.pyplot as plt


def analyze_model_performance(evaluation_charts_folder, output_folder):
    """
    Analyzes and compares model performance across domains, creates statistics for each domain,
    and generates charts for comparisons and deviations.

    Parameters:
        evaluation_charts_folder (str): The folder containing evaluation results for different models.
        output_folder (str): The folder where charts and JSON files will be saved.

    Returns:
        dict: A dictionary containing statistics and deviations for each domain.
    """
    domain_stats = {}

    # Iterate through each model folder (e.g., wordninja, bert, etc.)
    for model_name in os.listdir(evaluation_charts_folder):
        model_path = os.path.join(evaluation_charts_folder, model_name)
        if not os.path.isdir(model_path):
            continue

        # Iterate through domain folders within each model's folder
        for domain_folder in os.listdir(model_path):
            domain_path = os.path.join(model_path, domain_folder)
            if not os.path.isdir(domain_path):
                continue

            metrics_file = os.path.join(domain_path, f"{domain_folder}_metrics.json")
            if not os.path.exists(metrics_file):
                print(f"Metrics file missing for domain {domain_folder} in model {model_name}, skipping.")
                continue

            # Load metrics for the domain
            with open(metrics_file, 'r', encoding='utf-8') as mf:
                metrics = json.load(mf)

            # Initialize domain stats if not already created
            if domain_folder not in domain_stats:
                domain_stats[domain_folder] = {
                    "models": {},
                    "comparisons": {}
                }

            # Store model metrics for the domain
            domain_stats[domain_folder]["models"][model_name] = metrics

    # Compute statistics and deviations
    for domain, stats in domain_stats.items():
        model_metrics = stats["models"]

        # Extract all precision, recall, and F1-scores
        precisions = [metrics["precision"] for metrics in model_metrics.values()]
        recalls = [metrics["recall"] for metrics in model_metrics.values()]
        f1_scores = [metrics["f1_score"] for metrics in model_metrics.values()]

        # Compute statistics
        stats["comparisons"]["precision"] = {
            "mean": np.mean(precisions),
            "std_dev": np.std(precisions),
            "min": np.min(precisions),
            "max": np.max(precisions),
            "range": np.max(precisions) - np.min(precisions)
        }
        stats["comparisons"]["recall"] = {
            "mean": np.mean(recalls),
            "std_dev": np.std(recalls),
            "min": np.min(recalls),
            "max": np.max(recalls),
            "range": np.max(recalls) - np.min(recalls)
        }
        stats["comparisons"]["f1_score"] = {
            "mean": np.mean(f1_scores),
            "std_dev": np.std(f1_scores),
            "min": np.min(f1_scores),
            "max": np.max(f1_scores),
            "range": np.max(f1_scores) - np.min(f1_scores)
        }

    # Save results to JSON files in the output folder
    os.makedirs(output_folder, exist_ok=True)
    stats_file = os.path.join(output_folder, "domain_model_comparisons.json")
    with open(stats_file, 'w', encoding='utf-8') as sf:
        json.dump(domain_stats, sf, indent=4)

    # Save deviations in a separate JSON file
    deviations = {}
    for domain, stats in domain_stats.items():
        model_metrics = stats["models"]

        # Calculate deviations for each model compared to the mean
        deviations[domain] = {
            model: {
                "precision_deviation": metrics["precision"] - stats["comparisons"]["precision"]["mean"],
                "recall_deviation": metrics["recall"] - stats["comparisons"]["recall"]["mean"],
                "f1_score_deviation": metrics["f1_score"] - stats["comparisons"]["f1_score"]["mean"]
            }
            for model, metrics in model_metrics.items()
        }

        # Create charts for the domain
        create_domain_charts(domain, stats, deviations[domain], output_folder)

    deviations_file = os.path.join(output_folder, "domain_model_deviations.json")
    with open(deviations_file, 'w', encoding='utf-8') as df:
        json.dump(deviations, df, indent=4)

    return domain_stats, deviations


def create_domain_charts(domain, stats, deviations, base_folder):
    """
    Create bar charts for model comparisons and deviations for a specific domain.

    Parameters:
        domain (str): The domain name.
        stats (dict): Statistics for the domain.
        deviations (dict): Deviations for each model in the domain.
        base_folder (str): Base folder for evaluation charts.
    """
    domain_folder = os.path.join(base_folder, f"{domain}_charts")
    os.makedirs(domain_folder, exist_ok=True)

    models = list(stats["models"].keys())
    precisions = [stats["models"][model]["precision"] for model in models]
    recalls = [stats["models"][model]["recall"] for model in models]
    f1_scores = [stats["models"][model]["f1_score"] for model in models]

    # Plot comparisons
    plt.figure()
    x = np.arange(len(models))
    width = 0.25
    plt.bar(x - width, precisions, width, label="Precision")
    plt.bar(x, recalls, width, label="Recall")
    plt.bar(x + width, f1_scores, width, label="F1-Score")
    plt.xticks(x, models, rotation=45)
    plt.title(f"Model Comparisons for {domain}")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    comparison_chart = os.path.join(domain_folder, f"{domain}_comparison_chart.png")
    plt.savefig(comparison_chart)
    plt.close()

    # Plot deviations
    precision_devs = [deviations[model]["precision_deviation"] for model in models]
    recall_devs = [deviations[model]["recall_deviation"] for model in models]
    f1_devs = [deviations[model]["f1_score_deviation"] for model in models]

    plt.figure()
    plt.bar(x - width, precision_devs, width, label="Precision Deviation")
    plt.bar(x, recall_devs, width, label="Recall Deviation")
    plt.bar(x + width, f1_devs, width, label="F1-Score Deviation")
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.7)
    plt.xticks(x, models, rotation=45)
    plt.title(f"Model Deviations for {domain}")
    plt.ylabel("Deviation")
    plt.tight_layout()
    deviation_chart = os.path.join(domain_folder, f"{domain}_deviation_chart.png")
    plt.savefig(deviation_chart)
    plt.close()


if __name__ == '__main__':
    evaluation_charts_folder = './evaluation_charts'
    output_folder = './evaluation_results'  # New folder for saving results
    domain_stats, deviations = analyze_model_performance(evaluation_charts_folder, output_folder)
    print("Domain Statistics and Deviation Analysis Complete.")
