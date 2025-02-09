import json
import wordninja
import enchant
from collections import Counter
import os
import matplotlib.pyplot as plt
from rapidfuzz.distance import Levenshtein
import math

# Initialize PyEnchant English dictionary
d = enchant.Dict("en_US")

def is_subword(subword, word, error_margin=0.3):

    # Compute Levenshtein distance
    distance = Levenshtein.distance(subword, word)
    threshold = math.ceil(error_margin * max(len(subword), len(word)))

    return distance <= threshold

def evaluate_segmentation(sample_file, segmented_file, output_folder, error_margin=0.3):
    """
    Evaluates the segmentation predictions in two stages:
      1) Exact Match (Counters)
      2) Subword or Fuzzy Match on leftover tokens (using percentage-based margin of error).

    Saves debugging info for each item in the console and
    writes it to a text file in the output folder.
    Returns (precision, recall, f1_score).
    """
    with open(sample_file, 'r', encoding='utf-8') as f:
        sample_data = json.load(f)
    with open(segmented_file, 'r', encoding='utf-8') as f:
        segmented_data = json.load(f)

    domain = os.path.basename(sample_file)[:3]
    output_text_file = os.path.join(output_folder, f"{domain}_evaluation_output.txt")

    with open(output_text_file, 'w', encoding='utf-8') as out_file:
        out_file.write(f"\n=== Evaluating Domain File ===\n")
        out_file.write(f"Sample file:    {sample_file}\n")
        out_file.write(f"Segmented file: {segmented_file}\n\n")

        total_tp = 0
        total_fp = 0
        total_fn = 0

        def get_predicted_tokens(text):
            """
            Remove underscores, then apply wordninja.
            Keep only valid English words (PyEnchant).
            """
            joined = text.replace('_', '')
            tokens = wordninja.split(joined)
            return [word.lower() for word in tokens if d.check(word) and not word.isdigit() and len(word) > 1]

        for category in ['predicates', 'constants', 'variables']:
            out_file.write(f"--- Category: {category} ---\n")
            for i, original_string in enumerate(sample_data[category]):
                if i >= len(segmented_data[category]):
                    # No ground truth available for this index => ground truth = []
                    gt_tokens = []
                    # The WordNinja segmentation is now our prediction
                    predicted_tokens = get_predicted_tokens(original_string)

                    out_file.write(f"  Original:        {original_string}\n")
                    out_file.write(f"  Ground Truth:    {gt_tokens}\n")
                    out_file.write(f"  Prediction:      {predicted_tokens}\n")
                    total_fn += len(gt_tokens)
                    continue

                # Otherwise, ground truth comes from segmented_data
                gt_tokens = segmented_data[category][i]
                # WordNinja output is the "prediction"
                predicted_tokens = get_predicted_tokens(original_string)

                out_file.write(f"  Original:     {original_string}\n")
                out_file.write(f"  Ground Truth: {gt_tokens}\n")
                out_file.write(f"  Prediction:   {predicted_tokens}\n")

                # 1) Exact match counting
                gt_counter = Counter(gt_tokens)
                pred_counter = Counter(predicted_tokens)

                # Count true positives for exact matches
                exact_tp_per_token = {}
                for token in set(gt_counter.keys()).union(pred_counter.keys()):
                    matched = min(gt_counter[token], pred_counter[token])
                    exact_tp_per_token[token] = matched

                exact_tp = sum(exact_tp_per_token.values())

                # Figure out leftover tokens after exact matches
                leftover_gt_counter = Counter({
                    token: gt_counter[token] - exact_tp_per_token[token]
                    for token in gt_counter if gt_counter[token] > exact_tp_per_token[token]
                })
                leftover_pred_counter = Counter({
                    token: pred_counter[token] - exact_tp_per_token[token]
                    for token in pred_counter if pred_counter[token] > exact_tp_per_token[token]
                })

                leftover_gt_list = list(leftover_gt_counter.elements())
                leftover_pred_list = list(leftover_pred_counter.elements())

                # 2) Subword/fuzzy matching with a percentage-based Levenshtein distance
                subword_tp = 0
                for pred_token in leftover_pred_list:
                    best_match = None
                    best_idx = None

                    for idx, gt_token in enumerate(leftover_gt_list):
                        if is_subword(pred_token, gt_token, error_margin=0.3):
                            best_match = gt_token
                            best_idx = idx
                            break

                    if best_match is not None:
                        # Award partial match (example: 0.5 so it doesn't dominate exact matches)
                        subword_tp += 1
                        leftover_gt_list.pop(best_idx)

                item_tp = exact_tp + (subword_tp/2)
                item_fp = len(leftover_pred_list) - subword_tp
                item_fn = len(leftover_gt_list)

                total_tp += item_tp
                total_fp += item_fp
                total_fn += item_fn

                out_file.write(f"    Exact TP: {exact_tp}, Subword TP: {subword_tp}, "
                               f"FP: {item_fp}, FN: {item_fn}\n")

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        out_file.write("\n=== Final Evaluation Results for This Domain ===\n")
        out_file.write(f"  Precision: {precision:.4f}\n")
        out_file.write(f"  Recall:    {recall:.4f}\n")
        out_file.write(f"  F1-Score:  {f1_score:.4f}\n")
        out_file.write("===============================================\n\n")

    return precision, recall, f1_score

def process_folders(sample_folder, segmented_folder, output_folder, error_margin=0.3):
    """
    Processes all sample and segmented files in their respective folders.
    Creates evaluation metrics and saves charts for each domain.
    Also writes a JSON file containing the precision, recall, and F1 for each domain.
    """
    for sample_file in os.listdir(sample_folder):
        if not sample_file.endswith('.json'):
            continue

        domain = sample_file[:3]  # Get domain name from first 3 characters
        segmented_file = os.path.join(segmented_folder, sample_file)

        if not os.path.exists(segmented_file):
            print(f"Missing segmented file for {sample_file}, skipping.")
            continue

        sample_file_path = os.path.join(sample_folder, sample_file)

        # Create output folder for the domain
        domain_output_folder = os.path.join(output_folder, domain)
        os.makedirs(domain_output_folder, exist_ok=True)

        # Evaluate segmentation with a percentage-based margin of error
        precision, recall, f1 = evaluate_segmentation(sample_file_path,
                                                      segmented_file,
                                                      domain_output_folder,
                                                      error_margin=error_margin)

        # Save chart
        save_chart(domain_output_folder, precision, recall, f1, domain)

        # Save JSON metrics
        save_metrics_json(domain_output_folder, precision, recall, f1, domain)

def save_chart(output_folder, precision, recall, f1, domain):
    metrics = ['Precision', 'Recall', 'F1-Score']
    values = [precision, recall, f1]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Assign specified colors

    plt.figure(figsize=(6, 4))
    plt.bar(metrics, values, color=colors)
    plt.ylim(0, 1)
    plt.title(f"Evaluation Metrics for {domain}")
    plt.ylabel("Score")

    # Save the chart
    output_path = os.path.join(output_folder, f"{domain}_metrics.png")
    plt.savefig(output_path)

    plt.close()

def save_metrics_json(output_folder, precision, recall, f1, domain):
    """
    Saves a JSON file with Precision, Recall, and F1 for the given domain.
    """
    results_dict = {
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    output_json_path = os.path.join(output_folder, f"{domain}_metrics.json")
    with open(output_json_path, 'w', encoding='utf-8') as jf:
        json.dump(results_dict, jf, indent=4)

def calculate_global_metrics(output_folder):
    """
    Calculates the global average metrics (Precision, Recall, F1-Score) 
    across all processed domains.

    Parameters:
        output_folder (str): The folder where individual domain metrics JSON files are stored.

    Returns:
        dict: Dictionary containing global average Precision, Recall, and F1-score.
    """
    global_precision = 0.0
    global_recall = 0.0
    global_f1_score = 0.0
    domain_count = 0

    # Iterate through domain subfolders
    for domain_folder in os.listdir(output_folder):
        domain_path = os.path.join(output_folder, domain_folder)
        if not os.path.isdir(domain_path):
            continue

        metrics_file = os.path.join(domain_path, f"{domain_folder}_metrics.json")
        if not os.path.exists(metrics_file):
            print(f"Metrics file missing for domain {domain_folder}, skipping.")
            continue

        # Load metrics for the domain
        with open(metrics_file, 'r', encoding='utf-8') as mf:
            metrics = json.load(mf)

        global_precision += metrics.get("precision", 0.0)
        global_recall += metrics.get("recall", 0.0)
        global_f1_score += metrics.get("f1_score", 0.0)
        domain_count += 1

    # Calculate averages
    if domain_count > 0:
        global_precision /= domain_count
        global_recall /= domain_count
        global_f1_score /= domain_count

    global_metrics = {
        "average_precision": global_precision,
        "average_recall": global_recall,
        "average_f1_score": global_f1_score
    }

    # Save the global metrics as a JSON file
    global_metrics_file = os.path.join(output_folder, "global_metrics.json")
    with open(global_metrics_file, 'w', encoding='utf-8') as gm_file:
        json.dump(global_metrics, gm_file, indent=4)

    return global_metrics

def plot_global_metrics(output_folder):
    """
    Reads the global metrics JSON file and plots a bar chart.
    """
    global_metrics_file = os.path.join(output_folder, "global_metrics.json")
    
    if not os.path.exists(global_metrics_file):
        print("Global metrics file not found.")
        return
    
    # Load global metrics
    with open(global_metrics_file, 'r', encoding='utf-8') as gm_file:
        global_metrics = json.load(gm_file)
    
    # Extract metric values
    metrics = ["Precision", "Recall", "F1-Score"]
    values = [global_metrics["average_precision"], global_metrics["average_recall"], global_metrics["average_f1_score"]]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Fixed colors for consistency
    
    # Plot the metrics
    plt.figure(figsize=(8, 5))
    plt.bar(metrics, values, color=colors)
    plt.ylim(0, 1)
    plt.title("Global Evaluation Metrics")
    plt.ylabel("Score")
    
    # Save the chart
    output_path = os.path.join(output_folder, "global_metrics.png")
    plt.savefig(output_path)

def save_all_domain_metrics(output_folder):
    """
    Aggregates all domain metrics and saves them into a single JSON file.
    """
    all_metrics = {}

    # Iterate through domain subfolders
    for domain_folder in os.listdir(output_folder):
        domain_path = os.path.join(output_folder, domain_folder)
        if not os.path.isdir(domain_path):
            continue

        metrics_file = os.path.join(domain_path, f"{domain_folder}_metrics.json")
        if not os.path.exists(metrics_file):
            print(f"Metrics file missing for domain {domain_folder}, skipping.")
            continue

        # Load domain metrics
        with open(metrics_file, 'r', encoding='utf-8') as mf:
            metrics = json.load(mf)

        all_metrics[domain_folder] = metrics

    # Save the aggregated metrics as a JSON file
    aggregated_metrics_file = os.path.join(output_folder, "all_domain_metrics.json")
    with open(aggregated_metrics_file, 'w', encoding='utf-8') as all_mf:
        json.dump(all_metrics, all_mf, indent=4)

    print(f"All domain metrics saved to {aggregated_metrics_file}")

if __name__ == '__main__':
    sample_folder = './samples_of_each_domain'
    segmented_folder = './segmented_samples'
    output_folder = './evaluation_charts/wordninja'
    
    # Set error_margin to 30% (default). Adjust if you want more or less tolerance.
    process_folders(sample_folder, segmented_folder, output_folder, error_margin=0.3)

    global_metrics = calculate_global_metrics(output_folder)
    print("Global Metrics:", global_metrics)

    # Generate and display the global metrics graph
    plot_global_metrics(output_folder)

    # Save all domain metrics into a single JSON file
    save_all_domain_metrics(output_folder)