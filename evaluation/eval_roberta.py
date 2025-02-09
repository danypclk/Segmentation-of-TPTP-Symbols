from rapidfuzz.distance import Levenshtein
import json
import enchant
import math
import re
from collections import Counter
import os
import matplotlib.pyplot as plt
import torch
from transformers import RobertaForTokenClassification, RobertaTokenizer
from safetensors.torch import load_file
from transformers import RobertaConfig

MODEL_DIR = "./roberta-segmentation"
HUGGINGFACE_MODEL_NAME = "danypereira264/roberta-segmentation"

def load_roberta_model_and_tokenizer(local_model_dir=MODEL_DIR, remote_model_name=HUGGINGFACE_MODEL_NAME):
    """
    Load a fine-tuned RoBERTa segmentation model.
    - If a local model exists in `local_model_dir`, it loads from there.
    - If not, it loads directly from Hugging Face cache (without copying to `local_model_dir`).

    Returns:
        model (RobertaForTokenClassification): The loaded model in evaluation mode.
        tokenizer (RobertaTokenizer): The corresponding tokenizer.
    """
    # Check if local model exists
    if os.path.isdir(local_model_dir):
        print(f" Loading model from local directory: {local_model_dir}")
        model_path = local_model_dir
    else:
        print(f" Local model not found. Using Hugging Face cache: {remote_model_name}")
        model_path = remote_model_name  # Use Hugging Face directly without saving

    # Load model and tokenizer
    model = RobertaForTokenClassification.from_pretrained(model_path)
    tokenizer = RobertaTokenizer.from_pretrained(model_path)

    model.eval()  # Set model to evaluation mode
    return model, tokenizer

# 🔹 Load the RoBERTa model and tokenizer globally
roberta_model, roberta_tokenizer = load_roberta_model_and_tokenizer()

# Initialize PyEnchant English dictionary
d = enchant.Dict("en_US")

def predict_labels(phrase, model, tokenizer):
    """
    Predict segmentation labels for the input phrase and return segmented words.
    """
    encoding = tokenizer(
        list(phrase),
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1).squeeze(0).tolist()
    
    sequence_length = attention_mask.sum().item()
    preds = preds[:sequence_length]
    if len(preds) > 2:
        preds = preds[:-2]
    else:
        preds = []
    
    def segment_text(text, labels):
        words, current_word = [], []
        for char, label in zip(text, labels):
            char = char.lower()  # Convert character to lowercase
            if label == 1:
                if current_word:
                    words.append("".join(current_word))
                current_word = [char]
            else:
                current_word.append(char)
        if current_word:
            words.append("".join(current_word))
        # Remove special characters including underscores explicitly
        words = [re.sub(r'[^a-zA-Z]', '', word) for word in words if len(re.sub(r'[^a-zA-Z]', '', word)) > 1 and d.check(word)]
        return [word for word in words if word]  # Remove empty strings
    
    return segment_text(phrase, preds)

def is_subword(subword, word, error_margin=0.3):
    return Levenshtein.distance(subword, word) <= math.ceil(error_margin * max(len(subword), len(word)))

# Evaluate segmentation
def evaluate_segmentation(sample_file, segmented_file, output_folder, max_subword_distance=1):
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

        for category in ['predicates', 'constants', 'variables']:
            out_file.write(f"--- Category: {category} ---\n")
            for i, original_string in enumerate(sample_data[category]):
                if i >= len(segmented_data[category]):
                    # No ground truth available, so ground truth = []
                    gt_tokens = []
                    predicted_tokens = predict_labels(original_string, roberta_model, roberta_tokenizer)
                    out_file.write(f"  Original:        {original_string}\n")
                    out_file.write(f"  Ground Truth:    {gt_tokens}\n")
                    out_file.write(f"  Prediction:      {predicted_tokens}\n")
                    total_fn += len(gt_tokens)
                    continue

                # Now ground truth comes from segmented_data,
                # and BERT inference is our prediction:
                gt_tokens = segmented_data[category][i]
                predicted_tokens = predict_labels(original_string, roberta_model, roberta_tokenizer)

                out_file.write(f"  Original:     {original_string}\n")
                out_file.write(f"  Ground Truth: {gt_tokens}\n")
                out_file.write(f"  Prediction:   {predicted_tokens}\n")

                gt_counter = Counter(gt_tokens)
                pred_counter = Counter(predicted_tokens)
                exact_tp_per_token = {token: min(gt_counter[token], pred_counter[token]) for token in set(gt_counter) | set(pred_counter)}
                exact_tp = sum(exact_tp_per_token.values())
                
                leftover_gt = Counter({token: gt_counter[token] - exact_tp_per_token[token] for token in gt_counter if gt_counter[token] > exact_tp_per_token[token]})
                leftover_pred = Counter({token: pred_counter[token] - exact_tp_per_token[token] for token in pred_counter if pred_counter[token] > exact_tp_per_token[token]})
                leftover_gt_list, leftover_pred_list = list(leftover_gt.elements()), list(leftover_pred.elements())
                
                subword_tp = sum(1 for pred_token in leftover_pred_list if any(is_subword(pred_token, gt_token) for gt_token in leftover_gt_list))
                item_tp = exact_tp + (subword_tp / 2)
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

def process_folders(sample_folder, segmented_folder, output_folder):
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
        segmented_file_path = segmented_file

        # Create output folder for the domain
        domain_output_folder = os.path.join(output_folder, domain)
        os.makedirs(domain_output_folder, exist_ok=True)

        # Evaluate segmentation (prints everything you need in the console and writes to a text file)
        precision, recall, f1 = evaluate_segmentation(sample_file_path, segmented_file_path, domain_output_folder)

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
    output_folder = './evaluation_charts/roberta'

    process_folders(sample_folder, segmented_folder, output_folder)

    global_metrics = calculate_global_metrics('./evaluation_charts/roberta')
    print("Global Metrics:", global_metrics)

    # Generate and display the global metrics graph
    plot_global_metrics(output_folder)

    # Save all domain metrics into a single JSON file
    save_all_domain_metrics(output_folder)
