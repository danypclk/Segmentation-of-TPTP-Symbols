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

MODEL_DIR = "./roberta-segmentation_2"
HUGGINGFACE_MODEL_NAME = "danypereira264/roberta-segmentation_2"

# Debugging: Check if the directory exists
if os.path.exists(MODEL_DIR):
    print(f"✅ Model directory exists: {MODEL_DIR}")
else:
    print(f"❌ Model directory is MISSING: {MODEL_DIR}")

# Debugging: Check if model.safetensors exists instead of pytorch_model.bin
if os.path.exists(os.path.join(MODEL_DIR, "model.safetensors")):
    print(f"✅ Found model.safetensors in {MODEL_DIR}")
    model_file = "model.safetensors"
elif os.path.exists(os.path.join(MODEL_DIR, "pytorch_model.bin")):
    print(f"✅ Found pytorch_model.bin in {MODEL_DIR}")
    model_file = "pytorch_model.bin"
else:
    print(f"❌ No valid model file found in {MODEL_DIR}, downloading...")

    # Download model and tokenizer
    model = RobertaForTokenClassification.from_pretrained(HUGGINGFACE_MODEL_NAME)
    tokenizer = RobertaTokenizer.from_pretrained(HUGGINGFACE_MODEL_NAME)

    # Save model in `safetensors` format
    model.save_pretrained(MODEL_DIR, safe_serialization=True)
    tokenizer.save_pretrained(MODEL_DIR)

    print(f"✅ Model downloaded and saved in '{MODEL_DIR}'.")
    model_file = "model.safetensors"

# Load model configuration
config = RobertaConfig.from_pretrained(MODEL_DIR)

# Load model state dict based on format
if model_file == "model.safetensors":
    print(f"✅ Loading model weights from {MODEL_DIR}/model.safetensors")
    state_dict = load_file(os.path.join(MODEL_DIR, "model.safetensors"))
else:
    print(f"✅ Loading model weights from {MODEL_DIR}/pytorch_model.bin")
    state_dict = torch.load(os.path.join(MODEL_DIR, "pytorch_model.bin"), map_location="cpu")

# Initialize model
model = RobertaForTokenClassification(config)
model.load_state_dict(state_dict)

# Load tokenizer
tokenizer = RobertaTokenizer.from_pretrained(MODEL_DIR)

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
            if label == 1:
                if current_word:
                    words.append("".join(current_word))
                current_word = [char]
            else:
                current_word.append(char)
        if current_word:
            words.append("".join(current_word))
        # Remove special characters including underscores explicitly
        words = [re.sub(r'[^a-zA-Z]', '', word) for word in words if len(re.sub(r'[^a-zA-Z]', '', word)) > 1]
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
                    gt_tokens = predict_labels(original_string, model, tokenizer)
                    out_file.write(f"  Original:     {original_string}\n")
                    out_file.write(f"  Ground Truth: {gt_tokens}\n")
                    out_file.write(f"  Prediction:   Missing (no tokens predicted)\n")
                    total_fn += len(gt_tokens)
                    continue
                
                predicted_tokens = segmented_data[category][i]
                gt_tokens = predict_labels(original_string, model, tokenizer)

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
    """
    Save a bar chart of precision, recall, and F1-score for the given domain.
    """
    metrics = ['Precision', 'Recall', 'F1-Score']
    values = [precision, recall, f1]

    plt.figure()
    plt.bar(metrics, values)
    plt.ylim(0, 1)
    plt.title(f"Evaluation Metrics for {domain}")
    plt.ylabel("Score")

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

if __name__ == '__main__':
    sample_folder = './samples_of_each_domain'
    segmented_folder = './segmented_samples'
    output_folder = './evaluation_charts/roberta_2'

    process_folders(sample_folder, segmented_folder, output_folder)

    global_metrics = calculate_global_metrics('./evaluation_charts/roberta_2')
    print("Global Metrics:", global_metrics)
