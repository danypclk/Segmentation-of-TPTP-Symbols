from rapidfuzz.distance import Levenshtein
import json
import enchant
from collections import Counter
import os
import math
import matplotlib.pyplot as plt
import torch

# Initialize PyEnchant English dictionary
d = enchant.Dict("en_US")

# Load the BERT model and tokenizer globally
from transformers import BertForTokenClassification

# Define the CharLevelTokenizer class
class CharLevelTokenizer:
    def __init__(self):
        self.vocab = {chr(i): i - 32 for i in range(32, 127)}  # ASCII characters
        self.pad_token_id = 0
        self.unk_token_id = len(self.vocab)

    def encode(self, text):
        return [self.vocab.get(char, self.unk_token_id) for char in text]

    def decode(self, ids):
        rev_vocab = {v: k for k, v in self.vocab.items()}
        return "".join([rev_vocab.get(i, "?") for i in ids])

    def pad(self, sequences, max_length):
        return [
            seq + [self.pad_token_id] * (max_length - len(seq)) for seq in sequences
        ]

def segment_text(text, labels):
    """
    Segment text into words based on the labels.

    :param text: Input string
    :param labels: List of labels corresponding to each character in the text
    :return: List of segmented words
    """
    words = []
    current_word = []

    for char, label in zip(text, labels):
        if label == 1:  # Start of a new word
            if current_word:
                words.append("".join(current_word))
            current_word = [char]
        else:  # Continuation of the current word
            current_word.append(char)

    # Add the last word if any
    if current_word:
        words.append("".join(current_word))

    return words

# Load the BERT model and tokenizer globally
bert_model = BertForTokenClassification.from_pretrained("bert-segmentation_2")
bert_tokenizer = CharLevelTokenizer()

def infer_segmentation(model, tokenizer, input_text, max_length=128):
    """
    Perform inference on a given input string to predict segmentation labels.
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tokenize the input text
    input_ids = tokenizer.encode(input_text)
    input_ids = input_ids[:max_length]  # Truncate if longer than max_length
    attention_mask = [1] * len(input_ids)

    # Pad input_ids and attention_mask
    input_ids = tokenizer.pad([input_ids], max_length=max_length)[0]
    attention_mask = tokenizer.pad([attention_mask], max_length=max_length)[0]

    # Convert to PyTorch tensors
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)  # Batch size = 1
    attention_mask = torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask).logits
        preds = torch.argmax(outputs, dim=-1).squeeze().cpu().numpy()

    # Remove padding and return the labels corresponding to the input text
    labels = preds[:len(input_text)]  # Align labels to the original text
    return labels.tolist()

def infer_segmentation_with_words(model, tokenizer, input_text, max_length=128):
    """
    Perform inference with BERT for segmentation and return segmented words.
    """
    # Get token-level predictions
    predicted_labels = infer_segmentation(model, tokenizer, input_text, max_length)
    
    # Convert character-level predictions into words
    segmented_words = segment_text(input_text, predicted_labels)
    return segmented_words

def is_subword(subword, word, error_margin=0.3):

    # Compute Levenshtein distance
    distance = Levenshtein.distance(subword, word)
    threshold = math.ceil(error_margin * max(len(subword), len(word)))

    return distance <= threshold

def evaluate_segmentation(sample_file, segmented_file, output_folder, max_subword_distance=1):
    """
    Evaluates the segmentation predictions in two stages:
      1) Exact Match (Counters)
      2) Subword or Fuzzy Match on leftover tokens

    Saves debugging info for each item in the console and writes it to a text file.
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

        def get_ground_truth_tokens(text):
            """
            Use BERT-based segmentation to get ground truth tokens.
            """
            # Remove underscores
            joined = text.replace('_', '')

            # Use BERT-based segmentation
            segmented_words = infer_segmentation_with_words(bert_model, bert_tokenizer, joined)

            # Optional: Filter valid English words
            valid_words = [word for word in segmented_words if d.check(word) and not word.isdigit() and len(word) > 1]
            return valid_words

        for category in ['predicates', 'constants', 'variables']:
            out_file.write(f"--- Category: {category} ---\n")
            for i, original_string in enumerate(sample_data[category]):
                if i >= len(segmented_data[category]):
                    gt_tokens = get_ground_truth_tokens(original_string)
                    out_file.write(f"  Original:     {original_string}\n")
                    out_file.write(f"  Ground Truth: {gt_tokens}\n")
                    out_file.write(f"  Prediction:   Missing (no tokens predicted)\n")
                    total_fn += len(gt_tokens)
                    continue

                predicted_tokens = segmented_data[category][i]
                gt_tokens = get_ground_truth_tokens(original_string)

                out_file.write(f"  Original:     {original_string}\n")
                out_file.write(f"  Ground Truth: {gt_tokens}\n")
                out_file.write(f"  Prediction:   {predicted_tokens}\n")

                # Calculate precision, recall, and F1
                gt_counter = Counter(gt_tokens)
                pred_counter = Counter(predicted_tokens)

                exact_tp_per_token = {}
                for token in set(gt_counter.keys()).union(pred_counter.keys()):
                    matched = min(gt_counter[token], pred_counter[token])
                    exact_tp_per_token[token] = matched

                exact_tp = sum(exact_tp_per_token.values())

                leftover_gt_counter = Counter({token: gt_counter[token] - exact_tp_per_token[token] for token in gt_counter if gt_counter[token] > exact_tp_per_token[token]})
                leftover_pred_counter = Counter({token: pred_counter[token] - exact_tp_per_token[token] for token in pred_counter if pred_counter[token] > exact_tp_per_token[token]})

                leftover_gt_list = list(leftover_gt_counter.elements())
                leftover_pred_list = list(leftover_pred_counter.elements())

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

        # Final precision, recall, and F1 calculations
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
    output_folder = './evaluation_charts/bert_2'

    process_folders(sample_folder, segmented_folder, output_folder)

    global_metrics = calculate_global_metrics('./evaluation_charts/bert_2')
    print("Global Metrics:", global_metrics)
