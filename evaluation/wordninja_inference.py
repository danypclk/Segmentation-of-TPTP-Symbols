import json
import wordninja
import enchant
from collections import Counter

# For fuzzy similarity:
# pip install rapidfuzz
from rapidfuzz import fuzz

# Initialize PyEnchant English dictionary
d = enchant.Dict("en_US")

def fuzzy_similarity(s1, s2):
    """
    Compute a similarity score between 0.0 and 1.0 using RapidFuzz.
    We'll use 'fuzz.ratio' which is basically a Levenshtein-based measure
    scaled to 0-100. We'll convert it to 0.0-1.0 by dividing by 100.
    """
    return fuzz.ratio(s1, s2) / 100.0

def evaluate_segmentation(sample_file, segmented_file, fuzzy_threshold=0.8):
    """
    Evaluates the segmentation predictions in two stages:
      1) Exact Match (Counters)
      2) Fuzzy Match on leftover tokens

    NOTE: 'fuzzy_threshold' is the minimum similarity required
          to consider two leftover tokens as a fuzzy match.
    """

    # Load the JSON data
    with open(sample_file, 'r', encoding='utf-8') as f:
        sample_data = json.load(f)
    with open(segmented_file, 'r', encoding='utf-8') as f:
        segmented_data = json.load(f)

    # Keep track of total TPs (exact + fuzzy), FPs, FNs across all domains
    total_tp = 0
    total_fp = 0
    total_fn = 0

    def get_ground_truth_tokens(text):
        """
        Remove underscores, then apply wordninja.
        Keep only valid English words (PyEnchant).
        """
        joined = text.replace('_', '')
        tokens = wordninja.split(joined)
        return [word for word in tokens if d.check(word) and not word.isdigit()]

    # Evaluate each category: predicates, constants, variables
    for category in ['predicates', 'constants', 'variables']:
        print(f"Category: {category}")

        # Iterate through each item in this category
        for i, original_string in enumerate(sample_data[category]):
            if i >= len(segmented_data[category]):
                # If there's no prediction, everything is missed
                gt_tokens = get_ground_truth_tokens(original_string)
                print(f"  Original:     {original_string}")
                print(f"  Ground Truth: {gt_tokens}")
                print(f"  Prediction:   Missing (all tokens missed)")
                total_fn += len(gt_tokens)
                continue

            predicted_tokens = segmented_data[category][i]
            gt_tokens = get_ground_truth_tokens(original_string)

            # Print debug
            print(f"  Original:     {original_string}")
            print(f"  Ground Truth: {gt_tokens}")
            print(f"  Prediction:   {predicted_tokens}")

            # 1) Exact match (Counter-based) -----------------------------
            gt_counter = Counter(gt_tokens)
            pred_counter = Counter(predicted_tokens)

            # We'll track how many EXACT TPs we found for each token
            exact_tp_per_token = {}
            for token in set(gt_counter.keys()).union(pred_counter.keys()):
                matched = min(gt_counter[token], pred_counter[token])
                exact_tp_per_token[token] = matched

            # Sum up exact matches
            exact_tp = sum(exact_tp_per_token.values())

            # We'll remove those matched tokens from leftover counters
            leftover_gt_counter = Counter()
            leftover_pred_counter = Counter()

            for token in gt_counter:
                leftover_count = gt_counter[token] - exact_tp_per_token[token]
                if leftover_count > 0:
                    leftover_gt_counter[token] += leftover_count

            for token in pred_counter:
                leftover_count = pred_counter[token] - exact_tp_per_token[token]
                if leftover_count > 0:
                    leftover_pred_counter[token] += leftover_count

            # Now let's do fuzzy matching on leftover tokens only
            # 2) Fuzzy Match --------------------------------------------
            # Convert counters to lists so we can pop matched tokens
            leftover_gt_list = []
            for t, c in leftover_gt_counter.items():
                leftover_gt_list.extend([t] * c)

            leftover_pred_list = []
            for t, c in leftover_pred_counter.items():
                leftover_pred_list.extend([t] * c)

            fuzzy_tp = 0

            # For each leftover predicted token, find the best fuzzy match
            # in leftover GT. If above threshold, consider it a match
            for pred_token in leftover_pred_list:
                # Find best match among leftover_gt_list
                best_match = None
                best_score = 0.0
                best_idx = None

                for idx, gt_token in enumerate(leftover_gt_list):
                    score = fuzzy_similarity(pred_token, gt_token)
                    if score > best_score:
                        best_score = score
                        best_match = gt_token
                        best_idx = idx

                # If best_score exceeds fuzzy_threshold, it's a fuzzy match
                if best_score >= fuzzy_threshold and best_match is not None:
                    # Count it as a fuzzy true positive
                    fuzzy_tp += 1
                    # Remove that GT token from the leftover list
                    leftover_gt_list.pop(best_idx)
                else:
                    # No fuzzy match found -> remains leftover FP
                    pass

            # Now we can compute final counts for this item:
            # Exact TPs + fuzzy TPs
            item_tp = exact_tp + fuzzy_tp

            # Remaining leftover predicted items are all FPs
            # leftover_pred_list had len(...) elements. We matched 'fuzzy_tp' of them
            # so leftover predicted (unmatched) = len(leftover_pred_list) - fuzzy_tp
            item_fp = (len(leftover_pred_list) - fuzzy_tp)

            # Remaining leftover ground-truth tokens are all FNs
            item_fn = len(leftover_gt_list)

            # Add to global totals
            total_tp += item_tp
            total_fp += item_fp
            total_fn += item_fn

            # Print summary for this item
            print(f"    Exact TP: {exact_tp}, Fuzzy TP: {fuzzy_tp}, FP: {item_fp}, FN: {item_fn}")

    # After all items in all categories, compute final metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return precision, recall, f1_score


if __name__ == '__main__':
    sample_file = './samples_of_each_domain/agt_samples.json'
    segmented_file = './segmented_samples/agt_samples.json'

    precision, recall, f1 = evaluate_segmentation(
        sample_file,
        segmented_file,
        fuzzy_threshold=0.8  # you can tweak this threshold
    )

    print("\nEvaluation Results (Exact + Fuzzy):")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-score:  {f1:.4f}")
