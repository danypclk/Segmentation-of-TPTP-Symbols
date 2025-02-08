import os
import re
import json
import torch
import subprocess
import sys
import importlib

# Attempt to import pyenchant, install if needed
try:
    import enchant
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyenchant"])
    import enchant

from transformers import BertForTokenClassification

##############################
# 1) CharLevelTokenizer
##############################
class CharLevelTokenizer:
    def __init__(self):
        # Example of simple ASCII-based vocab, from 32 to 126
        self.vocab = {chr(i): i - 32 for i in range(32, 127)}  # ASCII chars
        self.pad_token_id = 0
        self.unk_token_id = len(self.vocab)

    def encode(self, text):
        return [self.vocab.get(char, self.unk_token_id) for char in text]

    def decode(self, ids):
        rev_vocab = {v: k for k, v in self.vocab.items()}
        return "".join([rev_vocab.get(i, "?") for i in ids])

    def pad(self, sequences, max_length):
        return [
            seq + [self.pad_token_id] * (max_length - len(seq))
            for seq in sequences
        ]

##############################
# 2) Model Loading
##############################
def load_model_and_tokenizer(
    local_model_dir="bert-segmentation", 
    remote_model_name="danypereira264/bert-segmentation_2"
):
    """
    Load a fine-tuned BERT segmentation model. This function first checks if there is
    a local folder named 'bert-segmentation' (or the name you specify).
    If found, it loads the model from that local path.
    Otherwise, it downloads/loads from the Hugging Face Hub.
    """
    tokenizer = CharLevelTokenizer()
    if os.path.isdir(local_model_dir):
        print(f"Loading model from local directory: {local_model_dir}")
        model = BertForTokenClassification.from_pretrained(local_model_dir)
    else:
        print(f"Local directory '{local_model_dir}' not found. Loading model from '{remote_model_name}'.")
        model = BertForTokenClassification.from_pretrained(remote_model_name)

    model.eval()
    return model, tokenizer

##############################
# 3) Infer Segmentation
##############################
def infer_segmentation(model, tokenizer, input_text, max_length=128):
    """
    Perform inference on a given string to predict segmentation labels (0 or 1).
    Returns a list of integer labels (per character).
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tokenize input text
    input_ids = tokenizer.encode(input_text)
    input_ids = input_ids[:max_length]  # truncate if needed
    attention_mask = [1] * len(input_ids)

    # Pad
    input_ids = tokenizer.pad([input_ids], max_length=max_length)[0]
    attention_mask = tokenizer.pad([attention_mask], max_length=max_length)[0]

    # Convert to tensors
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)  # (1, max_len)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask).logits
        preds = torch.argmax(outputs, dim=-1).squeeze().cpu().numpy()

    # Only take labels for the original input length
    labels = preds[:len(input_text)]
    return labels.tolist()

##############################
# 3.1) Enchant Dictionary
##############################
# Create an English dictionary for checking recognized words
d = enchant.Dict("en_US")

##############################
# 4) Segment using BERT + Filter with PyEnchant
##############################
def segment_with_bert(model, tokenizer, symbol):
    """
    Splits a symbol into word-level segments using the predicted labels from 'infer_segmentation'.
    Label '1' indicates the start of a new word, '0' indicates continuation.
    
    We then:
        1) Remove non-alphabetic characters.
        2) Keep only words with at least 2 characters.
        3) Keep only words that are recognized by the pyenchant dictionary.
        
    Returns a list of valid segmented words.
    """
    if not symbol:
        return []

    labels = infer_segmentation(model, tokenizer, symbol)

    words = []
    current_word = []

    for char, lbl in zip(symbol, labels):
        if lbl == 1:
            # Start of a new word
            if current_word:
                words.append("".join(current_word))
            current_word = [char]
        else:
            # Continuation
            current_word.append(char)

    # Append the last word if it exists
    if current_word:
        words.append("".join(current_word))

    # Remove non-alphabetic characters
    clean_words = []
    for w in words:
        cleaned = re.sub(r"[^a-zA-Z]", "", w)
        if cleaned:
            clean_words.append(cleaned)

    # Filter out words that do not meet length >= 2 or are not recognized in the dictionary
    filtered_words = [
        cw for cw in clean_words 
        if len(cw) >= 2 and d.check(cw.lower())
    ]

    return filtered_words

##############################
# 5) Regex and Extraction
##############################
regex_patterns = {
    "predicates": r"(?<=[^\"fofcnf])([a-z][a-zA-Z0-9_!%&'*+\-/;<>\\^`{}]*)(?=[(])",
    "constants":  r"(?<=[,()\s])([a-z][a-zA-Z0-9_!%&'*+\-/;<>\\^`{}]*)(?=[,)\s])",
    "variables":  r"(?<=[,()\s])([A-Z][a-zA-Z0-9_!%&'*+\-/;<>\\^`{}]*)(?=[,)\s])"
}

# Patterns for quoted terms
quoted_predicate_pattern = r"(?<=')([^'()!=,\n]+)(?='[(])"
quoted_constant_pattern  = r"(?<=['])(?:(?!!=)[^'(),\n]+)(?='(?!\())"

def extract_matches(formulas, patterns, model, tokenizer):
    """
    1) Extract & remove quoted predicates ('p'(...) form).
    2) Extract & remove quoted constants ('c' not followed by '(').
    3) Skip the first two unquoted constants.
    4) Apply regex patterns to find unquoted predicates, constants, variables.
    5) Segment each symbol with the BERT model + filter with pyenchant.
    """
    extracted_data = {
        "predicates": {},
        "constants": {},
        "variables": {}
    }

    all_quoted_constants = []

    for formula in formulas:
        # ----------- Quoted Predicates -----------
        qp_matches = re.findall(quoted_predicate_pattern, formula)
        for qp in qp_matches:
            seg_qp = segment_with_bert(model, tokenizer, qp)
            if seg_qp:
                extracted_data["predicates"][qp] = seg_qp
            # Remove the `'qp'(` substring but keep '('
            formula = re.sub(rf"'{re.escape(qp)}'\(", "(", formula)

        # ----------- Quoted Constants -----------
        qc_matches = re.findall(quoted_constant_pattern, formula)
        for qc in qc_matches:
            if qc.strip():
                all_quoted_constants.append(qc)
            # Remove `'qc'`
            formula = re.sub(rf"'{re.escape(qc)}'", " ", formula)

        # ----------- Skip first 2 unquoted constants -----------
        unquoted_constants = re.findall(patterns["constants"], formula)
        unquoted_constant_count = 0

        tokens = re.split(r"(\W)", formula)
        for i, token in enumerate(tokens):
            if token in unquoted_constants:
                if unquoted_constant_count < 2:
                    # Remove by replacing with space
                    tokens[i] = " "
                    unquoted_constant_count += 1
                else:
                    pass
        modified_formula = "".join(tokens)

        # ----------- Apply regex to find unquoted symbols -----------
        for label, pat in patterns.items():
            matches = re.findall(pat, modified_formula)

            # Filter out unwanted 'fof' for predicates
            if label == "predicates":
                matches = [m for m in matches if not m.startswith("fof")]

            for match in matches:
                if match not in extracted_data[label]:
                    seg_match = segment_with_bert(model, tokenizer, match)
                    if seg_match:
                        extracted_data[label][match] = seg_match

    # Merge quoted constants
    for qc in all_quoted_constants:
        if qc not in extracted_data["constants"]:
            seg_qc = segment_with_bert(model, tokenizer, qc)
            if seg_qc:
                extracted_data["constants"][qc] = seg_qc

    return extracted_data

##############################
# 6) Main Execution
##############################
def main():
    # Example file paths
    input_file_path = r"formulas.txt"
    output_file_path = r".\extracted_formulas.json"

    # 1) Load model/tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # 2) Read formulas
    with open(input_file_path, "r", encoding="utf-8") as f:
        formulas = [line.strip() for line in f if line.strip()]

    # 3) Extract + segment
    extracted_data = extract_matches(formulas, regex_patterns, model, tokenizer)

    # 4) Save to JSON
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, indent=4)

    print(f"Extraction complete. Results saved to {output_file_path}.")

if __name__ == "__main__":
    main()
