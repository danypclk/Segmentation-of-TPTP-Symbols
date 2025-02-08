import os
import re
import json
import torch
import subprocess
import importlib
from transformers import RobertaTokenizer, RobertaForTokenClassification

###################################
# Optional: PyEnchant Installation
###################################

USE_PYENCHANT = False  # Set to True if you want dictionary checks

if USE_PYENCHANT:
    def ensure_package_installed(package, pip_name=None):
        try:
            importlib.import_module(package)
        except ImportError:
            print(f"'{package}' not found. Installing...")
            subprocess.run(["pip", "install", pip_name or package], check=True)
    ensure_package_installed("enchant", "pyenchant")
    import enchant
    dictionary = enchant.Dict("en_US")
else:
    dictionary = None

###################################
# RoBERTa Segmentation Code
###################################

# Function to load the trained model and tokenizer
def load_model_and_tokenizer(
    model_path="danypereira264/roberta-segmentation_2",
    base_tokenizer="roberta-base"):
    """
    Load the pretrained tokenizer (based on roberta-base)
    and the fine-tuned RoBERTa segmentation model from the specified Hugging Face repository.
    If it isn't in your local cache, it will be downloaded automatically.
    """
    # Load tokenizer from roberta-base (can also load from the repository if it has its own tokenizer)
    tokenizer = RobertaTokenizer.from_pretrained(base_tokenizer)
    
    # Load the fine-tuned model
    model = RobertaForTokenClassification.from_pretrained(model_path)
    model.eval()
    
    return model, tokenizer

def segment_with_roberta(symbol, model, tokenizer):
    """
    Segments the input text (symbol) into words using a character-level 
    token classification approach with RoBERTa. Returns the list of segmented words.
    """
    # Tokenize input text at character level
    encoding = tokenizer(
        list(symbol),
        is_split_into_words=True,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        # Get predicted class per character
        preds = torch.argmax(logits, dim=-1).squeeze(0).tolist()

    # Remove padding predictions if present
    sequence_length = attention_mask.sum().item()
    preds = preds[:sequence_length]

    # The original snippet removed the last two predictions
    if len(preds) > 2:
        preds = preds[:-2]
    else:
        preds = []

    # Segment according to predicted labels
    words = []
    current_word = []
    for char, label in zip(symbol, preds):
        # '1' means start of new word, '0' means continuation
        if label == 1:
            if current_word:
                words.append("".join(current_word))
            current_word = [char]
        else:
            current_word.append(char)

    # Append any remaining word
    if current_word:
        words.append("".join(current_word))

    # Clean out non-alphabetical chars inside each chunk
    cleaned_words = []
    for w in words:
        cleaned = re.sub(r"[^a-zA-Z]", "", w)
        if cleaned:
            cleaned_words.append(cleaned)

    return cleaned_words

def segment_and_validate_symbol(symbol, model, tokenizer):
    """
    Segments a symbol using RoBERTa and optionally filters (if PyEnchant is used).
    - Only includes words with length > 2
    - Checks if the word is an English word (if dictionary is available)
    """
    segmented = segment_with_roberta(symbol, model, tokenizer)
    if dictionary:
        # Filter: must be valid English, length > 2, not just digits
        segmented = [
            w for w in segmented
            if dictionary.check(w) and len(w) > 2 and not w.isdigit()
        ]
    else:
        # If no dictionary, optionally just filter out length <= 1
        segmented = [w for w in segmented if len(w) > 1]
    return segmented

###################################
# Extraction with Quoted Patterns
###################################

def extract_matches(formulas, patterns, model, tokenizer):
    """
    Extracts (and segments) predicates, constants, and variables from each formula.
    Includes:
    - Quoted predicates: text between single quotes immediately followed by '('
    - Quoted constants: text between single quotes NOT followed by '('
    - Removes first two unquoted constants
    - Finally extracts unquoted predicates, constants, variables
    - Returns a dict with keys: 'predicates', 'constants', 'variables'
      and values: { original_symbol: [segmented_words], ... }
    """

    # We'll store the final data as dictionaries of { original_symbol -> [segment, ...] }
    # so that you can see the original symbol and its RoBERTa segmentation.
    extracted_data = {
        "predicates": {},
        "constants": {},
        "variables": {}
    }

    # Regex for quoted terms
    quoted_predicate_pattern = r"(?<=')([^'()!=,\n]+)(?='[(])"
    quoted_constant_pattern  = r"(?<=['])(?:(?!!=)[^'(),\n]+)(?='(?!\())"

    # Temporary storage for all quoted constants (will move them at the end)
    quoted_constants_list = []

    for formula in formulas:
        # 1) Quoted predicates ('p'(...) form)
        qp_matches = re.findall(quoted_predicate_pattern, formula)
        for qp in qp_matches:
            seg_qp = segment_and_validate_symbol(qp, model, tokenizer)
            if seg_qp:
                extracted_data["predicates"][qp] = seg_qp

            # Remove the `'qp'(` substring (but keep '(' if needed)
            formula = re.sub(rf"'{re.escape(qp)}'\(", "(", formula)

        # 2) Quoted constants ('c' not followed by '(')
        qc_matches = re.findall(quoted_constant_pattern, formula)
        for qc in qc_matches:
            if qc.strip():
                # We'll collect them first; merging happens after unquoted extraction
                quoted_constants_list.append(qc)

            # Remove `'qc'` from the formula
            formula = re.sub(rf"'{re.escape(qc)}'", " ", formula)

        # 3) Identify unquoted constants and remove the first two from the formula
        unquoted_constants = re.findall(patterns["constants"], formula)
        unquoted_constant_count = 0

        # Split the formula into tokens (preserving delimiters in group 1)
        tokens = re.split(r"(\W)", formula)
        for i, token in enumerate(tokens):
            if token in unquoted_constants:
                if unquoted_constant_count < 2:
                    # "Remove" by replacing with a space
                    tokens[i] = " "
                    unquoted_constant_count += 1
                else:
                    # Only remove the first two, leave others as is
                    pass

        # Reconstruct the modified formula after removing first two unquoted constants
        modified_formula = "".join(tokens)

        # 4) Extract unquoted predicates, constants, variables from the modified formula
        for label, pat in patterns.items():
            matches = re.findall(pat, modified_formula)

            # E.g., remove 'fof' for predicates if that's required
            if label == "predicates":
                matches = [m for m in matches if not m.startswith("fof")]

            for match in matches:
                if match not in extracted_data[label]:
                    seg_match = segment_and_validate_symbol(match, model, tokenizer)
                    if seg_match:
                        extracted_data[label][match] = seg_match

    # 5) Merge quoted constants into extracted_data["constants"]
    #    Perform segmentation on each one before adding
    for qc in quoted_constants_list:
        if qc not in extracted_data["constants"]:
            seg_qc = segment_and_validate_symbol(qc, model, tokenizer)
            if seg_qc:
                extracted_data["constants"][qc] = seg_qc

    return extracted_data

###################################
# Putting It All Together
###################################

def main():
    # Example usage
    input_file_path = "formulas.txt"
    output_file_path = "./extracted_formulas.json"

    regex_patterns = {
        "predicates": r"(?<=[^\"fofcnf])([a-z][a-zA-Z0-9_!%&'*+\-/;<>\\^`{}]+)(?=[(])",
        "constants":  r"(?<=[,()\s])([a-z][a-zA-Z0-9_!%&'*+\-/;<>\\^`{}]+)(?=[,)\s])",
        "variables":  r"(?<=[,()\s])([A-Z][a-zA-Z0-9_!%&'*+\-/;<>\\^`{}]+)(?=[,)\s])"
    }

    # 1) Load the RoBERTa model + tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # 2) Read formulas
    with open(input_file_path, "r", encoding="utf-8") as f:
        formulas = [line.strip() for line in f if line.strip()]

    # 3) Extract + Segment
    extracted_data = extract_matches(formulas, regex_patterns, model, tokenizer)

    # 4) Write JSON
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, indent=4)

    print(f"Extraction complete. See '{output_file_path}' for results.")

if __name__ == "__main__":
    main()
