import os
import re
import json
import wordninja
import enchant

import subprocess
import importlib

def ensure_package_installed(package, pip_name=None):
    """
    Ensures that a Python package is installed. If not, installs it.
    Args:
        package (str): The package name to check for in Python.
        pip_name (str): The name to use for pip installation, if different from `package`.
    """
    try:
        importlib.import_module(package)
    except ImportError:
        print(f"'{package}' not found. Installing...")
        subprocess.run(["pip", "install", pip_name or package], check=True)

# Ensure wordninja and enchant are installed
ensure_package_installed("wordninja")
ensure_package_installed("enchant", "pyenchant")

# Initialize PyEnchant English dictionary
dictionary = enchant.Dict("en_US")

# Define the input and output file paths
input_file_path = r"formulas.txt"  # Replace with your input file
output_file_path = r".\extracted_formulas.json"

# Define regex patterns for unquoted symbols
regex_patterns = {
    "predicates": r"(?<=[^\"fofcnf])([a-z][a-zA-Z0-9_!%&'*+\-/;<>\\^`{}]*)(?=[(])",  # Match predicates
    "constants":  r"(?<=[,()\s])([a-z][a-zA-Z0-9_!%&'*+\-/;<>\\^`{}]*)(?=[,)\s])",   # Match constants
    "variables":  r"(?<=[,()\s])([A-Z][a-zA-Z0-9_!%&'*+\-/;<>\\^`{}]*)(?=[,)\s])"    # Match variables
}

# Regex for quoted terms
# 1) Quoted predicate pattern: `'predicate'(` form
quoted_predicate_pattern = r"(?<=')([^'()!=,\n]+)(?='[(])"
# 2) Quoted constant pattern: `'constant'` not followed by '('
quoted_constant_pattern  = r"(?<=['])(?:(?!!=)[^'(),\n]+)(?='(?!\())"

def segment_and_validate_symbol(symbol):
    """
    Segments a symbol into words using WordNinja and keeps only valid words
    according to PyEnchant. Words must also have length > 2 and not be purely digits.
    """
    # 1) Use WordNinja to split the symbol
    segmented_words = wordninja.split(symbol)
    # 2) Filter using PyEnchant: valid English, length > 2, and not just digits
    valid_words = [
        w for w in segmented_words 
        if dictionary.check(w) and len(w) > 2 and not w.isdigit()
    ]
    return valid_words

def extract_matches(formulas, patterns):
    """
    Extracts predicates, constants, and variables from a list of formulas.

    Steps:
      1. Extract & remove quoted predicates ('p'(...) form).
      2. Extract & remove quoted constants ('c' not followed by '(').
      3. Skip the first two unquoted constants in each formula.
      4. Apply regex patterns to the modified formula to find unquoted symbols.
      5. Segment each symbol with WordNinja and store in extracted_data.
    """

    # We'll store each symbol -> segmented words in dicts, rather than sets,
    # so we don't lose the original string if you want to see it later.
    extracted_data = {
        "predicates": {},
        "constants": {},
        "variables": {}
    }

    all_quoted_constants = []  # For merging later

    for formula in formulas:
        # -----------------------
        # 1) Quoted Predicates
        # -----------------------
        qp_matches = re.findall(quoted_predicate_pattern, formula)
        for qp in qp_matches:
            seg_qp = segment_and_validate_symbol(qp)
            if seg_qp:
                extracted_data["predicates"][qp] = seg_qp

            # Remove the `'qp'(` substring, keeping '(' if it's meaningful
            formula = re.sub(rf"'{re.escape(qp)}'\(", "(", formula)

        # ----------------------
        # 2) Quoted Constants
        # ----------------------
        qc_matches = re.findall(quoted_constant_pattern, formula)
        for qc in qc_matches:
            # Segment each quoted constant
            if qc.strip():
                all_quoted_constants.append(qc)
            # Remove `'qc'` from the formula
            formula = re.sub(rf"'{re.escape(qc)}'", " ", formula)

        # ---------------------------------------
        # 3) Skip the first two UNQUOTED constants
        # ---------------------------------------
        unquoted_constants = re.findall(patterns["constants"], formula)
        unquoted_constant_count = 0

        # Tokenize formula while preserving punctuation/spaces
        tokens = re.split(r"(\W)", formula)
        for i, token in enumerate(tokens):
            if token in unquoted_constants:
                if unquoted_constant_count < 2:
                    # Remove by replacing with space
                    tokens[i] = " "
                    unquoted_constant_count += 1
                else:
                    pass  # Only remove the first two

        # Reconstruct the formula after removing first two constants
        modified_formula = "".join(tokens)

        # -----------------------
        # 4) Apply regex patterns
        # -----------------------
        for label, pattern in patterns.items():
            matches = re.findall(pattern, modified_formula)

            # Filter out unwanted 'fof' predicates
            if label == "predicates":
                matches = [m for m in matches if not m.startswith("fof")]

            # Segment each match
            for match in matches:
                if match not in extracted_data[label]:
                    valid_words = segment_and_validate_symbol(match)
                    if valid_words:
                        extracted_data[label][match] = valid_words

    # ---------------------------------
    # 5) Merge quoted constants at end
    # ---------------------------------
    for qc in all_quoted_constants:
        if qc not in extracted_data["constants"]:
            valid_words = segment_and_validate_symbol(qc)
            if valid_words:
                extracted_data["constants"][qc] = valid_words

    return extracted_data

def main():
    # Read formulas from file
    with open(input_file_path, "r", encoding="utf-8") as f:
        formulas = [line.strip() for line in f if line.strip()]

    # Extract matches (predicates, constants, variables)
    extracted_data = extract_matches(formulas, regex_patterns)

    # Write the extracted data to a JSON file
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, indent=4)

    print(f"Extraction complete. Results saved to {output_file_path}.")

if __name__ == "__main__":
    main()
