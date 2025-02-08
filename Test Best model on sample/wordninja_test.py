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
        # Check if the package is already installed
        importlib.import_module(package)
        #print(f"'{package}' is already installed.")
    except ImportError:
        # Install the package if it's not found
        print(f"'{package}' not found. Installing...")
        subprocess.run(["pip", "install", pip_name or package], check=True)

# Use the function to ensure required libraries are installed
ensure_package_installed("wordninja")
ensure_package_installed("enchant", "pyenchant")

# Initialize PyEnchant English dictionary
dictionary = enchant.Dict("en_US")

# Define the input and output file paths
input_file_path = r"formulas.txt"  # Replace with your input file
output_file_path = r".\extracted_formulas.json"

# Define regex patterns
regex_patterns = {
    "predicates": r"(?<=[^\"fofcnf])([a-z][a-zA-Z0-9_!%&'*+\-/;<>\\^`{}]*)(?=[(])",  # Match predicates
    "constants": r"(?<=[,()\s])([a-z][a-zA-Z0-9_!%&'*+\-/;<>\\^`{}]*)(?=[,)\s])",   # Match constants
    "variables": r"(?<=[,()\s])([A-Z][a-zA-Z0-9_!%&'*+\-/;<>\\^`{}]*)(?=[,)\s])"    # Match variables
}

# Segment and validate symbol using WordNinja and PyEnchant
def segment_and_validate_symbol(symbol):
    """
    Segments a symbol into words using WordNinja and keeps only valid words
    according to PyEnchant. Words must also have a length > 2.
    """
    segmented_words = wordninja.split(symbol)
    valid_words = [word for word in segmented_words if dictionary.check(word) and len(word) > 2 and not word.isdigit()]
    return valid_words

# Function to extract matches from formulas and map symbols to valid words
def extract_matches(formulas, patterns):
    extracted_data = {
        "predicates": {},
        "constants": {},
        "variables": {}
    }

    for formula in formulas:
        for label, pattern in patterns.items():
            matches = re.findall(pattern, formula)
            # Filter out unwanted predicates starting with 'fof'
            if label == "predicates":
                matches = [match for match in matches if not match.startswith("fof")]

            # Skip the first two constants if the label is 'constants'
            if label == "constants":
                matches = matches[2:]  # Exclude the first two constants

            for match in matches:
                # Segment and validate the symbol
                valid_words = segment_and_validate_symbol(match)
                if valid_words:
                    # Add the match and its valid words as a key-value pair
                    extracted_data[label][match] = valid_words

    return extracted_data

# Read the formulas from the input file
with open(input_file_path, "r", encoding="utf-8") as f:
    formulas = f.readlines()

# Clean and strip each formula
formulas = [formula.strip() for formula in formulas if formula.strip()]

# Extract matches from the formulas
extracted_data = extract_matches(formulas, regex_patterns)

# Write the extracted data to a JSON file
os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
with open(output_file_path, "w", encoding="utf-8") as f:
    json.dump(extracted_data, f, indent=4)

print(f"Extraction complete. Results saved to {output_file_path}.")