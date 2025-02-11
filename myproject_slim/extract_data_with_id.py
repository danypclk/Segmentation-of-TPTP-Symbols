import os
import re
import json

# Define the directory paths
formulas_dir = r".\formulas\tptp_library"
extracted_data_dir = r".\extracted_data\tptp_library_with_id"

# Ensure the output directory exists
os.makedirs(extracted_data_dir, exist_ok=True)

# Define regex patterns
regex_patterns = {
    "predicates": r"(?<=[^\"fofcnf])([a-z][a-zA-Z0-9_!%&'*+\-/;<>\\^`{}]+)(?=[(])",  # Match predicates by looking for lowercase strings before '('
    "constants": r"(?<=[,()\s])([a-z][a-zA-Z0-9_!%&'*+\-/;<>\\^`{}]+)(?=[,)\s])",  # Match constants by looking for lowercase strings before delimiters
    "variables": r"(?<=[,()\s])([A-Z][a-zA-Z0-9_!%&'*+\-/;<>\\^`{}]+)(?=[,)\s])"  # Match variables by looking for uppercase strings before delimiters
}

# Function to extract matches from formulas
def extract_matches(formulas, patterns):
    """
    Extracts unique predicates, constants, and variables from a list of formulas.
    - Quoted predicates: text between quotes immediately followed by '('
    - Quoted constants: text between quotes NOT followed by '('
    - Then removes first two unquoted constants from each formula
    - Finally applies regex patterns to find unquoted predicates, constants, variables
    """

    extracted_data = {
        "predicates": set(),
        "constants": set(),
        "variables": set()
    }

    # Patterns for quoted terms
    quoted_predicate_pattern = r"(?<=')([^'()!=,\n]+)(?='[(])"
    quoted_constant_pattern  = r"(?<=['])(?:(?!.*!=.*)[^'(),\n]+)(?='(?!\())"

    # Will store all quoted constants before adding them to 'constants'
    all_quoted_constants = []

    for formula in formulas:
        # Step 1: Extract and remove "quoted predicates" ('p'(...) form)
        qp_matches = re.findall(quoted_predicate_pattern, formula)
        for qp in qp_matches:
            extracted_data["predicates"].add(qp)
            # Remove the `'qp'(` substring from the formula
            # Keep the '(' if it's meaningful to the syntax
            formula = re.sub(rf"'{re.escape(qp)}'\(", "(", formula)

        # Step 2: Extract and remove "quoted constants" ('c' not followed by '(')
        qc_matches = re.findall(quoted_constant_pattern, formula)
        for qc in qc_matches:
            if qc:  # Ensure qc is not an empty string
                all_quoted_constants.append(qc)
                # Remove the `'qc'` substring
                formula = re.sub(rf"'{re.escape(qc)}'", " ", formula)

        # Step 3: Identify unquoted constants and remove the first two
        unquoted_constants = re.findall(patterns["constants"], formula)
        unquoted_constant_count = 0

        # Tokenize formula while preserving spaces and punctuation
        tokens = re.split(r"(\W)", formula)

        for i, token in enumerate(tokens):
            if token in unquoted_constants:
                if unquoted_constant_count < 2:
                    tokens[i] = " "  # remove/replace with space
                    unquoted_constant_count += 1
                else:
                    break  # stop after skipping two constants

        # Reconstruct the modified formula
        modified_formula = "".join(tokens)

        # Step 4: Apply regex patterns to extract unquoted predicates,constants, and variables
        for label, pattern in patterns.items():
            matches = re.findall(pattern, modified_formula)

        # Remove 'fof' predicates
        if label == "predicates":
            matches = [match for match in matches if not match.startswith("fof")]

        # Filter out empty matches before adding them to extracted_data
        matches = list(filter(None, matches))  # Remove empty strings

        if label == "constants":
            extracted_data["constants"].update(matches)
        else:
            extracted_data[label].update(matches)

    # Step 5: Merge quoted constants into extracted constants
    extracted_data["constants"].update(all_quoted_constants)

    # Convert sets to lists for JSON (or other) serialization
    return {label: list(data) for label, data in extracted_data.items()}


# Process each file in the formulas directory
for filename in os.listdir(formulas_dir):
    if filename.endswith(".json"):
        # Read the JSON file
        with open(os.path.join(formulas_dir, filename), "r", encoding="utf-8") as f:
            data = json.load(f)

        # Extract data for each page identifier
        extracted_data = {}
        for page_identifier, formulas in data.items():
            extracted_data[page_identifier] = extract_matches(formulas, regex_patterns)

        # Determine the new filename for the extracted data
        output_filename = filename[:3] + ".json"
        output_path = os.path.join(extracted_data_dir, output_filename)

        # Write the extracted data to a new JSON file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(extracted_data, f, indent=4)

        print(f"Processed {filename} -> {output_filename}")

print("Extraction complete.")
