import os
import re
import json

# Define the directory paths
formulas_dir = r".\formulas\tptp_library"
extracted_data_dir = r".\extracted_data\tptp_library"

# Ensure the output directory exists
os.makedirs(extracted_data_dir, exist_ok=True)

# Define regex patterns
regex_patterns = {
    "predicates": r"(?<=[^\"fofcnf])([a-z][a-zA-Z0-9_!%&'*+\-/;<>\\^`{}]*)(?=[(])",  # Match predicates by looking for lowercase strings before '('
    "constants": r"(?<=[,()\s])([a-z][a-zA-Z0-9_!%&'*+\-/;<>\\^`{}]*)(?=[,)\s])",  # Match constants by looking for lowercase strings before delimiters
    "variables": r"(?<=[,()\s])([A-Z][a-zA-Z0-9_!%&'*+\-/;<>\\^`{}]*)(?=[,)\s])"   # Match variables by looking for uppercase strings before delimiters
}

# You could define these at the top of your script
quoted_predicate_pattern = r"'([^']+)'(?=\()"   # `'someText'(` - Quoted text followed by '('
quoted_constant_pattern  = r"'([^']+)'(?!\()"   # `'someText'` - Quoted text NOT followed by '('

def extract_matches(formulas, patterns):
    """
    Extracts unique predicates, constants, and variables from a list of formulas.
    Returns a dictionary of sets: {"predicates": set(), "constants": set(), "variables": set()}.
    """

    extracted_data = {
        "predicates": set(),
        "constants": set(),
        "variables": set()
    }

    # Will collect all unquoted constants before we add them to the set.
    all_constants = []

    for formula in formulas:
        # 1) Quoted predicates: detect `'someText'(`, add to predicate set, remove from formula
        q_preds = re.findall(quoted_predicate_pattern, formula)
        for p in q_preds:
            extracted_data["predicates"].add(p)
            # Remove `'p'(` from the formula (roughly). 
            # In TPTP syntax, it might be `'p'(...)` so we replace `'p'(` with just a space or something similar.
            # Carefully replace all occurrences.
            formula = re.sub(rf"'{re.escape(p)}'\(", " ", formula)

        # 2) Quoted constants: detect `'someText'` not followed by '('
        q_consts = re.findall(quoted_constant_pattern, formula)
        for c in q_consts:
            all_constants.append(c)
            # Remove `'c'` from the formula
            formula = re.sub(rf"'{re.escape(c)}'", " ", formula)

        # 3) Identify unquoted constants and remove the first two (as in your original logic)
        unquoted_constants = re.findall(patterns["constants"], formula)
        unquoted_constant_count = 0

        tokens = re.split(r"(\W)", formula)  # Tokenize while preserving punctuation
        for i, token in enumerate(tokens):
            if token in unquoted_constants:
                if unquoted_constant_count < 2:
                    tokens[i] = " "  # Remove it by replacing with a space
                    unquoted_constant_count += 1
                else:
                    break  # stop after skipping two constants

        modified_formula = "".join(tokens)

        # 4) Apply existing patterns to find unquoted predicates, constants, and variables
        for label, pattern in patterns.items():
            matches = re.findall(pattern, modified_formula)

            # If it's a predicate, exclude "fof"
            if label == "predicates":
                matches = [m for m in matches if not m.startswith("fof")]

            if label == "constants":
                all_constants.extend(matches)
            else:
                extracted_data[label].update(matches)

    # Finally add all constants to the set
    extracted_data["constants"].update(all_constants)

    return extracted_data


# Process each file in the formulas directory
for filename in os.listdir(formulas_dir):
    if filename.endswith(".json"):
        # Read the JSON file (domain-level file)
        with open(os.path.join(formulas_dir, filename), "r", encoding="utf-8") as f:
            data = json.load(f)

        # Prepare one set of each category per domain
        domain_predicates = set()
        domain_constants = set()
        domain_variables = set()

        # For each page identifier, gather the formulas and extract
        for page_identifier, formulas in data.items():
            extracted = extract_matches(formulas, regex_patterns)
            domain_predicates.update(extracted["predicates"])
            domain_constants.update(extracted["constants"])
            domain_variables.update(extracted["variables"])

        # Build the final dictionary for the entire domain
        domain_data = {
            "predicates": sorted(domain_predicates),
            "constants": sorted(domain_constants),
            "variables": sorted(domain_variables)
        }

        # Determine the new filename for the extracted data
        output_filename = filename[:3] + ".json"
        output_path = os.path.join(extracted_data_dir, output_filename)

        # Write the aggregated data to a new JSON file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(domain_data, f, indent=4)

        print(f"Processed {filename} -> {output_filename}")

print("Extraction complete.")
