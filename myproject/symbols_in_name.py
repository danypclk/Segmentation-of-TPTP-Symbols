import os
import json
import re
from collections import defaultdict

# Directory path
directory = r".\extracted_data\tptp_library"

# Dictionary to store one example occurrence for each special symbol per file
symbol_examples_per_file = defaultdict(dict)

# Regular expression to match special symbols (excluding letters, digits, and '_')
special_symbol_pattern = re.compile(r"[^a-zA-Z0-9_]")

# Loop through files in the directory
for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    
    # Read JSON file
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON file: {filename}")
                continue

        # Loop through lists in each JSON
        for list_key in data:
            if isinstance(data[list_key], list):
                for name in data[list_key]:
                    # Find special symbols in the name
                    found_symbols = special_symbol_pattern.findall(name)
                    for symbol in found_symbols:
                        # Only store the first occurrence of each symbol for this file
                        if symbol not in symbol_examples_per_file[filename]:
                            symbol_examples_per_file[filename][symbol] = {
                                "list": list_key,
                                "example": name
                            }

    except (IOError, FileNotFoundError) as e:
        print(f"Error reading file {filename}: {e}")

# Display the results
for filename, symbols in symbol_examples_per_file.items():
    print(f"File: {filename}")
    for symbol, occurrence in symbols.items():
        print(f"  Symbol: {symbol}")
        print(f"    List: {occurrence['list']}, Example: {occurrence['example']}")
    print()
