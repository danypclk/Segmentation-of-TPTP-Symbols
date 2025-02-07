import os
import json
import re

# Directory path
directory = r".\extracted_data\tptp_library"

# Set to store unique special symbols
special_symbols = set()

# Regular expression to match special symbols (excluding letters, digits, and '_')
special_symbol_pattern = re.compile(r"[^a-zA-Z0-9_]")

# Loop through files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".json"):
        filepath = os.path.join(directory, filename)
        
        # Read JSON file
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
            # Loop through lists in each JSON
            for list_key in data:
                if isinstance(data[list_key], list):
                    for name in data[list_key]:
                        # Find special symbols in the name
                        found_symbols = special_symbol_pattern.findall(name)
                        special_symbols.update(found_symbols)

# Print unique special symbols
print("Special symbols found:", sorted(special_symbols))
