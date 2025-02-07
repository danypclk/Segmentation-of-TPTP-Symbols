import os
import json
import re
import pandas as pd

# Directory containing the JSON files
directory = "./extracted_data/tptp_library_with_id"

# Regular expression for detecting special symbols
special_symbol_pattern = re.compile(r"[^a-zA-Z0-9_']|(?<!\\)\\")

# Prepare a list to store results
results = []

# Iterate through all JSON files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".json"):  # Ensure we process only JSON files
        file_path = os.path.join(directory, filename)
        
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue
        
        # Process each problem in the JSON file
        for problem_name, details in data.items():
            print(f"Processing problem: {problem_name} from file: {filename}")  # Debugging line
            # Ensure that each symbol is correctly linked to its problem name
            for symbol_type, symbols in details.items():
                if symbol_type in ["predicates", "constants", "variables"]:
                    for symbol in symbols:
                        if special_symbol_pattern.search(symbol):
                            results.append({
                                "Source File": filename,
                                "Problem": problem_name, 
                                "Type": symbol_type.capitalize(), 
                                "Symbol": symbol
                            })

# Convert results to a DataFrame
df_results = pd.DataFrame(results)

# Save the results to a CSV file
df_results.to_csv("special_symbols_results.csv", index=False)

# Print completion message
print("Analysis complete. Results saved to 'special_symbols_results.csv'.")