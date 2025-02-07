import os
import re
import json

import os
import re
import json

# Function to extract formulas from a single file
def extract_formulas(file_path):
    with open(file_path, 'r') as file:
        page_text = file.read()
        # Extract fof formulas
        fof_formulas = re.findall(r'fof\(.*?\)\.', page_text, re.DOTALL)
        # Extract cnf formulas
        cnf_formulas = re.findall(r'cnf\(.*?\)\.', page_text, re.DOTALL)
        print(f"Found {len(fof_formulas)} FOF formulas and {len(cnf_formulas)} CNF formulas in file: {file_path}")
        # Return both fof and cnf formulas as a combined list
        return fof_formulas + cnf_formulas

# Main function to process each domain and save JSON files
def process_domains(input_directory, output_directory, additional_directory):
    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    input_directory = os.path.abspath(input_directory)  # Get absolute path for Problems
    additional_directory = os.path.abspath(additional_directory)  # Get absolute path for Axioms
    print(f"Processing input directory: {input_directory}")
    print(f"Processing additional directory: {additional_directory}")

    # Process .p files in the Problems directory
    for domain in os.listdir(input_directory):
        domain_path = os.path.join(input_directory, domain)
        if os.path.isdir(domain_path):  # Ensure it's a directory
            print(f"Processing domain: {domain}")
            domain_formulas = {}

            # Process .p files within the domain
            for filename in os.listdir(domain_path):
                if filename.lower().endswith('.p') and ('+' in filename or '-' in filename):
                    file_path = os.path.join(domain_path, filename)
                    file_key = os.path.splitext(filename)[0]  # Remove .p extension for JSON key
                    print(f"Processing .p file: {file_path}")
                    try:
                        fof_formulas = extract_formulas(file_path)
                        if fof_formulas:  # Only add if there are formulas
                            domain_formulas[file_key] = fof_formulas
                    except Exception as e:
                        print(f"Error processing .p file {file_path}: {e}")

            # Save the domain JSON file if formulas were found
            if domain_formulas:
                output_file = os.path.join(output_directory, f"{domain}.json")
                with open(output_file, 'w') as json_file:
                    json.dump(domain_formulas, json_file, indent=4)
                print(f"Saved JSON for domain: {domain} -> {output_file}")

    # Process .ax files in the Axioms directory
    axiom_formulas = {}
    for filename in os.listdir(additional_directory):
        if filename.lower().endswith('.ax'):  # Check for .ax files
            file_path = os.path.join(additional_directory, filename)
            file_key = os.path.splitext(filename)[0]  # Remove .ax extension for JSON key
            print(f"Processing .ax file: {file_path}")
            try:
                fof_formulas = extract_formulas(file_path)
                if fof_formulas:  # Only add if there are formulas
                    axiom_formulas[file_key] = fof_formulas
            except Exception as e:
                print(f"Error processing .ax file {file_path}: {e}")

    # Save the axioms JSON file if formulas were found
    if axiom_formulas:
        output_file = os.path.join(output_directory, "axioms.json")
        with open(output_file, 'w') as json_file:
            json.dump(axiom_formulas, json_file, indent=4)
        print(f"Saved JSON for axioms -> {output_file}")

    print("Processing complete.")

# Example usage
input_directory = './TPTP/Problems' 
output_directory = './formulas/tptp_library'
additional_directory = './TPTP/Axioms'
process_domains(input_directory, output_directory, additional_directory)
