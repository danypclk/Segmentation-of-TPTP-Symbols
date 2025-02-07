import json
import re
import os

# Define the input and output directories
input_dir = r'formulas/tptp_library'
output_dir = r'parsed_formulas/tptp_library'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Define regex patterns
full_predicate_pattern = r"([^\s()]+)\(([^\s]+)\)"  # Full predicate pattern
predicate_name_pattern = r"([^\s()]+)\("  # Pattern to extract the predicate name
equation_pattern = r"(?<=\s)\([^\(\)]*\)"  # Pattern to extract equations

# Define regex for constants and variables outside predicates
constants_pattern = r"(?<=\s|^)[a-z][^\s\(\)]*(?=\s|$)"
variables_pattern = r"(?<=\s|^)[A-Z][^\s\(\)]*(?=\s|$)"

# Helper function to check if a token is a variable (capitalized)
def is_variable(token):
    return token[0].isupper()

# Helper function to classify a token (constant, variable, or function)
def classify_token(token, params_string, i, length):
    current_token = token
    while i < length and params_string[i] not in '(), ':
        current_token += params_string[i]
        i += 1
    return current_token.strip(), i - 1

# Recursive function to classify parameters inside a function (like a predicate)
def classify_function_parameters(params_string):
    variables = []
    constants = []
    functions = []
    current_function = []
    paren_count = 0
    i = 0
    length = len(params_string)

    while i < length:
        char = params_string[i]

        # Start of a function (when encountering '(')
        if char == '(':
            paren_count += 1
            current_function.append(char)

        # End of a function (when encountering ')')
        elif char == ')':
            paren_count -= 1
            current_function.append(char)

            # If paren_count reaches zero, it means the function is fully captured
            if paren_count == 0:
                function_str = ''.join(current_function)
                function_name, nested_variables, nested_constants, nested_functions = extract_function_name_and_parameters(function_str)
                functions.append({
                    "function_name": function_name,
                    "variables": nested_variables,
                    "constants": nested_constants,
                    "functions": nested_functions
                })
                current_function = []

        # If inside a function, continue capturing characters
        elif paren_count > 0:
            current_function.append(char)

        # Outside of functions, handle variables and constants
        elif paren_count == 0:
            if char.isupper():  # Potential variable or function
                token, i = classify_token(char, params_string, i + 1, length)
                if is_variable(token):
                    variables.append(token)
                else:
                    constants.append(token)
            elif char not in '(), ':
                token, i = classify_token(char, params_string, i, length)
                constants.append(token)

        # Move to the next character
        i += 1

    return variables, constants, functions

# Helper function to extract the name of the function and classify its parameters
def extract_function_name_and_parameters(function_string):
    function_name_pattern = r"([^\s()]+)\("  # Pattern to extract the function name
    function_match = re.search(function_name_pattern, function_string)

    if function_match:
        function_name = function_match.group(1)  # Extract function name
        # Extract parameters inside the parentheses
        parameters_start = function_string.find('(') + 1
        parameters_end = function_string.rfind(')')
        parameters = function_string[parameters_start:parameters_end]

        # Recursively classify parameters (could include more functions inside)
        variables, constants, functions = classify_function_parameters(parameters)

        return function_name, variables, constants, functions
    return None, [], [], []  # If no function name is found

# Function to extract constants and variables outside of predicates and equations
def extract_constants_and_variables_outside_predicates(formula):
    # Find constants and variables outside predicates using regex
    constants_outside_predicates = re.findall(constants_pattern, formula)
    variables_outside_predicates = re.findall(variables_pattern, formula)
    return variables_outside_predicates, constants_outside_predicates

# Updated function to extract multiple predicates and handle edge cases
def extract_predicate_name_and_parameters(formula):
    # Find all predicates in the formula
    full_predicate_matches = re.findall(full_predicate_pattern, formula)
    predicates = []
    for match in full_predicate_matches:
        predicate_name = match[0]  # The first element of the tuple is the predicate name
        parameters = match[1]      # The second element is the parameter string
        variables, constants, functions = classify_function_parameters(parameters)
        predicates.append({
            "predicate_name": predicate_name,
            "variables": variables,
            "constants": constants,
            "functions": functions
        })

    # Extract variables and constants outside predicates
    variables_outside, constants_outside = extract_constants_and_variables_outside_predicates(formula)

    # Handle equations if no predicates were found
    equation_match = re.search(equation_pattern, formula)
    equation = equation_match.group(0) if equation_match else None

    return {
        "predicates": predicates,
        "equation": equation,
        "variables_outside": variables_outside,
        "constants_outside": constants_outside
    }

# Updated function to process multiple formulas per key
def process_formulas(data):
    results = {}
    for key, formulas in data.items():
        results[key] = []  # Use a list to store multiple results
        for formula in formulas:
            # Print feedback for each formula being processed
            print(f"Processing formula: {formula}")
            extraction = extract_predicate_name_and_parameters(formula)
            results[key].append({
                "formula": formula,
                "predicates": extraction["predicates"],
                "equation": extraction["equation"],
                "variables_outside": extraction["variables_outside"],
                "constants_outside": extraction["constants_outside"]
            })
    return results

# Loop through all JSON files in the input directory and process them
for filename in os.listdir(input_dir):
    if filename.endswith('-formulas.json'):  # Only process formula files
        input_file_path = os.path.join(input_dir, filename)

        # Determine the output file path
        base_name = filename[:3]  # Use the first three letters for the new file name
        output_file_path = os.path.join(output_dir, f'{base_name}-parsed.json')

        print(f"Processing file: {filename}")

        # Load the input JSON file
        with open(input_file_path, 'r') as file:
            data = json.load(file)

        # Process the JSON data
        extracted_data = process_formulas(data)

        # Save the extracted data to a new JSON file
        with open(output_file_path, 'w') as output_file:
            json.dump(extracted_data, output_file, indent=4)

        print(f"Extracted data saved to {output_file_path}")
