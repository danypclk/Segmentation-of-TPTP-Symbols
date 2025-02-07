import os
import pandas as pd
import re

# Function to split text into phrases using periods and question marks only
def split_into_phrases(text):
    # Use regular expression to split by periods or question marks, excluding newlines
    phrases = re.split(r'[.?]+', text)
    # Remove leading/trailing whitespace from each phrase
    phrases = [phrase.strip() for phrase in phrases if phrase.strip()]
    return phrases

# Function to create two datasets: concatenated and separated phrases, discarding empty ones
def create_datasets(phrases):
    dataset = []
    for phrase in phrases:
        # Create concatenated version by removing spaces
        concatenated = phrase.replace(" ", "")
        separated = phrase
        # Only add to the dataset if neither version is empty
        if concatenated and separated:
            dataset.append((concatenated, separated))
    return dataset

# Function to clean and prepare text from a file
def read_and_clean_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    # Retain all characters except newlines (or customize as needed)
    cleaned_text = re.sub(r'[\n]', ' ', text)
    return cleaned_text

# Function to process a single file
def process_file(file_path, output_dir):
    # Extract file name without extension
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Read and clean the text file
    cleaned_text = read_and_clean_file(file_path)
    
    # Split the text into phrases
    phrases = split_into_phrases(cleaned_text)
    
    # Create the datasets with concatenated and separated versions
    dataset = create_datasets(phrases)
    
    # Create a dataframe with the results
    df = pd.DataFrame(dataset, columns=['Concatenated', 'Separated'])
    
    # Define output CSV file path
    output_csv_path = os.path.join(output_dir, f"{file_name}_phrases.csv")
    
    # Save the dataframe to a CSV file
    df.to_csv(output_csv_path, index=False)
    print(f"CSV file created: {output_csv_path}")

# Main function to process all files in a given input directory
def process_directory(input_dir, output_dir, prefix_filter=""):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Loop through all files in the input directory
    for file_name in os.listdir(input_dir):
        # Process only files starting with the specified prefix and ending with '.txt'
        if file_name.startswith(prefix_filter) and file_name.endswith(".txt"):
            file_path = os.path.join(input_dir, file_name)
            process_file(file_path, output_dir)

# Main execution
if __name__ == "__main__":
    # Directories for processing
    input_dirs = {
        "cleaned_essays": "processed_essays",  # Source: cleaned_essays -> Destination: processed_essays
        "special_essays_modified": "processed_special_essays",  # Source: special_essays_modified -> Destination: processed_special_essays
    }
    
    # Process each directory
    for input_dir, output_dir in input_dirs.items():
        print(f"Processing directory: {input_dir}")
        # Apply prefix filter only for 'cleaned_essays'
        prefix_filter = "cleaned" if "cleaned_essays" in input_dir else ""
        process_directory(input_dir, output_dir, prefix_filter=prefix_filter)
