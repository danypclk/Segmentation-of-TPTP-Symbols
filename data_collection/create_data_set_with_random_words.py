import os
import pandas as pd
import re
import random
import string

# Utility Functions

def generate_random_string(min_length=1, max_length=5):
    """
    Generate a random string of letters (a-z, A-Z) 
    with length between min_length and max_length (inclusive).
    """
    length = random.randint(min_length, max_length)
    return ''.join(random.choice(string.ascii_letters) for _ in range(length))

def insert_random_strings(phrase, probability=0.5):
    """
    Inserts random strings (1–5 chars) between words in the phrase
    with a 'probability' chance at each boundary.
    Returns a tuple: (concatenated_version, separated_version).
    """
    # Split the phrase into words by whitespace
    words = phrase.split()
    
    # If there's 0 or 1 word, there's no space to insert random characters
    if len(words) <= 1:
        # "Separated" is the same as original,
        # "Concatenated" is the same but with no spaces
        return (phrase.replace(" ", ""), phrase)
    
    # Build the new versions step by step
    separated_version = words[0]  # first word
    concatenated_version = words[0]  # first word (no space)
    
    # Iterate over subsequent words
    for w in words[1:]:
        # Check if we insert random characters (50% chance by default)
        if random.random() < probability:
            rand_str = generate_random_string(1, 5)
            # Add to separated version (with spaces around the rand_str)
            separated_version += " " + rand_str + " " + w
            # Add to concatenated version (no spaces around the rand_str)
            concatenated_version += rand_str + w
        else:
            # No random insertion, just add a space + next word for the separated version
            separated_version += " " + w
            # Just concatenate for the concatenated version
            concatenated_version += w
    
    return (concatenated_version, separated_version)

def split_into_phrases(text):
    """
    Split text into phrases using periods and question marks only,
    excluding newlines.
    """
    # Use regular expression to split by periods or question marks
    phrases = re.split(r'[.?]+', text)
    # Remove leading/trailing whitespace from each phrase
    phrases = [phrase.strip() for phrase in phrases if phrase.strip()]
    return phrases

def create_datasets(phrases):
    """
    Creates a list of (Concatenated, Separated) tuples by:
    1) Splitting each phrase into words
    2) Randomly inserting strings 50% of the time between words
    3) Returning the results
    """
    dataset = []
    for phrase in phrases:
        concatenated, separated = insert_random_strings(phrase, probability=0.5)
        # Only add if both are non-empty
        if concatenated and separated:
            dataset.append((concatenated, separated))
    return dataset

def read_and_clean_file(file_path):
    """
    Reads a file and cleans it by replacing newlines with spaces.
    Customize or remove cleaning steps if needed.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    # Replace newlines with spaces
    cleaned_text = re.sub(r'[\n]', ' ', text)
    return cleaned_text

def process_file(file_path, output_dir):
    """
    Processes a single file:
    1) Reads and cleans the text
    2) Splits into phrases
    3) Inserts random strings in both segmented and concatenated forms
    4) Saves as CSV in output_dir
    """
    # Extract file name without extension
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Read and clean the text file
    cleaned_text = read_and_clean_file(file_path)
    
    # Split the text into phrases
    phrases = split_into_phrases(cleaned_text)
    
    # Create the dataset with concatenated + separated (random inserts)
    dataset = create_datasets(phrases)
    
    # Create a DataFrame from the results
    df = pd.DataFrame(dataset, columns=['Concatenated', 'Separated'])
    
    # Define output CSV file path
    output_csv_path = os.path.join(output_dir, f"{file_name}_random_words.csv")
    
    # Save the DataFrame to a CSV file
    df.to_csv(output_csv_path, index=False)
    print(f"CSV file created: {output_csv_path}")

# Main Execution (Example Usage)
if __name__ == "__main__":
    input_file = "./cleaned_essays/cleaned_words-of-each-domain.txt"
    output_dir = "./processed_essays/"

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Create an output filename by appending "_random_words" before the extension
    base_name = os.path.basename(input_file)
    name, ext = os.path.splitext(base_name)
    output_file = os.path.join(output_dir, name + "_random_words" + ext)

    print(f"Processing file: {input_file}")
    print(f"Saving to: {output_file}")

    # Pass the correct output file path to process_file
    process_file(input_file, output_dir)
