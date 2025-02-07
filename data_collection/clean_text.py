import os
import string
import re

# Dictionary of contractions and their expanded forms
contractions = {
    "aren't": "are not",
    "can't": "cannot",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he's": "he is",
    "here's": "here is",
    "I'm": "I am",
    "isn't": "is not",
    "it's": "it is",
    "let's": "let us",
    "she's": "she is",
    "that's": "that is",
    "there's": "there is",
    "they're": "they are",
    "we're": "we are",
    "weren't": "were not",
    "what's": "what is",
    "who's": "who is",
    "you're": "you are",
    "wouldn't": "would not",
    "could've": "could have",
    "should've": "should have",
    "would've": "would have",
    "I'll": "I will",
    "you'll": "you will",
    "she'll": "she will",
    "he'll": "he will",
    "we'll": "we will",
    "they'll": "they will",
    "I'd": "I would",
    "you'd": "you would",
    "he'd": "he would",
    "she'd": "she would",
    "they'd": "they would",
    "we'd": "we would",
    "isn't": "is not",
    "aren't": "are not",
    "wasn't": "was not",
    "weren't": "were not",
    # Add more as needed
}

# Function to replace contractions with their expanded forms
def replace_contractions(text):
    contractions_re = re.compile(r'\b(' + '|'.join(contractions.keys()) + r')\b')
    return contractions_re.sub(lambda match: contractions[match.group(0)], text)

# Function to clean text
def clean_text(text):
    text = replace_contractions(text)
    text = text.lower()
    text = re.sub(r'[-,]+', ' ', text)
    text = text.replace('?', '.').replace('!', '.').replace(';', '.').replace(':', '.')
    allowed_symbols = {'.'}
    special_letters = "òèâïéèôâü"
    cleaned_text = ''.join([char for char in text if char in string.ascii_letters + string.digits + string.whitespace + special_letters or char in allowed_symbols])
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

# Function to read text from a file
def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Function to write cleaned text to a file
def write_cleaned_text(file_path, cleaned_text):
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(cleaned_text)

# Function to process files and save to a specific cleaned directory
def process_files_in_directory(directory, output_folder_name):
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_folder_name)
    os.makedirs(output_dir, exist_ok=True)

    for file_name in os.listdir(directory):
        if file_name.endswith(".txt") and not file_name.startswith("cleaned"):
            input_file_path = os.path.join(directory, file_name)
            output_file_path = os.path.join(output_dir, f"cleaned_{file_name}")
            
            # Read and clean the file
            text = read_text_file(input_file_path)
            cleaned_text = clean_text(text)
            
            # Write cleaned text to a new file in the output folder
            write_cleaned_text(output_file_path, cleaned_text)
            print(f"Processed and cleaned: {file_name}")

# Main function to run the script
if __name__ == "__main__":
    # Process essays folder
    essays_directory_path = "essays"  # Path to your essays folder
    process_files_in_directory(essays_directory_path, "cleaned_essays")
    
    # Process special_essays folder
    special_essays_directory_path = "special_essays"  # Path to your special_essays folder
    process_files_in_directory(special_essays_directory_path, "cleaned_special_essays")
