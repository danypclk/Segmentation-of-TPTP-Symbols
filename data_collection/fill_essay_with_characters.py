import os
import random
import string

def add_random_characters(word):
    """Add random characters around the word."""
    num_specials = random.randint(1, 2)  # Number of special characters to add
    specials = ' '.join(random.choices(string.punctuation, k=num_specials))  # Special characters separated by spaces
    num_chars = random.randint(1, 5)  # Number of normal characters to add
    extra_chars = ''.join(random.choices(string.ascii_letters, k=num_chars))  # Normal characters concatenated together
    
    if random.choice([True, False]):  # Randomly prepend or append
        return f"{specials} {word} {extra_chars}"
    else:
        return f"{extra_chars} {word} {specials}"

def randomize_word(word):
    """Randomly apply transformations to a word."""
    if random.random() < 0.8:  # 80% chance for change
        # Capitalize the original word with 50% probability
        if random.random() < 0.5:  # 50% chance to capitalize original word
            word = word.capitalize()
        word = add_random_characters(word)  # Add random characters around the word
    return word

def process_text_files(input_folder, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Loop through all text files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):  # Process only .txt files
            input_file_path = os.path.join(input_folder, filename)
            output_file_path = os.path.join(output_folder, filename)  # Save with the same name
            
            try:
                with open(input_file_path, 'r', encoding='utf-8') as file:
                    lines = file.readlines()
                
                # Process all lines as a single continuous string
                all_text = ' '.join(line.strip() for line in lines)  # Remove \n and join lines with spaces
                words = all_text.split()  # Split the continuous text into words
                modified_words = [randomize_word(word) for word in words]
                
                # Join modified words into a single string without \n
                final_output = ' '.join(modified_words)
                
                with open(output_file_path, 'w', encoding='utf-8') as file:
                    file.write(final_output)  # Write as a single continuous string
                print(f"Processed file saved at: {output_file_path}")
            except UnicodeDecodeError as e:
                print(f"Error reading file {input_file_path}: {e}")

# Folder paths
input_folder = 'special_essays'  # Folder containing input .txt files
output_folder = 'special_essays_modified'  # Folder to save processed files

# Process all text files in the input folder
process_text_files(input_folder, output_folder)