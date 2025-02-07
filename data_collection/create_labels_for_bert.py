import os
import pandas as pd
import json

# Define the directories
input_folders = ["./processed_essays", "./processed_special_essays"]  # Folders containing the CSV files
output_folder = "./segmentation_jsons"  # Folder to save the JSON files

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Function to generate labels for segmentation
def create_labels(concatenated, separated):
    words = separated.split()  # Split separated sentence into words
    label = []
    pos = 0

    for word in words:
        for i, char in enumerate(word):
            if pos < len(concatenated) and concatenated[pos] == char:
                label.append(1 if i == 0 else 0)  # 1 for new word start, 0 otherwise
                pos += 1

    return label

# Process all CSV files in the input folders
for input_folder in input_folders:
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".csv"):  # Check for CSV files
            file_path = os.path.join(input_folder, file_name)
            df = pd.read_csv(file_path)

            # Initialize lists to store phrases and labels
            phrases = []
            labels = []

            # Process each row in the DataFrame
            for _, row in df.iterrows():
                concatenated = row['Concatenated']
                separated = row['Separated']
                phrases.append(concatenated)
                labels.append(create_labels(concatenated, separated))

            # Save phrases and labels in JSON format
            data = {"phrases": phrases, "labels": labels}
            
            # Create JSON file name
            json_file_name = os.path.splitext(file_name)[0] + ".json"
            json_file_path = os.path.join(output_folder, json_file_name)

            with open(json_file_path, "w") as f:
                json.dump(data, f, indent=4)  # Use indent for readability

print(f"All CSV files have been processed and JSON files are saved in '{output_folder}'.")
