import os
import json
import torch
from transformers import RobertaTokenizer

# Load the tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Function to load dataset
def load_json_files(directory):
    data_instances = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r") as file:
                data = json.load(file)
                for phrase, label in zip(data["phrases"], data["labels"]):
                    data_instances.append((phrase, label))
    return data_instances

# Set dataset directory
directory = "segmentation_jsons"
data_instances = load_json_files(directory)

# Remove instances where labels exceed the 512-token limit
filtered_instances = []
for phrase, label in data_instances:
    tokenized = tokenizer(list(phrase), is_split_into_words=True, truncation=True, max_length=512)
    max_len = len(tokenized["input_ids"])
    if len(label) <= max_len:
        filtered_instances.append((phrase, label))

# Save filtered dataset
filtered_data = {"phrases": [p for p, _ in filtered_instances], "labels": [l for _, l in filtered_instances]}
filtered_file_path = os.path.join("training_set", "filtered_dataset.json")
with open(filtered_file_path, "w") as file:
    json.dump(filtered_data, file, indent=4)

print(f"Filtered dataset saved to {filtered_file_path}")