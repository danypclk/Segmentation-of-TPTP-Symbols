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

# Find instances where labels exceed the 512-token limit
long_label_instances = []
for idx, (phrase, label) in enumerate(data_instances):
    tokenized = tokenizer(list(phrase), is_split_into_words=True, truncation=True, max_length=512)
    max_len = len(tokenized["input_ids"])
    if len(label) > max_len:
        long_label_instances.append((idx, phrase, label, len(label), max_len))

# Report results
if long_label_instances:
    print("The following instances exceed the 512-token limit:")
    for idx, phrase, label, label_len, max_len in long_label_instances:
        print(f"Index: {idx}, Phrase: {phrase[:50]}..., Label Length: {label_len}, Allowed Max Length: {max_len}")
else:
    print("No instances exceed the 512-token limit. Everything looks fine!")
