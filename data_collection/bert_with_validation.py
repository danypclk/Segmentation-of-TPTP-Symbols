import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForTokenClassification
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Function to load all JSON files from a directory
def load_json_files(directory):
    phrases = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r") as file:
                data = json.load(file)
                phrases.extend(data["phrases"])
                labels.extend(data["labels"])
    return phrases, labels

# Load the dataset
directory = "training_set"
phrases, labels = load_json_files(directory)

# Character-level tokenizer
class CharLevelTokenizer:
    def __init__(self):
        self.vocab = {chr(i): i - 32 for i in range(32, 127)}  # ASCII characters
        self.pad_token_id = 0
        self.unk_token_id = len(self.vocab)

    def encode(self, text):
        return [self.vocab.get(char, self.unk_token_id) for char in text]

    def decode(self, ids):
        rev_vocab = {v: k for k, v in self.vocab.items()}
        return "".join([rev_vocab.get(i, "?") for i in ids])

    def pad(self, sequences, max_length):
        return [seq + [self.pad_token_id] * (max_length - len(seq)) for seq in sequences]

# Dataset class
class SegmentationDataset(Dataset):
    def __init__(self, phrases, labels, tokenizer, max_length):
        self.phrases = phrases
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.phrases)

    def __getitem__(self, idx):
        text = self.phrases[idx]
        label = self.labels[idx]

        # Tokenize the phrase and pad to max_length
        input_ids = self.tokenizer.encode(text)
        input_ids = input_ids[:self.max_length]
        label = label[:self.max_length]

        input_ids = self.tokenizer.pad([input_ids], self.max_length)[0]
        label = self.tokenizer.pad([label], self.max_length)[0]

        # Generate attention_mask
        attention_mask = [1 if token_id != self.tokenizer.pad_token_id else 0 for token_id in input_ids]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
        }

# Hyperparameters
max_length = 128
batch_size = 16
epochs = 30
lr = 5e-5

# Initialize tokenizer and dataset
tokenizer = CharLevelTokenizer()
dataset = SegmentationDataset(phrases, labels, tokenizer, max_length)

# Split dataset into training and validation sets
train_val_data, test_data = train_test_split(dataset, test_size=0.1, random_state=42)
train_data, val_data = train_test_split(train_val_data, test_size=0.1, random_state=42)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

# Load pre-trained BERT model for token classification
model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.resize_token_embeddings(len(tokenizer.vocab) + 1)

# Move model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=lr)

# Training and validation loop
for epoch in range(epochs):
    # --- Training ---
    model.train()
    total_train_loss = 0

    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.4f}")

    # --- Validation ---
    model.eval()
    total_val_loss = 0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Get model outputs; passing labels computes the loss
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_val_loss += loss.item()

            # Calculate accuracy
            preds = torch.argmax(logits, dim=-1)
            total_correct += (preds == labels).sum().item()
            total_tokens += labels.numel()

    avg_val_loss = total_val_loss / len(val_loader)
    accuracy = total_correct / total_tokens
    print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}")

# Save the model after training
model.save_pretrained("bert-segmentation")
