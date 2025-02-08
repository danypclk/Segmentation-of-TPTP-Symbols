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
        return [
            seq + [self.pad_token_id] * (max_length - len(seq)) for seq in sequences
        ]

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
        input_ids = input_ids[: self.max_length]
        label = label[: self.max_length]

        input_ids = self.tokenizer.pad([input_ids], self.max_length)[0]
        label = self.tokenizer.pad([label], self.max_length)[0]

        # Generate attention_mask
        attention_mask = [1 if id != self.tokenizer.pad_token_id else 0 for id in input_ids]

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

# Split dataset
train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)

# Load pre-trained BERT model
model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.resize_token_embeddings(len(tokenizer.vocab) + 1)

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer and loss function
optimizer = AdamW(model.parameters(), lr=lr)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in tqdm(train_loader):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}")

# Evaluation loop
model.eval()
total_accuracy = 0
total_count = 0

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask).logits
        preds = torch.argmax(outputs, dim=-1)

        total_accuracy += (preds == labels).sum().item()
        total_count += labels.numel()

print(f"Validation Accuracy: {total_accuracy / total_count:.4f}")

# Save the model
model.save_pretrained("bert-segmentation")
