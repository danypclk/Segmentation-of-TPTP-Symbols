import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
directory = "segmentation_jsons"
phrases, labels = load_json_files(directory)

# Split into training and testing sets
train_phrases, test_phrases, train_labels, test_labels = train_test_split(
    phrases, labels, test_size=0.2, random_state=42
)

# Create a character-to-index mapping
all_chars = set("".join(phrases))
char_to_idx = {char: idx for idx, char in enumerate(sorted(all_chars))}

# Add the special padding token to the mapping
char_to_idx["<PAD>"] = len(char_to_idx)

# Dataset class
class CharSegmentationDataset(Dataset):
    def __init__(self, phrases, labels, char_to_idx):
        self.phrases = phrases
        self.labels = labels
        self.char_to_idx = char_to_idx

    def __len__(self):
        return len(self.phrases)

    def __getitem__(self, idx):
        phrase = self.phrases[idx]
        label = self.labels[idx]
        # Convert characters to indices
        input_ids = torch.tensor(
            [self.char_to_idx[char] for char in phrase], dtype=torch.long
        )
        labels = torch.tensor(label, dtype=torch.float)
        return input_ids, labels

# Collate function for dynamic padding
def collate_fn(batch, char_to_idx):
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    max_len = max(len(x) for x in inputs)

    # Pad inputs and targets
    padded_inputs = torch.stack(
        [torch.cat([x, torch.full((max_len - len(x),), char_to_idx["<PAD>"], dtype=torch.long)]) for x in inputs]
    )
    padded_targets = torch.stack(
        [torch.cat([y, torch.full((max_len - len(y),), 0, dtype=torch.float)]) for y in targets]
    )
    return padded_inputs, padded_targets

# Create datasets and dataloaders
train_dataset = CharSegmentationDataset(train_phrases, train_labels, char_to_idx)
test_dataset = CharSegmentationDataset(test_phrases, test_labels, char_to_idx)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=lambda batch: collate_fn(batch, char_to_idx))
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=lambda batch: collate_fn(batch, char_to_idx))

# Model definition
class CharSegmentationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(CharSegmentationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.transformer = nn.Transformer(
            embed_dim, nhead=8, num_encoder_layers=4, batch_first=True
        )
        self.fc = nn.Linear(embed_dim, 1)  # Binary classification for each character
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)  # Shape: (batch_size, seq_len, embed_dim)
        x = self.transformer(x, x)  # Shape: (batch_size, seq_len, embed_dim)
        x = self.fc(x)  # Shape: (batch_size, seq_len, 1)
        x = self.sigmoid(x)  # Shape: (batch_size, seq_len, 1)
        return x

# Training setup
vocab_size = len(char_to_idx)
embed_dim = 128
hidden_dim = 256

model = CharSegmentationModel(vocab_size, embed_dim, hidden_dim)
criterion = nn.BCELoss()  # Binary cross-entropy for multi-label classification
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop with detailed epoch and batch logging
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        print(f"Epoch {epoch + 1}/{epochs}")
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(-1), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # Print loss for each batch
            print(f"  Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        # Print total loss for the epoch
        print(f"Epoch {epoch + 1} completed. Total Loss: {total_loss:.4f}\n")

train_model(model, train_loader, criterion, optimizer, epochs=1)

# Evaluation
def evaluate_model(model, test_loader):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            preds = (outputs.squeeze(-1) > 0.5).int()  # Threshold at 0.5
            all_preds.extend(preds.tolist())
            all_targets.extend(targets.tolist())

    # Flatten lists
    flat_preds = [item for sublist in all_preds for item in sublist]
    flat_targets = [item for sublist in all_targets for item in sublist]

    accuracy = accuracy_score(flat_targets, flat_preds)
    print(f"Accuracy: {accuracy:.4f}")

evaluate_model(model, test_loader)

# Inference
def predict_labels(phrase, model, char_to_idx):
    model.eval()
    input_ids = torch.tensor(
        [char_to_idx[char] for char in phrase], dtype=torch.long
    ).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_ids)
        preds = (outputs.squeeze(-1) > 0.5).int().squeeze(0).tolist()
    return preds

# Example prediction
new_phrase = "thespike"
predicted_labels = predict_labels(new_phrase, model, char_to_idx)
print(f"Phrase: {new_phrase}")
print(f"Predicted Labels: {predicted_labels}")
