import os
import json
import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaForTokenClassification
from torch.optim import AdamW
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
directory = "training_set"
phrases, labels = load_json_files(directory)

# Split into training and testing sets
train_val_phrases, test_phrases, train_val_labels, test_labels = train_test_split(
    phrases, labels, test_size=0.2, random_state=42
)

train_phrases, val_phrases, train_labels, val_labels = train_test_split(
    train_val_phrases, train_val_labels, test_size=0.25, random_state=42
)

# Load the tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Dataset class for RoBERTa
class RobertaSegmentationDataset(Dataset):
    def __init__(self, phrases, labels, tokenizer):
        self.phrases = phrases
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.phrases)

    def __getitem__(self, idx):
        phrase = self.phrases[idx]
        label = self.labels[idx]

        # Tokenize the phrase
        encoding = self.tokenizer(
            list(phrase),  # Tokenize at the character level
            is_split_into_words=True,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )

        # Pad labels to the same length
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        label = torch.tensor(label + [0] * (len(input_ids) - len(label)), dtype=torch.long)

        return input_ids, attention_mask, label

# Create datasets
train_dataset = RobertaSegmentationDataset(train_phrases, train_labels, tokenizer)
val_dataset = RobertaSegmentationDataset(val_phrases, val_labels, tokenizer)
test_dataset = RobertaSegmentationDataset(test_phrases, test_labels, tokenizer)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Load the pretrained RoBERTa model
model = RobertaForTokenClassification.from_pretrained("roberta-base", num_labels=2)

# Optimizer and loss function
optimizer = AdamW(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()

def save_checkpoint(model, optimizer, epoch, batch_idx, loss, checkpoint_path="my_checkpoints/model_checkpoint.pth"):
    # Ensure the directory exists
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)  # Create the directory if it doesn't exist
    
    # Save the checkpoint
    checkpoint = {
        "epoch": epoch,
        "batch_idx": batch_idx,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "loss": loss
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

def load_checkpoint(model, optimizer, checkpoint_path="checkpoint.pth"):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        print(f"Checkpoint loaded from epoch {checkpoint['epoch']}, batch {checkpoint['batch_idx']}")
        return checkpoint["epoch"], checkpoint["batch_idx"], checkpoint["loss"]
    return 0, 0, float("inf")  # Default start from scratch

def validate_model(model, val_loader):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for input_ids, attention_mask, labels in val_loader:
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    # Flatten lists for evaluation
    flat_preds = [item for sublist in all_preds for item in sublist]
    flat_targets = [item for sublist in all_targets for item in sublist]
    accuracy = accuracy_score(flat_targets, flat_preds)
    print(f"Validation Accuracy: {accuracy:.4f}")
    model.train()  # Switch back to training mode

# Training loop with time tracking
def train_model(model, train_loader, criterion, optimizer, epochs=3, checkpoint_path="checkpoint.pth"):
    start_epoch, start_batch, _ = load_checkpoint(model, optimizer, checkpoint_path)

    model.train()
    try:
        for epoch in range(start_epoch, epochs):
            total_loss = 0
            for batch_idx, (input_ids, attention_mask, labels) in enumerate(train_loader):
                if epoch == start_epoch and batch_idx < start_batch:
                    continue  # Skip processed batches

                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                print(f"Epoch {epoch + 1}/{epochs} - Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")

                # Save checkpoint every 500 batches
                if batch_idx % 500 == 0:
                    save_checkpoint(model, optimizer, epoch, batch_idx, total_loss, checkpoint_path)

            print(f"Epoch {epoch + 1} completed. Total Loss: {total_loss:.4f}")

            # Validate after each epoch using the validation set
            validate_model(model, val_loader)

            # Save checkpoint at the end of each epoch
            save_checkpoint(model, optimizer, epoch, batch_idx, total_loss, checkpoint_path)

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint before exit...")
        save_checkpoint(model, optimizer, epoch, batch_idx, total_loss, checkpoint_path)
        print("Checkpoint saved. You can resume training later.")

    print("Training complete.")

# Train the model
train_model(model, train_loader, criterion, optimizer, epochs=3, checkpoint_path="my_checkpoints/model_checkpoint.pth")

# Save the trained RoBERTa model
model.save_pretrained("roberta-segmentation")
tokenizer.save_pretrained("roberta-segmentation")
print("Model and tokenizer have been saved to the 'roberta-segmentation' directory.")

# Evaluation
def evaluate_model(model, test_loader):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for input_ids, attention_mask, labels in test_loader:
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    # Flatten lists for evaluation
    flat_preds = [item for sublist in all_preds for item in sublist]
    flat_targets = [item for sublist in all_targets for item in sublist]

    accuracy = accuracy_score(flat_targets, flat_preds)
    print(f"Accuracy: {accuracy:.4f}")

# Evaluate the model
evaluate_model(model, test_loader)

# Inference
def predict_labels(phrase, model, tokenizer):
    model.eval()
    encoding = tokenizer(
        list(phrase), is_split_into_words=True, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1).squeeze(0).tolist()
    return preds

# Example prediction
new_phrase = "thespike"
predicted_labels = predict_labels(new_phrase, model, tokenizer)
print(f"Phrase: {new_phrase}")
print(f"Predicted Labels: {predicted_labels}")