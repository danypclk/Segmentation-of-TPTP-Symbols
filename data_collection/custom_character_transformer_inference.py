import torch
from torch import nn

# Function to load the trained model and character-to-index mapping
def load_model_and_mapping(model_path="char_segmentation_model.pth", char_to_idx_path="char_to_idx.json"):
    """
    Load the trained model and character-to-index mapping.

    Args:
    - model_path (str): Path to the trained model's state_dict.
    - char_to_idx_path (str): Path to the character-to-index JSON file.

    Returns:
    - model: Loaded model.
    - char_to_idx (dict): Character-to-index mapping.
    """
    # Load the character-to-index mapping
    with open(char_to_idx_path, "r") as file:
        char_to_idx = json.load(file)

    # Initialize the model
    vocab_size = len(char_to_idx)
    embed_dim = 128
    hidden_dim = 256
    model = CharSegmentationModel(vocab_size, embed_dim, hidden_dim)

    # Load the trained model weights
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set model to evaluation mode

    return model, char_to_idx

# Function to predict labels for a given phrase
def predict_labels(phrase, model, char_to_idx):
    """
    Predict segmentation labels for a given phrase.

    Args:
    - phrase (str): Input phrase to segment.
    - model: Trained segmentation model.
    - char_to_idx (dict): Character-to-index mapping.

    Returns:
    - List[int]: Predicted binary labels for each character in the input phrase.
    """
    # Convert characters to indices
    input_ids = torch.tensor(
        [char_to_idx.get(char, char_to_idx["<PAD>"]) for char in phrase], dtype=torch.long
    ).unsqueeze(0)  # Add batch dimension

    # Get predictions
    with torch.no_grad():
        outputs = model(input_ids)
        preds = (outputs.squeeze(-1) > 0.5).int().squeeze(0).tolist()

    return preds

# Main function for inference
def main():
    # Load the model and character-to-index mapping
    model, char_to_idx = load_model_and_mapping()

    # Run predictions interactively
    while True:
        phrase = input("\nEnter a phrase for segmentation (or 'exit' to quit): ").strip()
        if phrase.lower() == "exit":
            break

        # Predict labels
        predicted_labels = predict_labels(phrase, model, char_to_idx)

        # Display results
        print(f"Phrase: {phrase}")
        print(f"Predicted Labels: {predicted_labels}")

if __name__ == "__main__":
    main()
