import torch
from transformers import BertForTokenClassification

# Define the CharLevelTokenizer class
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

# Define the segmentation function

def segment_text(text, labels):
    """
    Segment text into words based on the labels.

    :param text: Input string
    :param labels: List of labels corresponding to each character in the text
    :return: List of segmented words
    """
    words = []
    current_word = []

    for char, label in zip(text, labels):
        if label == 1:  # Start of a new word
            if current_word:
                words.append("".join(current_word))
            current_word = [char]
        else:  # Continuation of the current word
            current_word.append(char)

    # Add the last word if any
    if current_word:
        words.append("".join(current_word))

    return words

# Define the inference function
def infer_segmentation(model, tokenizer, input_text, max_length=128):
    """
    Perform inference on a given input string to predict segmentation labels.
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Tokenize the input text
    input_ids = tokenizer.encode(input_text)
    input_ids = input_ids[:max_length]  # Truncate if longer than max_length
    attention_mask = [1] * len(input_ids)

    # Pad input_ids and attention_mask
    input_ids = tokenizer.pad([input_ids], max_length=max_length)[0]
    attention_mask = tokenizer.pad([attention_mask], max_length=max_length)[0]

    # Convert to PyTorch tensors
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)  # Batch size = 1
    attention_mask = torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask).logits
        preds = torch.argmax(outputs, dim=-1).squeeze().cpu().numpy()

    # Remove padding and return the labels corresponding to the input text
    labels = preds[:len(input_text)]
    return labels.tolist()

# Main program for inference
if __name__ == "__main__":
    # Load the model
    model = BertForTokenClassification.from_pretrained("bert-segmentation")
    tokenizer = CharLevelTokenizer()  # Ensure this matches the tokenizer used during training

    # Input text for testing
    test_phrase = input("Enter a concatenated string for segmentation: ").strip()

    # Perform inference
    predicted_labels = infer_segmentation(model, tokenizer, test_phrase)

    # Segment text based on labels
    segmented_words = segment_text(test_phrase, predicted_labels)

    # Display the result
    print(f"Input: {test_phrase}")
    print(f"Predicted Labels: {predicted_labels}")
    print(f"Segmented Words: {segmented_words}")