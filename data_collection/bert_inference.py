import torch
from transformers import BertForTokenClassification

# Define the CharLevelTokenizer class (as provided in your code)
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

def load_model_and_tokenizer(
    local_model_dir="bert-segmentation", 
    remote_model_name="danypereira264/bert-segmentation_2"
):
    """
    Load a fine-tuned BERT segmentation model. This function first checks if there is
    a local folder named 'bert-segmentation' (or the name you specify).
    If found, it loads the model from that local path.
    Otherwise, it downloads/loads from the Hugging Face Hub: 'danypereira264/bert-segmentation_2'.

    Returns:
        model (BertForTokenClassification): The loaded model in evaluation mode
        tokenizer (CharLevelTokenizer): The corresponding tokenizer
    """
    # Initialize the custom CharLevelTokenizer
    tokenizer = CharLevelTokenizer()
    
    # Check if the local model directory exists
    if os.path.isdir(local_model_dir):
        # Load model from local directory
        print(f"Loading model from local directory: {local_model_dir}")
        model = BertForTokenClassification.from_pretrained(local_model_dir)
    else:
        # Otherwise, load model from Hugging Face Hub
        print(f"Local directory '{local_model_dir}' not found. "
              f"Loading model from '{remote_model_name}'.")
        model = BertForTokenClassification.from_pretrained(remote_model_name)
    
    model.eval()
    return model, tokenizer

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

if __name__ == "__main__":
    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # Input text for testing
    test_phrase = input("Enter a concatenated string for segmentation: ").strip()

    # Perform inference
    predicted_labels = infer_segmentation(model, tokenizer, test_phrase)

    # Display the result
    print(f"Input: {test_phrase}")
    print(f"Predicted Labels: {predicted_labels}")
