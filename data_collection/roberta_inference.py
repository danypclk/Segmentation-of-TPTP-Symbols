import torch
import re
from transformers import RobertaTokenizer, RobertaForTokenClassification

# Function to load the trained model and tokenizer
def load_model_and_tokenizer(model_path="roberta-segmentation"):
    # Load the tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    # Load the trained model
    model = RobertaForTokenClassification.from_pretrained(model_path)
    model.eval()  # Set model to evaluation mode
    return model, tokenizer

# Function to predict labels for a given phrase
def predict_labels(phrase, model, tokenizer):
    """
    Predict segmentation labels for the input phrase and print separated words.
    
    Args:
    - phrase (str): Input phrase to segment.
    - model: Trained RoBERTa model.
    - tokenizer: Tokenizer for RoBERTa.

    Returns:
    - List[int]: Predicted labels for each character in the input phrase.
    """
    # Tokenize input phrase at character level
    encoding = tokenizer(
        list(phrase), 
        is_split_into_words=True, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=512
    )
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    
    # Make predictions
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1).squeeze(0).tolist()
    
    # Remove padding predictions if present
    sequence_length = attention_mask.sum().item()
    preds = preds[:sequence_length]

    # Ensure last two elements are always discarded
    if len(preds) > 2:
        preds = preds[:-2]
    else:
        preds = []  # If fewer than 3 elements, return an empty list
    
    # Segment the phrase based on predicted labels
    def segment_text(text, labels):
        words = []
        current_word = []
        
        for char, label in zip(text, labels):
            if label == 1:  # Start of a new word
                if current_word:
                    words.append("".join(current_word))
                current_word = [char]
            else:  # Continuation of the current word
                current_word.append(char)
        
        if current_word:
            words.append("".join(current_word))
        
        # Remove special characters including underscores explicitly
        words = [re.sub(r'[^a-zA-Z]', '', word) for word in words if len(re.sub(r'[^a-zA-Z]', '', word)) > 1]
        return [word for word in words if word]  # Remove empty strings
    
    segmented_words = segment_text(phrase, preds)

    print(segmented_words);
    
    return preds

# Main inference function
def main():
    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Input phrase for inference
    while True:
        phrase = input("\nEnter a phrase for segmentation (or 'exit' to quit): ").strip()
        if phrase.lower() == 'exit':
            break
        # Get predicted labels
        predicted_labels = predict_labels(phrase, model, tokenizer)
        
        # Display results
        print(f"Phrase: {phrase}")
        print(f"Predicted Labels: {predicted_labels}")

if __name__ == "__main__":
    main()
