import torch
import re
import wordninja
import os
from transformers import RobertaTokenizer, RobertaForTokenClassification

# Function to load the trained model and tokenizer
def load_model_and_tokenizer(model_path="danypereira264/roberta-segmentation_2"):
    local_cache_path = os.path.expanduser("~/.cache/huggingface/hub/models--danypereira264--roberta-segmentation_2")
    
    if os.path.exists(local_cache_path):
        model_path = local_cache_path
    
    # Load the tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    # Load the trained model
    model = RobertaForTokenClassification.from_pretrained(model_path)
    model.eval()  # Set model to evaluation mode
    return model, tokenizer

# Function to predict labels for a given phrase
def predict_labels(phrase, model, tokenizer):
    """
    Predict segmentation labels for the input phrase and segment it using wordninja.
    
    Args:
    - phrase (str): Input phrase to segment.
    - model: Trained RoBERTa model.
    - tokenizer: Tokenizer for RoBERTa.

    Returns:
    - List[str]: Segmented words.
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
    
    # Segment text using wordninja
    segmented_words = wordninja.split(phrase)
    
    # Remove special characters and short words
    segmented_words = [re.sub(r'[^a-zA-Z]', '', word) for word in segmented_words if len(re.sub(r'[^a-zA-Z]', '', word)) > 2]
    segmented_words = [word for word in segmented_words if word]
    
    print(segmented_words)
    
    return segmented_words

# Main inference function
def main():
    # Load the model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    
    # Input phrase for inference
    while True:
        phrase = input("\nEnter a phrase for segmentation (or 'exit' to quit): ").strip()
        if phrase.lower() == 'exit':
            break
        
        # Get segmented words
        segmented_words = predict_labels(phrase, model, tokenizer)
        
        # Display results
        print(f"Phrase: {phrase}")
        print(f"Segmented Words: {segmented_words}")

if __name__ == "__main__":
    main()
