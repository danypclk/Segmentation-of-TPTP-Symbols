import wordninja

def segment_phrase(phrase):
    # Use wordninja to split concatenated words
    segmented_words = wordninja.split(phrase)
    
    # Join the words with spaces
    segmented_phrase = " ".join(segmented_words)
    
    return segmented_phrase

# Example usage
phrase = "howabout you trythis12134oneout_an_now_youlittlehjkhhakjhthing"
segmented_result = segment_phrase(phrase)
print("Original phrase:", phrase)
print("Segmented phrase:", segmented_result)