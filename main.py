# Main script to run the entire pipeline

from data.preprocess import preprocess_text

# Example texts for preprocessing
texts = ["Example sentence one.", "Another example sentence."]
encodings = preprocess_text(texts)
print(encodings)
