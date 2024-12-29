import pandas as pd
from transformers import BertTokenizer

# File paths
DATASET_PATH = "datasets/ADHD.csv"

# Load the dataset
print("Loading dataset...")
data = pd.read_csv(DATASET_PATH)

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize and calculate sequence lengths
print("Calculating token lengths...")
text_lengths = [
    len(tokenizer.encode(text, truncation=True, max_length=512, add_special_tokens=True))
    for text in data["selftext"].dropna()
]

# Analyze the token lengths
average_length = sum(text_lengths) / len(text_lengths)
max_length = max(text_lengths)
min_length = min(text_lengths)

print(f"Average token length: {average_length:.2f}")
print(f"Maximum token length: {max_length}")
print(f"Minimum token length: {min_length}")

# Optional: Save the lengths for further analysis
data["token_length"] = text_lengths
data.to_csv("datasets/ADHD_with_token_lengths.csv", index=False)
print("Token lengths saved to datasets/ADHD_with_token_lengths.csv.")
