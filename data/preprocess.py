import re
import pandas as pd
from transformers import BertTokenizerFast
import torch

# Initialize the BERT tokenizer (fast version for better performance)
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def clean_text(text):
    """
    Cleans text by removing special characters, URLs, and unnecessary whitespace.
    Parameters:
        text (str): Raw text to be cleaned.
    Returns:
        str: Cleaned text.
    """
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    # Remove special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    # Remove extra whitespace
    text = " ".join(text.split())
    return text

def batch_tokenize(texts, batch_size=512, max_length=512):
    """
    Tokenizes text data in batches for faster processing.
    Parameters:
        texts (list): List of strings (texts) to be tokenized.
        batch_size (int): Number of texts to process per batch.
        max_length (int): Maximum length for tokenized sequences.
    Returns:
        dict: Encodings with 'input_ids' and 'attention_mask'.
    """
    all_encodings = {"input_ids": [], "attention_mask": []}
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1} of {len(texts) // batch_size + 1}")
        batch_encodings = tokenizer(
            batch_texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
        )
        all_encodings["input_ids"].append(batch_encodings["input_ids"])
        all_encodings["attention_mask"].append(batch_encodings["attention_mask"])
    
    # Concatenate batches
    all_encodings["input_ids"] = torch.cat(all_encodings["input_ids"], dim=0)
    all_encodings["attention_mask"] = torch.cat(all_encodings["attention_mask"], dim=0)
    return all_encodings

def preprocess_text(texts, max_length=512, batch_size=512):
    """
    Cleans and tokenizes a list of texts for BERT.
    Parameters:
        texts (list): List of strings (texts) to be tokenized.
        max_length (int): Maximum length for tokenized sequences.
        batch_size (int): Number of texts to process in each batch.
    Returns:
        dict: Encodings with input_ids, attention_masks, etc.
    """
    # Clean the text
    print("Cleaning text data...")
    cleaned_texts = [clean_text(text) for text in texts]

    # Tokenize text in batches
    print("Tokenizing text data...")
    encodings = batch_tokenize(cleaned_texts, batch_size=batch_size, max_length=max_length)

    return encodings

def clean_dataset(data):
    """
    Clean the dataset by handling missing values and ensuring consistency.
    Parameters:
        data (pd.DataFrame): Input dataset.
    Returns:
        pd.DataFrame: Cleaned dataset.
    """
    print("Cleaning dataset...")
    # Handle missing values in 'selftext' column
    data['selftext'] = data['selftext'].fillna("missing_text")

    # Convert 'score' to numeric, coercing errors to NaN
    data['score'] = pd.to_numeric(data['score'], errors='coerce')

    # Drop rows with missing critical values
    data = data.dropna(subset=['selftext', 'score', 'created_utc'])

    return data
