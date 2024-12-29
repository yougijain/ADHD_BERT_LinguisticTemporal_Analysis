import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, AdamW
# preprocess_text for smaller datasets, batch_tokenize for larger datasets (default)
from data.preprocess import preprocess_text, clean_dataset
 
from data.data_loader import ADHDTextDataset
from training.train import train_model
from training.evaluate import evaluate_model
from utils.time_utils import convert_to_datetime
from data.preprocess import batch_tokenize

# Disable symlink warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# File paths
DATASET_PATH = "datasets/ADHD.csv"

def load_and_prepare_data():
    """
    Load, clean, and preprocess the dataset, then prepare DataLoader objects for training and validation.
    Returns:
        DataLoader: Train DataLoader
        DataLoader: Validation DataLoader
    """
    # Load the dataset
    print("Loading dataset...")
    data = pd.read_csv(DATASET_PATH)

    # Clean the dataset
    print("Cleaning dataset...")
    data = clean_dataset(data)

    # Convert created_utc to datetime
    print("Converting timestamps...")
    data = convert_to_datetime(data, "created_utc")

    # Tokenize the text data
    print("Tokenizing text data...")
    encodings = batch_tokenize(data["selftext"].tolist(), batch_size=512)
    labels = data["score"].apply(lambda x: 1 if x > 0 else 0).tolist()  # Example: Binary labels

    # Split into training and validation sets
    train_size = int(0.8 * len(data))
    val_size = len(data) - train_size
    train_encodings = {key: val[:train_size] for key, val in encodings.items()}
    val_encodings = {key: val[train_size:] for key, val in encodings.items()}
    train_labels = labels[:train_size]
    val_labels = labels[train_size:]

    # Create ADHDTextDataset and DataLoaders
    print("Creating DataLoaders...")
    train_dataset = ADHDTextDataset(train_encodings, train_labels)
    val_dataset = ADHDTextDataset(val_encodings, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    return train_loader, val_loader

def main():
    """
    Main function to train and evaluate the ADHD classification model.
    """
    # Load and prepare data
    train_loader, val_loader = load_and_prepare_data()

    # Initialize model and optimizer
    print("Initializing model...")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # Train the model
    print("Starting training...")
    train_model(train_loader, model, optimizer, epochs=5, save_path="bert_adhd_model.pth")

    # Evaluate the model
    print("Evaluating model...")
    evaluate_model(val_loader, model)

if __name__ == "__main__":
    main()
