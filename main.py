# Main script to run the entire pipeline

import os
import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification
from torch.optim import AdamW  # Using PyTorch's AdamW to avoid deprecation warning
from data.preprocess import preprocess_text
from data.data_loader import ADHDTextDataset
from training.train import train_model
from training.evaluate import evaluate_model

# Disable symlink warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Larger synthetic dataset for testing (balanced and varied)
texts = [
    "I am feeling very restless today.", "I have been distracted a lot lately.",
    "I am focused and relaxed.", "I have been able to concentrate well.",
    "I feel anxious and can't sit still.", "I'm always on edge.",
    "Today, I feel calm and collected.", "I'm able to work without distractions.",
    "My thoughts are racing and I can't focus.", "I feel like I can't stay still.",
    "I am focused on my work and calm.", "I have clear thoughts and can concentrate.",
    "I am often fidgeting and can't sit still.", "I can't seem to finish tasks.",
    "I've been hyperactive and impatient.", "I feel like I can't focus on anything.",
    "I am at peace and focused.", "I am in control of my mind.",
    "I am constantly moving and restless.", "My mind keeps wandering.",
    "I can sit for hours without feeling anxious.", "I am relaxed and focused.",
    "I feel incredibly tense and distracted.", "I struggle with impulsive decisions.",
    "I find it hard to stay in one place.", "My mind is calm and focused on my work.",
    "I am jittery and feel like I'm always on the go.", "I tend to get sidetracked easily.",
    "I have a clear mind and can concentrate.", "I am calm and collected all day.",
    "I'm easily distracted and find it hard to concentrate.", "I am very relaxed and at ease.",
    "My thoughts are everywhere and I can't focus.", "I feel like I can't stop moving.",
    "I am attentive and focused on my tasks.", "I have peace of mind and no anxiety.",
]
labels = [
    1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
    1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
    1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0,
]

# Preprocess text data
encodings = preprocess_text(texts)

# Prepare dataset and dataloaders
dataset = ADHDTextDataset(encodings, labels)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)

# Load model and optimizer
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
optimizer = AdamW(model.parameters(), lr=2e-5)

# Train and evaluate
train_model(train_loader, model, optimizer, epochs=5)
evaluate_model(val_loader, model)
