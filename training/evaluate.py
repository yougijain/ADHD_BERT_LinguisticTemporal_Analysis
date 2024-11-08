# Evaluation metrics and testing methods

import torch
from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(val_loader, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            
            predictions.extend(preds.cpu().tolist())
            true_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(true_labels, predictions)
    print("Validation Accuracy:", acc)
    print("Classification Report:\n", classification_report(true_labels, predictions))
    print("Predictions:", predictions)
    print("True Labels:", true_labels)
