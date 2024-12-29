import torch
from torch.utils.data import DataLoader

def train_model(train_loader, model, optimizer, epochs=5, save_path=None):
    """
    Trains the BERT model on the given dataset and optionally saves it after training.

    Args:
        train_loader (DataLoader): DataLoader for the training data.
        model (nn.Module): BERT model for classification.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        epochs (int): Number of training epochs.
        save_path (str, optional): Path to save the trained model. Defaults to None.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    for epoch in range(epochs):
        print(f"Starting Epoch {epoch + 1}/{epochs}...")
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # Log progress for every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1} | Batch {batch_idx + 1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        print(f"Epoch {epoch + 1} Completed | Average Loss: {total_loss / len(train_loader):.4f}")

    # Save the model if a save path is provided
    if save_path:
        print(f"Saving the model to {save_path}...")
        torch.save(model.state_dict(), save_path)
        print("Model saved successfully!")
