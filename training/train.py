import torch
from torch.cuda.amp import GradScaler, autocast


def train_model(train_loader, model, optimizer, epochs=5, save_path=None):
    """
    Trains the BERT model using mixed precision and optionally saves it after training.

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

    # Initialize GradScaler for mixed precision
    scaler = GradScaler()

    # Track batch losses for visualization
    all_batch_losses = []

    for epoch in range(epochs):
        print(f"Starting Epoch {epoch + 1}/{epochs}...")
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            # Move inputs to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Use autocast for mixed precision
            with autocast():
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            # Scale the loss and backpropagate
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            all_batch_losses.append(loss.item())

            # Log progress for every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1} | Batch {batch_idx + 1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        # Log average loss for the epoch
        avg_epoch_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1} Completed | Average Loss: {avg_epoch_loss:.4f}")

        # Save intermediate checkpoint after each epoch
        if save_path:
            epoch_save_path = f"{save_path}_epoch_{epoch + 1}.pth"
            print(f"Saving checkpoint for Epoch {epoch + 1} to {epoch_save_path}...")
            torch.save(model.state_dict(), epoch_save_path)
            print("Checkpoint saved successfully!")

    # Final save for the full model if a save path is provided
    if save_path:
        print(f"Saving the final model to {save_path}...")
        torch.save(model.state_dict(), save_path)
        print("Model saved successfully!")

    # Return batch loss values for visualization
    return all_batch_losses
