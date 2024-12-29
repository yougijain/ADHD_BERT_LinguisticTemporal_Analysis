import torch
from torch.utils.data import DataLoader
from data.data_loader import ADHDTextDataset

def test_dataset_with_labels():
    # Example tokenized inputs and labels
    encodings = {
        "input_ids": torch.tensor([[101, 2009, 2001, 1037, 3867, 102], [101, 1045, 2293, 2023, 102]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
    }
    labels = [1, 0]

    # Create dataset and dataloader
    dataset = ADHDTextDataset(encodings, labels)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # Print batches
    for batch in dataloader:
        print("Batch with labels:", batch)

def test_dataset_without_labels():
    # Example tokenized inputs (no labels)
    encodings = {
        "input_ids": torch.tensor([[101, 2009, 2001, 1037, 3867, 102], [101, 1045, 2293, 2023, 102]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
    }

    # Create dataset and dataloader
    dataset = ADHDTextDataset(encodings)
    dataloader = DataLoader(dataset, batch_size=2)

    # Print batches
    for batch in dataloader:
        print("Batch without labels:", batch)

if __name__ == "__main__":
    test_dataset_with_labels()
    test_dataset_without_labels()
