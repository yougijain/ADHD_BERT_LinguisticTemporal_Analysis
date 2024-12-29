import torch
from torch.utils.data import Dataset

class ADHDTextDataset(Dataset):
    """
    Custom PyTorch Dataset for ADHD text data.
    Handles BERT tokenized inputs and associated labels.

    Attributes:
        encodings (dict): Dictionary containing tokenized inputs (input_ids, attention_mask, etc.).
        labels (list or None): List of labels corresponding to the data samples. Can be None for inference.
    """
    def __init__(self, encodings, labels=None):
        """
        Initialize the dataset with encodings and labels.
        Args:
            encodings (dict): Tokenized inputs for BERT.
            labels (list, optional): Labels for the data. Defaults to None (e.g., for inference).
        """
        self.encodings = encodings
        self.labels = labels

        # Ensure encodings and labels match in length if labels are provided
        if self.labels is not None and len(self.encodings['input_ids']) != len(self.labels):
            raise ValueError("Encodings and labels must have the same length.")

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        """
        Retrieve a single sample by index.
        Args:
            idx (int): Index of the sample to retrieve.
        Returns:
            dict: A dictionary containing the tokenized inputs and the label (if provided).
        """
        # Clone encodings to avoid modifying the original tensor
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}

        # Add label if available
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item
