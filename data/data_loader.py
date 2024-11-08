# Helper functions for loading and batching data

import torch
from torch.utils.data import Dataset

class ADHDTextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Clone only the encodings to avoid the AttributeError on integers
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])  # Convert label to tensor without clone().detach()
        return item
