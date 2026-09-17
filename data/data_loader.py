"""PyTorch Dataset and split helpers for ADHD post data."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class ADHDTextDataset(Dataset):
    """Tokenized post bodies, their labels, and the matching temporal features.

    Attributes:
        encodings (dict): input_ids / attention_mask tensors from the tokenizer.
        labels (array-like | None): Targets. None for inference.
        temporal_features (array-like | None): (n_samples, n_features) matrix
            aligned row-for-row with the encodings.
    """

    def __init__(self, encodings, labels=None, temporal_features=None):
        self.encodings = {k: _as_tensor(v) for k, v in encodings.items()}

        if "input_ids" not in self.encodings:
            raise KeyError("encodings must contain 'input_ids'.")

        n_samples = len(self.encodings["input_ids"])

        # Catch ragged encodings early. Building the tensors above would already
        # have failed on a ragged nested list, but a caller can still hand in
        # mismatched keys.
        for key, value in self.encodings.items():
            if len(value) != n_samples:
                raise ValueError(
                    f"Encoding '{key}' has {len(value)} rows but 'input_ids' has "
                    f"{n_samples}. All encoding tensors must align."
                )

        if labels is None:
            self.labels = None
        else:
            self.labels = torch.as_tensor(np.asarray(labels), dtype=torch.long)
            if len(self.labels) != n_samples:
                raise ValueError(
                    f"Got {len(self.labels)} labels for {n_samples} samples. "
                    "Encodings and labels must have the same length."
                )

        if temporal_features is None:
            self.temporal_features = None
        else:
            features = torch.as_tensor(np.asarray(temporal_features), dtype=torch.float32)
            if features.dim() == 1:
                features = features.unsqueeze(-1)
            if len(features) != n_samples:
                raise ValueError(
                    f"Got {len(features)} temporal feature rows for {n_samples} "
                    "samples. They must align with the encodings."
                )
            self.temporal_features = features

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        item = {key: value[idx].clone().detach() for key, value in self.encodings.items()}
        if self.temporal_features is not None:
            item["temporal_features"] = self.temporal_features[idx].clone().detach()
        if self.labels is not None:
            item["labels"] = self.labels[idx].clone().detach()
        return item

    @property
    def num_temporal_features(self):
        return 0 if self.temporal_features is None else int(self.temporal_features.shape[1])


def _as_tensor(value):
    """Coerce a list/array/tensor of encodings into a tensor with a clear error.

    A ragged nested list -- rows of different lengths, which is what you get from
    tokenizing without padding -- raises a cryptic message from torch.tensor.
    Restate it in terms the caller can act on.
    """
    if isinstance(value, torch.Tensor):
        return value
    try:
        return torch.as_tensor(np.asarray(value))
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "Could not build a tensor from the encodings. Rows of different "
            "lengths are the usual cause -- tokenize with padding='max_length' "
            f"so every row is the same width. Original error: {exc}"
        ) from exc


def split_indices(n_samples, val_split=0.2, strategy="temporal", seed=42):
    """Return (train_idx, val_idx).

    strategy="temporal" keeps the chronological order: the model trains on
    earlier posts and is validated on later ones, which is the honest setup when
    the dataset is a time series and the features include the timestamp. The
    caller is responsible for having sorted by time first.

    strategy="random" shuffles, which is the right choice when you only care
    about linguistic content and want an i.i.d. split.
    """
    if n_samples <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    n_val = int(round(n_samples * val_split))
    # Never hand back an empty validation set from a non-empty dataset; metrics
    # computed on zero rows are silently meaningless.
    n_val = min(max(n_val, 1), n_samples - 1) if n_samples > 1 else 0
    n_train = n_samples - n_val

    if strategy == "temporal":
        indices = np.arange(n_samples)
    elif strategy == "random":
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n_samples)
    else:
        raise ValueError(f"Unknown split strategy: {strategy!r}")

    return indices[:n_train], indices[n_train:]


def build_dataloaders(encodings, labels, temporal_features=None, batch_size=16,
                      val_split=0.2, strategy="temporal", seed=42, num_workers=0):
    """Split the encodings and wrap both halves in DataLoaders.

    The old main.py sliced `encodings` and `labels` with plain [:train_size] and
    [train_size:] and never checked the result, so a split that put every
    positive example on one side went unnoticed. This reports the class balance
    of each half.

    Returns:
        tuple[DataLoader, DataLoader]: train and validation loaders.
    """
    n_samples = len(encodings["input_ids"])
    train_idx, val_idx = split_indices(n_samples, val_split, strategy, seed)

    def subset(idx):
        sub_encodings = {k: v[idx] for k, v in
                         {k: _as_tensor(v) for k, v in encodings.items()}.items()}
        sub_labels = np.asarray(labels)[idx] if labels is not None else None
        sub_temporal = (np.asarray(temporal_features)[idx]
                        if temporal_features is not None else None)
        return ADHDTextDataset(sub_encodings, sub_labels, sub_temporal)

    train_dataset = subset(train_idx)
    val_dataset = subset(val_idx)

    if labels is not None:
        for name, idx in (("train", train_idx), ("val", val_idx)):
            counts = np.bincount(np.asarray(labels)[idx], minlength=2)
            print(f"  {name}: {len(idx)} samples | class balance {counts.tolist()}")
            if len(idx) and counts.min() == 0:
                print(
                    f"  WARNING: the {name} split contains a single class. "
                    "Try --split-strategy random, or check the label strategy."
                )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    return train_loader, val_loader
