"""Checkpointing, device selection, and reproducibility helpers."""

import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed=42):
    """Seed python, numpy, and torch so a run is reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_device(preference="auto"):
    """Turn a device preference into a torch.device.

    "auto" picks CUDA when it is genuinely available, otherwise CPU. Asking for
    "cuda" on a machine without it falls back with a warning instead of dying
    partway through data loading.
    """
    preference = (preference or "auto").lower()
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("WARNING: device='cuda' requested but no CUDA device is visible. Using CPU.")
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model, trainable_only=True):
    """Number of parameters in a model."""
    params = model.parameters()
    if trainable_only:
        params = (p for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def describe_model(model):
    """One-line summary of a model's size, handy in run logs."""
    trainable = count_parameters(model, trainable_only=True)
    total = count_parameters(model, trainable_only=False)
    return (
        f"{model.__class__.__name__}: {total:,} parameters "
        f"({trainable:,} trainable, {total - trainable:,} frozen)"
    )


def save_checkpoint(model, path, optimizer=None, epoch=None, metrics=None, config=None):
    """Save weights plus the context needed to interpret them later.

    The old code called torch.save(model.state_dict(), path) with nothing else,
    so a checkpoint on disk carried no record of which epoch or configuration
    produced it.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {"model_state_dict": model.state_dict()}
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        payload["epoch"] = epoch
    if metrics is not None:
        payload["metrics"] = metrics
    if config is not None:
        payload["config"] = config.to_dict() if hasattr(config, "to_dict") else dict(config)

    torch.save(payload, path)
    return path


def load_checkpoint(model, path, optimizer=None, device=None, strict=True):
    """Restore a checkpoint written by save_checkpoint.

    Also accepts a bare state_dict, which is what the original training loop
    produced, so old checkpoints still load.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint at {path}")

    device = device or resolve_device("auto")
    payload = torch.load(path, map_location=device, weights_only=False)

    # A checkpoint from save_checkpoint is a wrapper dict; a bare state_dict is
    # also a dict, so distinguish them by the wrapper key rather than by type.
    is_wrapped = isinstance(payload, dict) and "model_state_dict" in payload

    state_dict = payload["model_state_dict"] if is_wrapped else payload
    model.load_state_dict(state_dict, strict=strict)
    model.to(device)

    if optimizer is not None and is_wrapped and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    meta = {}
    if is_wrapped:
        meta = {k: v for k, v in payload.items()
                if k not in ("model_state_dict", "optimizer_state_dict")}
    return model, meta


def save_metrics(metrics, path):
    """Write a metrics dict to JSON, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, default=str)
    return path
