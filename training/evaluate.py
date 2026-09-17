"""Validation metrics for the classifier."""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


@torch.no_grad()
def evaluate_model(val_loader, model, device=None, verbose=True, class_names=None):
    """Run the model over a loader and return a metrics dict.

    The old version printed accuracy plus every prediction and every true label
    to stdout and returned nothing, so callers could not act on the result and a
    real validation set buried the terminal. This returns the numbers.

    Returns:
        dict: loss, accuracy, macro_f1, weighted_f1, precision, recall,
        per-class support, confusion matrix, and the majority-class baseline.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    predictions, true_labels = [], []
    total_loss, total_batches = 0.0, 0

    for batch in val_loader:
        inputs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
        }
        if "temporal_features" in batch:
            inputs["temporal_features"] = batch["temporal_features"].to(device)

        labels = batch["labels"].to(device)
        outputs = model(**inputs, labels=labels)

        if outputs.loss is not None:
            total_loss += float(outputs.loss.detach().item())
            total_batches += 1

        preds = torch.argmax(outputs.logits, dim=-1)
        predictions.extend(preds.cpu().tolist())
        true_labels.extend(labels.cpu().tolist())

    if not true_labels:
        return {"loss": 0.0, "accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0,
                "num_samples": 0, "majority_baseline": 0.0}

    y_true = np.asarray(true_labels)
    y_pred = np.asarray(predictions)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0,
        labels=sorted(set(y_true.tolist()) | set(y_pred.tolist())),
    )

    # Accuracy alone is misleading on an imbalanced split -- always answering
    # with the majority class scores this much, so anything at or below it means
    # the model learned nothing.
    counts = np.bincount(y_true)
    majority_baseline = float(counts.max() / counts.sum())

    metrics = {
        "loss": total_loss / max(total_batches, 1),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class_precision": [float(p) for p in precision],
        "per_class_recall": [float(r) for r in recall],
        "per_class_f1": [float(v) for v in f1],
        "per_class_support": [int(s) for s in support],
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "num_samples": int(len(y_true)),
        "majority_baseline": majority_baseline,
        "lift_over_baseline": float(accuracy_score(y_true, y_pred)) - majority_baseline,
    }

    if verbose:
        target_names = class_names or [f"class_{i}" for i in sorted(set(y_true.tolist()))]
        print(f"\nValidation on {metrics['num_samples']} samples")
        print(f"  accuracy        {metrics['accuracy']:.4f}")
        print(f"  majority baseline {majority_baseline:.4f} "
              f"(lift {metrics['lift_over_baseline']:+.4f})")
        print(f"  macro F1        {metrics['macro_f1']:.4f}")
        print(f"  loss            {metrics['loss']:.4f}")
        print("\n" + classification_report(y_true, y_pred, zero_division=0,
                                           target_names=target_names[:len(set(y_true.tolist()))]))
        print("Confusion matrix (rows = true, cols = predicted):")
        print(np.array(metrics["confusion_matrix"]))

    return metrics


@torch.no_grad()
def predict(model, loader, device=None, return_attention=False):
    """Collect predictions, class probabilities, and optionally attention weights."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_preds, all_probs, all_attention = [], [], []
    for batch in loader:
        inputs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
        }
        if "temporal_features" in batch:
            inputs["temporal_features"] = batch["temporal_features"].to(device)

        outputs = model(**inputs, return_attention=return_attention)
        probs = torch.softmax(outputs.logits, dim=-1)
        all_probs.append(probs.cpu())
        all_preds.append(torch.argmax(probs, dim=-1).cpu())
        if return_attention and outputs.attention_weights is not None:
            all_attention.append(outputs.attention_weights.cpu())

    result = {
        "predictions": torch.cat(all_preds).numpy() if all_preds else np.array([]),
        "probabilities": torch.cat(all_probs).numpy() if all_probs else np.array([]),
    }
    if return_attention and all_attention:
        result["attention_weights"] = torch.cat(all_attention).numpy()
    return result
