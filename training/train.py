"""Training loop with device-aware mixed precision, warmup, and validation."""

import math
import time

import torch
from torch.optim import AdamW

from models.model_utils import save_checkpoint
from training.evaluate import evaluate_model


def build_optimizer(model, learning_rate=2e-5, weight_decay=0.01):
    """AdamW with weight decay switched off for biases and LayerNorm.

    The old code imported AdamW from `transformers`, which was deprecated in
    v4.x and removed in v5 -- that import alone is enough to kill the script on
    a current install. torch.optim.AdamW is the drop-in replacement.

    Decaying bias and LayerNorm parameters is the standard BERT fine-tuning
    mistake; those are excluded here.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias") or "LayerNorm" in name:
            no_decay.append(param)
        else:
            decay.append(param)

    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return AdamW(groups, lr=learning_rate)


def build_scheduler(optimizer, num_training_steps, warmup_ratio=0.1):
    """Linear warmup then linear decay -- the standard BERT fine-tuning schedule.

    Training previously ran at a flat 2e-5 from step zero, which is the usual
    cause of an unstable first epoch.
    """
    num_training_steps = max(int(num_training_steps), 1)
    num_warmup_steps = max(int(num_training_steps * warmup_ratio), 1)

    def lr_lambda(step):
        if step < num_warmup_steps:
            return step / max(1, num_warmup_steps)
        progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return max(0.0, 1.0 - progress)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _amp_enabled(device, requested):
    """Mixed precision only makes sense on CUDA.

    torch.cuda.amp.GradScaler on a CPU box prints "enabled, but CUDA is not
    available. Disabling." and then silently no-ops, while
    torch.cuda.amp.autocast leaves the math in fp32 -- so the old code looked
    like it was doing mixed precision and was not. Be explicit about it.
    """
    return bool(requested) and device.type == "cuda"


def train_model(
    train_loader,
    model,
    optimizer=None,
    epochs=3,
    save_path=None,
    val_loader=None,
    device=None,
    scheduler=None,
    use_amp=True,
    max_grad_norm=1.0,
    log_every=10,
    config=None,
    save_every_epoch=False,
):
    """Fine-tune the model and return the run history.

    Args:
        train_loader (DataLoader): Training batches.
        model (nn.Module): Model exposing .loss/.logits from forward().
        optimizer (Optimizer, optional): Built from the model if omitted.
        epochs (int): Number of passes over the training set.
        save_path (str | Path, optional): Where to write the best checkpoint.
        val_loader (DataLoader, optional): Evaluated after every epoch; the best
            validation accuracy decides which checkpoint is kept.
        device (torch.device, optional): Defaults to CUDA when available.
        scheduler (LRScheduler, optional): Built as linear warmup+decay if None.
        use_amp (bool): Request mixed precision. Ignored on CPU.
        max_grad_norm (float): Gradient clipping threshold; 0 disables it.
        log_every (int): Batch logging interval.
        config (Config, optional): Stored inside the checkpoint.
        save_every_epoch (bool): Also write a per-epoch checkpoint.
    Returns:
        dict: batch_losses, epoch_losses, val_metrics, best_epoch,
        best_val_accuracy, seconds.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if optimizer is None:
        optimizer = build_optimizer(model)

    steps_per_epoch = max(len(train_loader), 1)
    if scheduler is None:
        scheduler = build_scheduler(optimizer, steps_per_epoch * max(epochs, 1))

    amp_on = _amp_enabled(device, use_amp)
    # torch.amp.* is the current API; torch.cuda.amp.* is deprecated as of 2.4.
    scaler = torch.amp.GradScaler(device.type, enabled=amp_on)
    print(
        f"Training on {device} | mixed precision: {'on' if amp_on else 'off'} | "
        f"{steps_per_epoch} batches/epoch x {epochs} epochs"
    )

    history = {
        "batch_losses": [],
        "epoch_losses": [],
        "val_metrics": [],
        "best_epoch": None,
        "best_val_accuracy": None,
        "seconds": 0.0,
    }
    best_val_acc = -math.inf
    started = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        print(f"\nEpoch {epoch}/{epochs}")
        total_loss = 0.0
        seen_batches = 0

        for batch_idx, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad(set_to_none=True)

            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
                "labels": batch["labels"].to(device),
            }
            if "temporal_features" in batch:
                inputs["temporal_features"] = batch["temporal_features"].to(device)

            with torch.autocast(device_type=device.type, enabled=amp_on):
                outputs = model(**inputs)
                loss = outputs.loss

            scaler.scale(loss).backward()

            if max_grad_norm and max_grad_norm > 0:
                # Gradients must be unscaled before clipping, or the threshold is
                # applied to scaled values and effectively does nothing.
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            loss_value = loss.detach().item()
            total_loss += loss_value
            seen_batches += 1
            history["batch_losses"].append(loss_value)

            if log_every and batch_idx % log_every == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  batch {batch_idx}/{steps_per_epoch} | "
                    f"loss {loss_value:.4f} | lr {current_lr:.2e}"
                )

        avg_loss = total_loss / max(seen_batches, 1)
        history["epoch_losses"].append(avg_loss)
        print(f"  epoch {epoch} average loss: {avg_loss:.4f}")

        if val_loader is not None:
            metrics = evaluate_model(val_loader, model, device=device, verbose=False)
            metrics["epoch"] = epoch
            history["val_metrics"].append(metrics)
            print(
                f"  val accuracy {metrics['accuracy']:.4f} | "
                f"macro F1 {metrics['macro_f1']:.4f} | val loss {metrics['loss']:.4f}"
            )

            if save_path and metrics["accuracy"] > best_val_acc:
                best_val_acc = metrics["accuracy"]
                history["best_epoch"] = epoch
                history["best_val_accuracy"] = best_val_acc
                save_checkpoint(model, save_path, optimizer=optimizer, epoch=epoch,
                                metrics=metrics, config=config)
                print(f"  new best -- checkpoint written to {save_path}")

        if save_path and save_every_epoch:
            stem = str(save_path).removesuffix(".pth")
            save_checkpoint(model, f"{stem}_epoch_{epoch}.pth", optimizer=optimizer,
                            epoch=epoch, config=config)

    # With no validation set there is nothing to select on, so keep the final
    # weights rather than leaving save_path empty.
    if save_path and val_loader is None:
        save_checkpoint(model, save_path, optimizer=optimizer, epoch=epochs, config=config)
        print(f"\nFinal model written to {save_path}")

    history["seconds"] = round(time.time() - started, 2)
    print(f"\nTraining finished in {history['seconds']}s")
    return history
