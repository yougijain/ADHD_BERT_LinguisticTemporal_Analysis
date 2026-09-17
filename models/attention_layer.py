"""Additive attention pooling over BERT's token representations.

BERT's default sentence vector is the [CLS] token. That works, but it throws
away which words the model actually leaned on. This layer learns a scalar
relevance score per token, softmaxes it into weights, and returns both the
weighted sum and the weights -- so a prediction can be traced back to the words
that drove it, which is the whole point of a linguistic analysis.
"""

import torch
import torch.nn as nn


class AttentionPooling(nn.Module):
    """Pool a (batch, seq_len, hidden) sequence into (batch, hidden).

    Args:
        hidden_size (int): Width of the incoming token representations.
        attention_size (int, optional): Width of the internal scoring layer.
            Defaults to half of hidden_size (minimum 1).
    """

    def __init__(self, hidden_size, attention_size=None):
        super().__init__()
        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        attention_size = attention_size or max(hidden_size // 2, 1)

        self.projection = nn.Sequential(
            nn.Linear(hidden_size, attention_size),
            nn.Tanh(),
            nn.Linear(attention_size, 1),
        )

    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states (Tensor): (batch, seq_len, hidden).
            attention_mask (Tensor, optional): (batch, seq_len), 1 for real
                tokens and 0 for padding.
        Returns:
            tuple[Tensor, Tensor]: pooled (batch, hidden) and the attention
            weights (batch, seq_len).
        """
        scores = self.projection(hidden_states).squeeze(-1)  # (batch, seq_len)

        if attention_mask is not None:
            mask = attention_mask.to(dtype=torch.bool)
            # finfo().min rather than -inf: it survives fp16 autocast without
            # turning the whole row into NaN.
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

        weights = torch.softmax(scores, dim=-1)

        if attention_mask is not None:
            # A row with no real tokens softmaxes to a uniform distribution over
            # padding. Zero it out so it contributes nothing rather than
            # silently pooling pad embeddings.
            weights = weights * mask.to(weights.dtype)
            total = weights.sum(dim=-1, keepdim=True)
            weights = weights / total.clamp(min=torch.finfo(weights.dtype).eps)

        pooled = torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)
        return pooled, weights


def masked_mean_pool(hidden_states, attention_mask=None):
    """Mean of the non-padding token vectors. Fallback when attention pooling
    is disabled, and a sane baseline to compare it against."""
    if attention_mask is None:
        return hidden_states.mean(dim=1)

    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    summed = (hidden_states * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts
