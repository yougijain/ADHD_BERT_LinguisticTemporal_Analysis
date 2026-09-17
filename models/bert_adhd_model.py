"""BERT encoder fused with engineered timestamp features.

This is the model the project is named for. A plain
BertForSequenceClassification sees only the words; it cannot know that a post
was typed at 3am on a Tuesday. Here the text goes through BERT, the timestamp
goes through a small MLP, and the two representations are concatenated before
the classification head -- so the classifier can weigh "what was said" against
"when it was said".

Set use_temporal_features=False to get the text-only ablation and measure what
the temporal branch is actually worth.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from models.attention_layer import AttentionPooling, masked_mean_pool


@dataclass
class ClassifierOutput:
    """Mirrors the fields of a HuggingFace SequenceClassifierOutput that this
    project uses, plus the attention weights for interpretability."""

    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    attention_weights: Optional[torch.Tensor] = None
    pooled_output: Optional[torch.Tensor] = None


class BertTemporalClassifier(nn.Module):
    """BERT + temporal-feature classifier.

    Args:
        model_name (str): HuggingFace model id for the encoder.
        num_labels (int): Number of output classes.
        num_temporal_features (int): Width of the temporal feature vector. Pass
            0 (or use_temporal_features=False) for the text-only ablation.
        temporal_hidden_size (int): Width of the temporal MLP's output.
        dropout (float): Dropout applied before the classification head.
        use_attention_pooling (bool): Attention-pool the token states instead of
            taking [CLS].
        freeze_bert (bool): Freeze the encoder and train only the head. Useful
            for a fast CPU run.
        encoder (nn.Module, optional): Pre-built encoder, bypassing the
            from_pretrained download. Used by `from_config` and the tests.
    """

    def __init__(
        self,
        model_name="bert-base-uncased",
        num_labels=2,
        num_temporal_features=0,
        temporal_hidden_size=32,
        dropout=0.1,
        use_attention_pooling=True,
        freeze_bert=False,
        encoder=None,
    ):
        super().__init__()

        self.bert = encoder if encoder is not None else AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        self.num_labels = num_labels
        self.num_temporal_features = max(int(num_temporal_features), 0)
        self.use_attention_pooling = use_attention_pooling

        self.pooler = AttentionPooling(hidden_size) if use_attention_pooling else None

        if self.num_temporal_features > 0:
            self.temporal_mlp = nn.Sequential(
                nn.Linear(self.num_temporal_features, temporal_hidden_size),
                nn.LayerNorm(temporal_hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(temporal_hidden_size, temporal_hidden_size),
                nn.ReLU(),
            )
            classifier_in = hidden_size + temporal_hidden_size
        else:
            self.temporal_mlp = None
            classifier_in = hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(classifier_in, num_labels)
        self.loss_fn = nn.CrossEntropyLoss()

        if freeze_bert:
            self.freeze_encoder()

    @classmethod
    def from_config(cls, config, num_temporal_features=None, encoder=None):
        """Build from a training.config.Config.

        `encoder` lets callers hand in an already-constructed (or randomly
        initialised) encoder so that tests never touch the network.
        """
        if num_temporal_features is None:
            num_temporal_features = config.num_temporal_features
        return cls(
            model_name=config.model_name,
            num_labels=config.num_labels,
            num_temporal_features=num_temporal_features,
            temporal_hidden_size=config.temporal_hidden_size,
            dropout=config.dropout,
            use_attention_pooling=config.use_attention_pooling,
            freeze_bert=config.freeze_bert,
            encoder=encoder,
        )

    @classmethod
    def tiny_for_testing(cls, num_temporal_features=8, num_labels=2, vocab_size=2048,
                         max_position_embeddings=512, **kwargs):
        """A randomly initialised miniature BERT. No download, no pretrained
        weights -- just enough model to exercise shapes and the training loop.

        vocab_size must cover the tokenizer it will be paired with; the offline
        path builds a corpus-trained tokenizer, so pass its real vocab size.
        """
        cfg = AutoConfig.for_model(
            "bert",
            vocab_size=vocab_size,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=64,
            max_position_embeddings=max_position_embeddings,
        )
        encoder = AutoModel.from_config(cfg)
        return cls(
            num_labels=num_labels,
            num_temporal_features=num_temporal_features,
            temporal_hidden_size=kwargs.pop("temporal_hidden_size", 8),
            encoder=encoder,
            **kwargs,
        )

    def freeze_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask=None, temporal_features=None,
                labels=None, return_attention=False):
        """
        Args:
            input_ids (Tensor): (batch, seq_len) token ids.
            attention_mask (Tensor): (batch, seq_len) 1 for real tokens.
            temporal_features (Tensor, optional): (batch, num_temporal_features).
                Required when the model was built with a temporal branch.
            labels (Tensor, optional): (batch,) int64 targets. When given, the
                loss is computed and returned.
            return_attention (bool): Also return the token attention weights.
        Returns:
            ClassifierOutput
        """
        encoder_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = encoder_out.last_hidden_state

        attention_weights = None
        if self.pooler is not None:
            pooled, attention_weights = self.pooler(sequence_output, attention_mask)
        else:
            pooled = masked_mean_pool(sequence_output, attention_mask)

        if self.temporal_mlp is not None:
            if temporal_features is None:
                raise ValueError(
                    "This model was built with a temporal branch "
                    f"({self.num_temporal_features} features) but forward() got "
                    "temporal_features=None. Pass the features, or rebuild with "
                    "use_temporal_features=False for the text-only ablation."
                )
            temporal = temporal_features.to(dtype=pooled.dtype)
            if temporal.dim() == 1:
                temporal = temporal.unsqueeze(0)
            if temporal.size(-1) != self.num_temporal_features:
                raise ValueError(
                    f"Expected {self.num_temporal_features} temporal features, "
                    f"got {temporal.size(-1)}."
                )
            pooled = torch.cat([pooled, self.temporal_mlp(temporal)], dim=-1)

        logits = self.classifier(self.dropout(pooled))

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

        return ClassifierOutput(
            logits=logits,
            loss=loss,
            attention_weights=attention_weights if return_attention else None,
            pooled_output=pooled,
        )
