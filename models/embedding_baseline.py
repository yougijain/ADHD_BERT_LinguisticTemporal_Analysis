"""Frozen sentence embeddings + logistic regression: the middle of the grid.

TF-IDF against fine-tuned BERT is a comparison with a hole in it. They differ on
two axes at once -- pretrained semantics and task-specific training -- so when
BERT wins you cannot say which one earned it. This row holds the second axis
fixed: pretrained semantics, no fine-tuning, the same logistic regression head
the TF-IDF baseline uses.

Three readings, and each says something different about what to build:

  * Embeddings beat TF-IDF, fine-tuned BERT beats embeddings -> both axes pay,
    and the fine-tuning budget is justified.
  * Embeddings beat TF-IDF, fine-tuning adds ~nothing -> pretrained semantics
    are the whole story. Ship the embedding model: it is one forward pass, no
    training, no GPU at inference, and it will not drift when the data does.
  * Embeddings tie TF-IDF -> the signal is lexical. A transformer is not
    reading anything a bag of words cannot, and the honest write-up says so.

That middle outcome is the common one on short text, and it is the one worth
knowing before anyone commits to a fine-tuning pipeline.

Mean pooling, not [CLS]. BERT's [CLS] vector only means something after it has
been trained for a task; on a frozen encoder it is close to arbitrary. The
sentence-transformer checkpoints used here were explicitly trained with mean
pooling, so that is what reproduces their behaviour.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from training.evaluate import compute_metrics, print_metrics

DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


def encode_texts(texts, model_name=DEFAULT_ENCODER, batch_size=64, max_length=256,
                 device=None, normalize=True, verbose=True):
    """Embed texts with a frozen transformer encoder. No gradients, no training.

    Mean-pools the last hidden state over the attention mask -- padding tokens
    are excluded from the average rather than dragging every short post toward
    the same vector, which is what a plain `.mean(dim=1)` would do.

    Args:
        texts (list[str]): Documents to embed.
        model_name (str): HuggingFace encoder id. The default is a 22M-parameter
            MiniLM: small enough to run on CPU, strong enough to be a fair test.
        batch_size (int): Documents per forward pass.
        max_length (int): Truncation length, matching the rest of the pipeline.
        device (str | None): torch device. None resolves automatically.
        normalize (bool): L2-normalise each vector. Puts every document on the
            unit sphere so the classifier sees direction rather than magnitude,
            which for a mean-pooled encoder mostly tracks document length.
    Returns:
        np.ndarray: (len(texts), hidden_size), float32.
    """
    # Before touching transformers: embedding nothing should not download a
    # 90MB encoder, and it must not fail in an environment that cannot reach
    # the hub.
    if len(texts) == 0:
        return np.zeros((0, 0), dtype="float32")

    import torch
    from transformers import AutoModel, AutoTokenizer

    from models.model_utils import resolve_device

    device = resolve_device(device or "auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name).to(device).eval()

    vectors = []
    total = (len(texts) + batch_size - 1) // batch_size

    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = [("" if t is None else str(t)) for t in texts[start:start + batch_size]]
            if verbose:
                print(f"  Embedding batch {start // batch_size + 1}/{total}")

            encoded = tokenizer(batch, truncation=True, padding=True,
                                max_length=max_length, return_tensors="pt").to(device)
            hidden = encoder(**encoded).last_hidden_state

            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            summed = (hidden * mask).sum(dim=1)
            # clamp: an all-padding row would divide by zero and poison the batch.
            counts = mask.sum(dim=1).clamp(min=1e-9)
            pooled = summed / counts

            if normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.cpu().numpy())

    return np.vstack(vectors).astype("float32")


class EmbeddingBaseline:
    """Frozen encoder -> mean-pooled vectors (+ temporal features) -> logistic regression.

    Deliberately mirrors TfidfBaseline's interface so benchmark.py can treat the
    two identically and the metrics come out of the same compute_metrics.

    Args:
        model_name (str): Encoder to embed with.
        use_temporal_features (bool): Concatenate the temporal feature matrix.
        C (float): Inverse regularisation strength.
        class_weight (str | None): "balanced" reweights by inverse frequency.
        max_iter (int): Solver iteration cap.
        max_length (int): Truncation length for the encoder.
        batch_size (int): Documents per forward pass.
        device (str | None): torch device, or None to resolve automatically.
        seed (int): Random state.
        embed_fn (callable | None): `texts -> (n, d) array`, replacing the real
            encoder. This is how the tests exercise the class with no download,
            and how you would swap in a cached embedding matrix.
    """

    def __init__(self, model_name=DEFAULT_ENCODER, use_temporal_features=True,
                 C=1.0, class_weight=None, max_iter=1000, max_length=256,
                 batch_size=64, device=None, seed=42, embed_fn=None):
        self.model_name = model_name
        self.use_temporal_features = use_temporal_features
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device
        self.embed_fn = embed_fn

        # Embeddings and temporal features are on different scales, and the
        # regulariser would otherwise spend its whole budget on whichever block
        # happens to have the larger magnitudes.
        self.feature_scaler = StandardScaler()

        self.classifier = LogisticRegression(
            C=C, class_weight=class_weight, max_iter=max_iter,
            random_state=seed, solver="liblinear",
        )
        self._fitted = False
        self._embedding_dim = None

    def _embed(self, texts, verbose=True):
        if self.embed_fn is not None:
            vectors = np.asarray(self.embed_fn(list(texts)), dtype="float32")
            if vectors.ndim != 2 or len(vectors) != len(texts):
                raise ValueError(
                    f"embed_fn returned shape {vectors.shape} for {len(texts)} "
                    "texts; expected (n_texts, hidden_size)."
                )
            return vectors
        return encode_texts(
            list(texts), model_name=self.model_name, batch_size=self.batch_size,
            max_length=self.max_length, device=self.device, verbose=verbose,
        )

    def _build_matrix(self, texts, temporal_features=None, fit=False, verbose=True):
        vectors = self._embed(texts, verbose=verbose)

        if fit:
            self._embedding_dim = int(vectors.shape[1])
        elif self._embedding_dim is not None and vectors.shape[1] != self._embedding_dim:
            raise ValueError(
                f"Encoder returned {vectors.shape[1]}-dim vectors but the model "
                f"was fitted on {self._embedding_dim}-dim ones. The encoder "
                "changed between fit and predict."
            )

        if self.use_temporal_features:
            if temporal_features is None:
                raise ValueError(
                    "use_temporal_features=True but no temporal_features were "
                    "passed. Provide them, or construct with "
                    "use_temporal_features=False for the text-only baseline."
                )
            dense = np.asarray(temporal_features, dtype="float32")
            if dense.ndim == 1:
                dense = dense.reshape(-1, 1)
            if len(dense) != len(texts):
                raise ValueError(
                    f"Got {len(dense)} temporal feature rows for {len(texts)} "
                    "texts. They must align."
                )
            vectors = np.hstack([vectors, dense])

        return (self.feature_scaler.fit_transform(vectors) if fit
                else self.feature_scaler.transform(vectors))

    def fit(self, texts, labels, temporal_features=None, verbose=True):
        """Embed, scale, and fit the classifier. Returns self."""
        labels = np.asarray(labels)
        if len(np.unique(labels)) < 2:
            raise ValueError(
                "Training labels contain a single class; logistic regression "
                "cannot fit. Check the label strategy."
            )
        matrix = self._build_matrix(texts, temporal_features, fit=True, verbose=verbose)
        self.classifier.fit(matrix, labels)
        self._fitted = True
        return self

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Call fit() before predicting.")

    def predict(self, texts, temporal_features=None, verbose=False):
        self._check_fitted()
        return self.classifier.predict(
            self._build_matrix(texts, temporal_features, verbose=verbose)
        )

    def predict_proba(self, texts, temporal_features=None, verbose=False):
        self._check_fitted()
        return self.classifier.predict_proba(
            self._build_matrix(texts, temporal_features, verbose=verbose)
        )

    def evaluate(self, texts, labels, temporal_features=None, verbose=True,
                 class_names=None):
        """Score with the same compute_metrics every other model in the grid uses.

        Returns:
            dict: Identical shape to training.evaluate.evaluate_model's output.
        """
        self._check_fitted()

        # One forward pass, reused for both the labels and the probabilities --
        # embedding the validation set twice is the expensive part of this model.
        matrix = self._build_matrix(texts, temporal_features, verbose=verbose)
        predictions = self.classifier.predict(matrix)

        loss = 0.0
        try:
            from sklearn.metrics import log_loss
            probabilities = self.classifier.predict_proba(matrix)
            loss = float(log_loss(labels, probabilities,
                                  labels=list(range(probabilities.shape[1]))))
        except (ValueError, ImportError):
            # log_loss raises when a split is missing a class. Accuracy and F1
            # are still meaningful, so report zero loss rather than failing.
            loss = 0.0

        metrics = compute_metrics(labels, predictions, loss=loss)
        if verbose:
            print_metrics(metrics, labels, predictions, class_names,
                          header=f"Frozen embeddings ({self.model_name})")
        return metrics

    @property
    def num_features(self):
        """Embedding dimensions plus any temporal features."""
        self._check_fitted()
        return int(self.feature_scaler.n_features_in_)

    @property
    def embedding_dim(self):
        """Width of the encoder's output, once fitted."""
        self._check_fitted()
        return self._embedding_dim

    def describe_encoder(self):
        """What actually produced the vectors, for the results file.

        Recording `model_name` unconditionally would have a run that used an
        injected `embed_fn` claim it used MiniLM. A results file that names the
        wrong encoder is worse than one that names none: it survives into the
        write-up and nobody can tell.
        """
        if self.embed_fn is not None:
            name = getattr(self.embed_fn, "__name__", type(self.embed_fn).__name__)
            return f"injected embed_fn ({name})"
        return self.model_name
