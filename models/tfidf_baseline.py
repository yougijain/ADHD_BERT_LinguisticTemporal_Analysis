"""TF-IDF + logistic regression baseline.

The number that tells you whether fine-tuning BERT was worth it. On short
user-generated text with a few thousand examples, a linear model over character
and word n-grams is a genuinely strong competitor -- often the winner -- and
without it a BERT accuracy figure has nothing to be compared against.

Deliberately mirrors the neural path: the same cleaned text, the same labels,
the same split indices, and the same metrics via training.evaluate. The only
thing that differs is the model, which is the point.

It also takes the same temporal features, so the text-only vs text+temporal
ablation runs on both architectures rather than just the neural one.
"""

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from training.evaluate import compute_metrics, print_metrics


class TfidfBaseline:
    """TF-IDF features (optionally concatenated with temporal features) -> logistic regression.

    Args:
        max_features (int): Vocabulary cap for the word n-gram vectorizer.
        ngram_range (tuple): Word n-gram range. (1, 2) picks up short phrases.
        min_df (int): Ignore terms appearing in fewer than this many documents.
        use_char_ngrams (bool): Add character n-grams alongside word n-grams.
            They are robust to the typos and inconsistent spacing that fill
            user-generated text.
        use_temporal_features (bool): Concatenate the temporal feature matrix.
        C (float): Inverse regularisation strength for the classifier.
        class_weight (str | None): "balanced" reweights by inverse class
            frequency. Worth setting when the label split is lopsided.
        max_iter (int): Solver iteration cap.
        seed (int): Random state.
    """

    def __init__(self, max_features=20000, ngram_range=(1, 2), min_df=2,
                 use_char_ngrams=True, use_temporal_features=True, C=1.0,
                 class_weight=None, max_iter=1000, seed=42):
        self.use_temporal_features = use_temporal_features
        self.use_char_ngrams = use_char_ngrams

        self.word_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            sublinear_tf=True,  # damps the effect of one word repeated many times
            strip_accents="unicode",
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=max_features,
            min_df=min_df,
            sublinear_tf=True,
        ) if use_char_ngrams else None

        # Temporal features are dense and on a different scale from TF-IDF
        # weights; without scaling the regulariser effectively ignores them.
        self.temporal_scaler = StandardScaler() if use_temporal_features else None

        self.classifier = LogisticRegression(
            C=C, class_weight=class_weight, max_iter=max_iter,
            random_state=seed, solver="liblinear",
        )
        self._fitted = False

    def _build_matrix(self, texts, temporal_features=None, fit=False):
        """Vectorize text and stack on the temporal block."""
        texts = ["" if t is None else str(t) for t in texts]

        if fit:
            blocks = [self.word_vectorizer.fit_transform(texts)]
            if self.char_vectorizer is not None:
                blocks.append(self.char_vectorizer.fit_transform(texts))
        else:
            blocks = [self.word_vectorizer.transform(texts)]
            if self.char_vectorizer is not None:
                blocks.append(self.char_vectorizer.transform(texts))

        if self.use_temporal_features:
            if temporal_features is None:
                raise ValueError(
                    "use_temporal_features=True but no temporal_features were "
                    "passed. Provide them, or construct with "
                    "use_temporal_features=False for the text-only baseline."
                )
            dense = np.asarray(temporal_features, dtype="float64")
            if dense.ndim == 1:
                dense = dense.reshape(-1, 1)
            if len(dense) != len(texts):
                raise ValueError(
                    f"Got {len(dense)} temporal feature rows for {len(texts)} "
                    "texts. They must align."
                )
            dense = (self.temporal_scaler.fit_transform(dense) if fit
                     else self.temporal_scaler.transform(dense))
            blocks.append(sparse.csr_matrix(dense))

        return sparse.hstack(blocks, format="csr")

    def fit(self, texts, labels, temporal_features=None):
        """Fit the vectorizers and the classifier. Returns self."""
        labels = np.asarray(labels)
        if len(np.unique(labels)) < 2:
            raise ValueError(
                "Training labels contain a single class; logistic regression "
                "cannot fit. Check the label strategy."
            )

        matrix = self._build_matrix(texts, temporal_features, fit=True)
        self.classifier.fit(matrix, labels)
        self._fitted = True
        return self

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Call fit() before predicting.")

    def predict(self, texts, temporal_features=None):
        self._check_fitted()
        return self.classifier.predict(self._build_matrix(texts, temporal_features))

    def predict_proba(self, texts, temporal_features=None):
        self._check_fitted()
        return self.classifier.predict_proba(self._build_matrix(texts, temporal_features))

    def evaluate(self, texts, labels, temporal_features=None, verbose=True,
                 class_names=None):
        """Score predictions with the same compute_metrics the neural path uses.

        Returns:
            dict: Identical shape to training.evaluate.evaluate_model's output.
        """
        self._check_fitted()
        predictions = self.predict(texts, temporal_features)

        # log_loss, for comparability with the neural model's cross-entropy.
        loss = 0.0
        try:
            from sklearn.metrics import log_loss
            probabilities = self.predict_proba(texts, temporal_features)
            loss = float(log_loss(labels, probabilities,
                                  labels=list(range(probabilities.shape[1]))))
        except (ValueError, ImportError):
            # log_loss raises when a split is missing a class. Not fatal --
            # accuracy and F1 are still meaningful, so report zero loss.
            loss = 0.0

        metrics = compute_metrics(labels, predictions, loss=loss)
        if verbose:
            print_metrics(metrics, labels, predictions, class_names,
                          header="TF-IDF baseline")
        return metrics

    @property
    def num_features(self):
        """Total feature count across every block."""
        self._check_fitted()
        total = len(self.word_vectorizer.vocabulary_)
        if self.char_vectorizer is not None:
            total += len(self.char_vectorizer.vocabulary_)
        if self.use_temporal_features:
            total += self.temporal_scaler.n_features_in_
        return total

    def top_features(self, n=15, temporal_feature_names=None):
        """The n most positive and most negative word features, by coefficient.

        This is the baseline's answer to the neural model's attention weights,
        and it is a good deal more legible: these are literal words with signed
        weights, not a distribution over wordpieces. If the top features look
        like artefacts, the label is leaking.

        Returns:
            dict: "positive" and "negative" lists of (feature, weight), plus
            "temporal" when temporal features are in use.
        """
        self._check_fitted()
        coefficients = self.classifier.coef_[0]

        word_names = self.word_vectorizer.get_feature_names_out()
        word_coefficients = coefficients[:len(word_names)]

        order = np.argsort(word_coefficients)
        result = {
            "negative": [(word_names[i], float(word_coefficients[i]))
                         for i in order[:n]],
            "positive": [(word_names[i], float(word_coefficients[i]))
                         for i in order[-n:][::-1]],
        }

        if self.use_temporal_features:
            n_temporal = self.temporal_scaler.n_features_in_
            temporal_coefficients = coefficients[-n_temporal:]
            names = temporal_feature_names or [f"temporal_{i}" for i in range(n_temporal)]
            result["temporal"] = [
                (name, float(weight))
                for name, weight in zip(names, temporal_coefficients)
            ]
        return result


def print_top_features(top, n=10):
    """Print the output of top_features in a readable block."""
    print(f"\n  Words pushing toward class 1 (top {n}):")
    for name, weight in top["positive"][:n]:
        print(f"    {name:<24} {weight:+.4f}")
    print(f"\n  Words pushing toward class 0 (top {n}):")
    for name, weight in top["negative"][:n]:
        print(f"    {name:<24} {weight:+.4f}")
    if "temporal" in top:
        print("\n  Temporal feature weights:")
        for name, weight in sorted(top["temporal"], key=lambda x: -abs(x[1])):
            print(f"    {name:<24} {weight:+.4f}")
