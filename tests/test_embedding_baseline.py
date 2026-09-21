"""Frozen-embedding baseline tests.

The encoder is injected via `embed_fn`, so nothing here downloads a model. What
is worth testing is the wiring around the encoder -- scaling, the temporal
block, the fit/predict contract -- because that is where a silent mistake turns
into a plausible-looking number.
"""

import numpy as np
import pytest

from models.embedding_baseline import DEFAULT_ENCODER, EmbeddingBaseline

DIM = 12


def fake_embedder(dim=DIM, seed=0):
    """Deterministic, content-dependent vectors. Same text -> same vector.

    Hashing the text rather than returning noise matters: a classifier fitted on
    random vectors cannot separate anything, so a test asserting it learns
    something would be asserting the wrong thing.
    """
    def embed(texts):
        out = np.zeros((len(texts), dim), dtype="float32")
        for row, text in enumerate(texts):
            rng = np.random.default_rng(abs(hash(str(text))) % (2**32) + seed)
            out[row] = rng.normal(size=dim)
        return out
    return embed


def separable_embedder(dim=DIM):
    """Vectors that encode the label in the first component, for fit tests."""
    def embed(texts):
        out = np.zeros((len(texts), dim), dtype="float32")
        for row, text in enumerate(texts):
            out[row, 0] = 1.0 if "positive" in str(text) else -1.0
            out[row, 1:] = 0.01 * (row % 3)
        return out
    return embed


@pytest.fixture
def texts():
    return [f"post number {i} is positive" if i % 2 else f"post number {i}"
            for i in range(40)]


@pytest.fixture
def labels(texts):
    return np.array([1 if "positive" in t else 0 for t in texts])


@pytest.fixture
def temporal(texts):
    rng = np.random.default_rng(3)
    return rng.normal(size=(len(texts), 8)).astype("float32")


class TestConstruction:
    def test_default_encoder_is_a_sentence_transformer(self):
        assert "sentence-transformers" in DEFAULT_ENCODER

    def test_predicting_before_fitting_raises(self):
        model = EmbeddingBaseline(embed_fn=fake_embedder())
        with pytest.raises(RuntimeError, match="Call fit"):
            model.predict(["anything"])

    def test_num_features_before_fitting_raises(self):
        model = EmbeddingBaseline(embed_fn=fake_embedder())
        with pytest.raises(RuntimeError, match="Call fit"):
            _ = model.num_features


class TestFit:
    def test_fits_and_learns_a_separable_signal(self, texts, labels):
        model = EmbeddingBaseline(use_temporal_features=False,
                                  embed_fn=separable_embedder())
        model.fit(texts, labels, verbose=False)
        assert (model.predict(texts) == labels).mean() > 0.9

    def test_single_class_labels_raise(self, texts):
        model = EmbeddingBaseline(use_temporal_features=False,
                                  embed_fn=fake_embedder())
        with pytest.raises(ValueError, match="single class"):
            model.fit(texts, np.zeros(len(texts), dtype=int), verbose=False)

    def test_embedding_dim_is_recorded(self, texts, labels):
        model = EmbeddingBaseline(use_temporal_features=False,
                                  embed_fn=fake_embedder())
        model.fit(texts, labels, verbose=False)
        assert model.embedding_dim == DIM

    def test_temporal_features_widen_the_matrix(self, texts, labels, temporal):
        model = EmbeddingBaseline(embed_fn=fake_embedder())
        model.fit(texts, labels, temporal, verbose=False)
        assert model.num_features == DIM + temporal.shape[1]
        assert model.embedding_dim == DIM

    def test_text_only_matrix_is_just_the_embedding(self, texts, labels):
        model = EmbeddingBaseline(use_temporal_features=False,
                                  embed_fn=fake_embedder())
        model.fit(texts, labels, verbose=False)
        assert model.num_features == DIM


class TestTemporalContract:
    def test_missing_temporal_features_raise(self, texts, labels):
        model = EmbeddingBaseline(use_temporal_features=True,
                                  embed_fn=fake_embedder())
        with pytest.raises(ValueError, match="no temporal_features"):
            model.fit(texts, labels, verbose=False)

    def test_misaligned_temporal_features_raise(self, texts, labels):
        model = EmbeddingBaseline(embed_fn=fake_embedder())
        with pytest.raises(ValueError, match="must align"):
            model.fit(texts, labels, np.zeros((3, 8)), verbose=False)

    def test_one_dimensional_temporal_block_is_reshaped(self, texts, labels):
        model = EmbeddingBaseline(embed_fn=fake_embedder())
        model.fit(texts, labels, np.arange(len(texts)), verbose=False)
        assert model.num_features == DIM + 1


class TestEmbedFnContract:
    def test_wrong_row_count_raises(self, texts, labels):
        model = EmbeddingBaseline(use_temporal_features=False,
                                  embed_fn=lambda t: np.zeros((2, DIM)))
        with pytest.raises(ValueError, match="expected \\(n_texts"):
            model.fit(texts, labels, verbose=False)

    def test_one_dimensional_output_raises(self, texts, labels):
        model = EmbeddingBaseline(use_temporal_features=False,
                                  embed_fn=lambda t: np.zeros(len(t)))
        with pytest.raises(ValueError, match="expected \\(n_texts"):
            model.fit(texts, labels, verbose=False)

    def test_changing_dimension_between_fit_and_predict_raises(self, texts, labels):
        calls = {"n": 0}

        def shifting(t):
            calls["n"] += 1
            width = DIM if calls["n"] == 1 else DIM + 4
            return np.zeros((len(t), width))

        model = EmbeddingBaseline(use_temporal_features=False, embed_fn=shifting)
        model.fit(texts, labels, verbose=False)
        with pytest.raises(ValueError, match="encoder changed|Encoder returned"):
            model.predict(texts)


class TestEvaluate:
    def test_metrics_match_the_shared_schema(self, texts, labels, temporal):
        model = EmbeddingBaseline(embed_fn=separable_embedder())
        model.fit(texts, labels, temporal, verbose=False)
        metrics = model.evaluate(texts, labels, temporal, verbose=False)

        for key in ("accuracy", "macro_f1", "majority_baseline",
                    "lift_over_baseline", "loss"):
            assert key in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_evaluate_embeds_the_set_once(self, texts, labels):
        # Embedding is the expensive part; scoring must not pay for it twice.
        calls = {"n": 0}

        def counting(t):
            calls["n"] += 1
            return separable_embedder()(t)

        model = EmbeddingBaseline(use_temporal_features=False, embed_fn=counting)
        model.fit(texts, labels, verbose=False)
        before = calls["n"]
        model.evaluate(texts, labels, verbose=False)
        assert calls["n"] - before == 1

    def test_probabilities_sum_to_one(self, texts, labels):
        model = EmbeddingBaseline(use_temporal_features=False,
                                  embed_fn=separable_embedder())
        model.fit(texts, labels, verbose=False)
        probabilities = model.predict_proba(texts)
        assert np.allclose(probabilities.sum(axis=1), 1.0)


class TestEncodeTexts:
    """The real encoder path. Only the parts that need no download."""

    def test_empty_input_returns_an_empty_array(self):
        from models.embedding_baseline import encode_texts
        assert encode_texts([], verbose=False).shape[0] == 0


class TestDescribeEncoder:
    """A results file that names the wrong encoder survives into the write-up."""

    def test_reports_the_model_name_when_no_embed_fn(self):
        assert EmbeddingBaseline().describe_encoder() == DEFAULT_ENCODER

    def test_reports_the_injection_when_embed_fn_is_used(self):
        model = EmbeddingBaseline(embed_fn=fake_embedder())
        described = model.describe_encoder()
        assert "injected" in described
        assert DEFAULT_ENCODER not in described
