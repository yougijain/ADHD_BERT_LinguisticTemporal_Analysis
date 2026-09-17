"""Cleaning, labelling, and tokenization."""

import numpy as np
import pandas as pd
import pytest

from data.preprocess import (
    batch_tokenize,
    build_labels,
    clean_dataset,
    clean_text,
    is_placeholder,
)


class TestCleanText:
    def test_strips_urls_and_markdown(self):
        out = clean_text("See **this** at https://example.com/x now")
        assert "http" not in out and "*" not in out
        assert "See" in out and "now" in out

    def test_keeps_sentence_punctuation(self):
        # The old regex kept only [a-zA-Z\s], deleting every ? and ! and digit.
        out = clean_text("I have 3 tabs open. Why?! It's a lot")
        assert "?" in out and "!" in out and "3" in out and "'" in out

    def test_placeholders_become_empty(self):
        for value in ("[removed]", "[deleted]", "  [REMOVED]  ", ""):
            assert clean_text(value) == ""

    def test_non_string_input(self):
        assert clean_text(None) == ""
        assert clean_text(float("nan")) == ""
        assert clean_text(123) == ""

    def test_collapses_whitespace(self):
        assert clean_text("a\n\n  b\t\tc") == "a b c"


class TestIsPlaceholder:
    @pytest.mark.parametrize("value", ["[removed]", "[deleted]", "", None, float("nan"), 42])
    def test_detects(self, value):
        assert is_placeholder(value)

    def test_real_text_is_not_placeholder(self):
        assert not is_placeholder("an actual post body")


class TestCleanDataset:
    def test_drops_placeholder_rows_instead_of_filling_them(self):
        # The original filled NaN selftext with "missing_text" and then dropped
        # NA on that column, so removed posts survived as one repeated string.
        frame = pd.DataFrame({
            "selftext": ["a genuinely long enough body of text here", "[removed]", None],
            "score": [5, 5, 5],
            "created_utc": [1700000000] * 3,
        })
        out = clean_dataset(frame, min_tokens=3)
        assert len(out) == 1
        assert "missing_text" not in out["clean_text"].tolist()

    def test_drops_short_posts(self, raw_frame):
        out = clean_dataset(raw_frame, min_tokens=5)
        assert (out["clean_text"].str.split().str.len() >= 5).all()

    def test_folds_title_into_text(self):
        frame = pd.DataFrame({
            "title": ["Focus question"],
            "selftext": ["body text that is long enough to survive"],
            "score": [3], "created_utc": [1700000000],
        })
        assert "Focus question" in clean_dataset(frame, min_tokens=3)["clean_text"].iloc[0]

    def test_missing_column_names_the_column(self):
        with pytest.raises(KeyError, match="score"):
            clean_dataset(pd.DataFrame({"selftext": ["x"], "created_utc": [1]}))

    def test_all_junk_raises_rather_than_returning_empty(self):
        frame = pd.DataFrame({
            "selftext": ["[removed]"] * 5, "score": [1] * 5,
            "created_utc": [1700000000] * 5,
        })
        with pytest.raises(ValueError, match="No usable rows"):
            clean_dataset(frame)

    def test_resets_index(self, raw_frame):
        out = clean_dataset(raw_frame)
        assert out.index.tolist() == list(range(len(out)))


class TestBuildLabels:
    def test_positive_strategy_is_degenerate_on_reddit_scores(self):
        # Documents the original bug: Reddit posts start at 1, so score > 0
        # puts everything in class 1.
        frame = pd.DataFrame({"score": [1] * 60 + [2] * 20 + [5] * 20})
        labels, summary = build_labels(frame, strategy="positive")
        assert summary["positive_rate"] == 1.0
        assert len(np.unique(labels)) == 1

    def test_median_strategy_splits_the_same_data(self):
        frame = pd.DataFrame({"score": [1] * 60 + [2] * 20 + [5] * 20})
        labels, summary = build_labels(frame, strategy="median")
        assert len(np.unique(labels)) == 2
        assert 0.2 < summary["positive_rate"] < 0.8

    def test_constant_scores_cannot_be_split(self):
        labels, summary = build_labels(pd.DataFrame({"score": [1] * 10}), "median")
        assert len(np.unique(labels)) == 1
        assert summary["class_1"] == 0

    def test_threshold_strategy(self):
        labels, summary = build_labels(
            pd.DataFrame({"score": [0, 1, 2, 3]}), "threshold", threshold=1
        )
        assert labels.tolist() == [0, 0, 1, 1]
        assert summary["cut_point"] == 1.0

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown label strategy"):
            build_labels(pd.DataFrame({"score": [1, 2]}), "nonsense")

    def test_labels_align_with_rows(self, clean_frame):
        labels, _ = build_labels(clean_frame)
        assert len(labels) == len(clean_frame)
        assert labels.dtype == np.int64


class TestBatchTokenize:
    def test_all_batches_share_one_width(self, local_tokenizer):
        # Each batch used to be padded to its own longest row, then torch.cat-ed,
        # which raises a size mismatch as soon as two batches disagree.
        texts = ["short text"] * 4 + ["a much longer piece of text " * 10] * 4
        enc = batch_tokenize(texts, tokenizer=local_tokenizer, batch_size=4,
                             max_length=32, verbose=False)
        assert enc["input_ids"].shape == (8, 32)
        assert enc["attention_mask"].shape == (8, 32)

    def test_empty_input(self, local_tokenizer):
        enc = batch_tokenize([], tokenizer=local_tokenizer, max_length=16, verbose=False)
        assert enc["input_ids"].shape[0] == 0

    def test_handles_none_entries(self, local_tokenizer):
        enc = batch_tokenize(["ok", None], tokenizer=local_tokenizer,
                             max_length=16, verbose=False)
        assert enc["input_ids"].shape == (2, 16)

    def test_truncates_to_max_length(self, local_tokenizer):
        enc = batch_tokenize(["word " * 500], tokenizer=local_tokenizer,
                             max_length=24, verbose=False)
        assert enc["input_ids"].shape == (1, 24)
