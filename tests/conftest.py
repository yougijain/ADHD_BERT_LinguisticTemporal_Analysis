"""Shared fixtures. Nothing here touches the network."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.make_sample_data import generate_dataset  # noqa: E402
from data.preprocess import build_local_tokenizer, clean_dataset  # noqa: E402


@pytest.fixture(scope="session")
def raw_frame():
    """A small synthetic dump, including the junk rows a real one has."""
    return generate_dataset(n_rows=200, seed=7)


@pytest.fixture(scope="session")
def clean_frame(raw_frame):
    return clean_dataset(raw_frame, min_tokens=5)


@pytest.fixture(scope="session")
def local_tokenizer(clean_frame):
    return build_local_tokenizer(clean_frame["clean_text"].tolist(), vocab_size=500)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
