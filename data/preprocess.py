"""Text cleaning, label construction, and batched BERT tokenization."""

import re

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from training.config import (
    REDDIT_PLACEHOLDERS,
    SCORE_COLUMN,
    TEXT_COLUMN,
    TIMESTAMP_COLUMN,
    TITLE_COLUMN,
)

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MARKDOWN_RE = re.compile(r"[*_~`>#\[\]()]")
_WHITESPACE_RE = re.compile(r"\s+")
# Keep letters, digits, and the punctuation that carries tone. The old version
# stripped everything but [a-zA-Z\s], which deleted numbers, apostrophes, and
# every question/exclamation mark -- exactly the punctuation an analysis of
# linguistic style needs.
_KEEP_RE = re.compile(r"[^a-zA-Z0-9\s.,!?'\-]")

_TOKENIZER_CACHE = {}


def get_tokenizer(model_name="bert-base-uncased"):
    """Return a cached tokenizer for `model_name`.

    Loading the tokenizer is slow, and the old module-level instantiation meant
    merely importing this file hit the network. Caching here keeps repeated
    calls cheap while leaving import side-effect free.
    """
    if model_name not in _TOKENIZER_CACHE:
        _TOKENIZER_CACHE[model_name] = AutoTokenizer.from_pretrained(model_name)
    return _TOKENIZER_CACHE[model_name]


def build_local_tokenizer(texts, vocab_size=2000, lowercase=True):
    """Train a small WordPiece tokenizer on the corpus itself. No network.

    `bert-base-uncased` is a 440MB download from huggingface.co, which is not
    available in every environment (air-gapped boxes, locked-down CI, a proxy
    that blocks the hub). This builds a BERT-compatible tokenizer from the text
    you already have, so the pipeline has a genuinely offline smoke path.

    The vocabulary is corpus-specific and carries none of BERT's pretraining, so
    this is for exercising the plumbing -- not for results.

    Returns:
        PreTrainedTokenizerFast: usable anywhere the HF tokenizer is.
    """
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors
    from tokenizers.trainers import WordPieceTrainer
    from transformers import PreTrainedTokenizerFast

    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    normalizer_steps = [normalizers.NFD(), normalizers.StripAccents()]
    if lowercase:
        normalizer_steps.append(normalizers.Lowercase())
    tokenizer.normalizer = normalizers.Sequence(normalizer_steps)
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.decoder = decoders.WordPiece(prefix="##")

    trainer = WordPieceTrainer(vocab_size=vocab_size, special_tokens=special_tokens,
                               min_frequency=1)
    corpus = [t for t in texts if isinstance(t, str) and t.strip()] or ["placeholder"]
    tokenizer.train_from_iterator(corpus, trainer=trainer)

    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls_id), ("[SEP]", sep_id)],
    )

    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )


def clean_text(text):
    """Normalize a single post body for BERT.

    Strips URLs, markdown syntax, and stray symbols while preserving sentence
    punctuation. Returns "" for anything that is not usable text.
    """
    if text is None or not isinstance(text, str):
        return ""
    if text.strip().lower() in REDDIT_PLACEHOLDERS:
        return ""

    text = _URL_RE.sub(" ", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = _MARKDOWN_RE.sub(" ", text)
    text = _KEEP_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def is_placeholder(text):
    """True if the value is one of Reddit's removed/deleted markers or empty."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return True
    if not isinstance(text, str):
        return True
    return text.strip().lower() in REDDIT_PLACEHOLDERS


def clean_dataset(data, min_tokens=5, combine_title=True):
    """Clean a raw Reddit dump into rows that are safe to train on.

    The previous implementation filled missing bodies with the literal string
    "missing_text" and *then* dropped NA on that column, so the dropna was a
    no-op and every removed post survived into training as the same sentinel
    string. Here, unusable rows are dropped outright.

    Args:
        data (pd.DataFrame): Raw dataset.
        min_tokens (int): Drop posts with fewer whitespace tokens than this.
        combine_title (bool): Prepend the title to the body when a title column
            exists. Titles carry a lot of signal on Reddit.
    Returns:
        pd.DataFrame: Cleaned copy with a reset index and a `clean_text` column.
    Raises:
        KeyError: If a required column is missing.
    """
    required = (TEXT_COLUMN, SCORE_COLUMN, TIMESTAMP_COLUMN)
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise KeyError(
            f"Dataset is missing required column(s): {missing}. "
            f"Expected at least {list(required)}."
        )

    data = data.copy()
    before = len(data)

    # Drop posts whose body was removed/deleted rather than substituting a
    # sentinel string for them.
    data = data[~data[TEXT_COLUMN].map(is_placeholder)]

    body = data[TEXT_COLUMN].map(clean_text)
    if combine_title and TITLE_COLUMN in data.columns:
        title = data[TITLE_COLUMN].map(clean_text)
        combined = (title + ". " + body).str.strip(". ").str.strip()
        # If the body cleaned away to nothing, the title alone still counts.
        data["clean_text"] = combined.where(combined.str.len() > 0, title)
    else:
        data["clean_text"] = body

    # Numeric score; rows without one cannot be labelled.
    data[SCORE_COLUMN] = pd.to_numeric(data[SCORE_COLUMN], errors="coerce")

    data = data.dropna(subset=[SCORE_COLUMN, TIMESTAMP_COLUMN])
    data = data[data["clean_text"].str.split().str.len() >= min_tokens]
    data = data.reset_index(drop=True)

    print(f"Cleaned dataset: {before} rows in, {len(data)} usable rows out.")
    if len(data) == 0:
        raise ValueError(
            "No usable rows survived cleaning. Check that the CSV has real post "
            f"bodies in '{TEXT_COLUMN}' and try lowering min_tokens."
        )
    return data


def _balanced_cut(scores):
    """Pick the score threshold that splits the data closest to 50/50.

    The plain median is often useless on Reddit scores because a large share of
    posts tie at exactly 1 -- `score > median` can then put everything in one
    class. Searching the distinct score values for the most balanced cut avoids
    that without inventing an ordering among tied posts, which would make the
    label pure noise.
    """
    distinct = np.unique(scores.dropna().to_numpy())
    if len(distinct) < 2:
        # Genuinely constant scores: no engagement signal exists to label on.
        return float(distinct[0]) if len(distinct) else 0.0

    # Only values below the max can serve as a cut; `> max` is always empty.
    candidates = distinct[:-1]
    rates = np.array([(scores > c).mean() for c in candidates])
    return float(candidates[np.argmin(np.abs(rates - 0.5))])


def build_labels(data, strategy="median", threshold=1, score_column=SCORE_COLUMN):
    """Turn post scores into binary classification targets.

    The original pipeline used `score > 0`, which is degenerate on Reddit: posts
    start at a score of 1 and downvoted posts bottom out at 0, so well over 90%
    of any real dump lands in class 1 and the model learns to answer "1" always.

    Strategies:
        median    -- above the dataset median score vs at-or-below. Balanced by
                     construction, and reads as "did this post land above
                     typical engagement for this subreddit".
        threshold -- score > `threshold`.
        positive  -- the original score > 0 rule, kept for reproducibility.

    Returns:
        tuple[np.ndarray, dict]: int64 labels and a summary dict describing the
        cut point and resulting class balance.
    """
    scores = pd.to_numeric(data[score_column], errors="coerce").fillna(0)

    if strategy == "median":
        cut = _balanced_cut(scores)
        labels = (scores > cut).astype("int64")
    elif strategy == "threshold":
        cut = float(threshold)
        labels = (scores > cut).astype("int64")
    elif strategy == "positive":
        cut = 0.0
        labels = (scores > 0).astype("int64")
    else:
        raise ValueError(f"Unknown label strategy: {strategy!r}")

    labels = labels.to_numpy(dtype="int64")
    counts = np.bincount(labels, minlength=2)
    summary = {
        "strategy": strategy,
        "cut_point": cut,
        "class_0": int(counts[0]),
        "class_1": int(counts[1]),
        "positive_rate": float(labels.mean()) if len(labels) else 0.0,
    }

    if summary["class_0"] == 0 or summary["class_1"] == 0:
        print(
            f"WARNING: label strategy '{strategy}' produced a single class "
            f"({summary}). The model cannot learn anything from this split. "
            "If every post has the same score there is no engagement signal to "
            "predict -- pick a different target column."
        )
    return labels, summary


def batch_tokenize(texts, model_name="bert-base-uncased", batch_size=512,
                   max_length=256, verbose=True, tokenizer=None):
    """Tokenize texts in batches, padded to a single common length.

    Each batch used to be padded to its own longest sequence and the batches
    were then torch.cat-ed together, which raises a size-mismatch as soon as two
    batches disagree on length. Padding to `max_length` makes the concatenation
    well-defined.

    Args:
        tokenizer: A pre-built tokenizer. When given, `model_name` is ignored --
            this is how the offline path passes in build_local_tokenizer's
            output.
    Returns:
        dict[str, torch.Tensor]: input_ids and attention_mask.
    """
    import torch

    if tokenizer is None:
        tokenizer = get_tokenizer(model_name)
    texts = ["" if t is None else str(t) for t in texts]

    if not texts:
        empty = torch.empty((0, max_length), dtype=torch.long)
        return {"input_ids": empty, "attention_mask": empty.clone()}

    input_ids, attention_mask = [], []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        if verbose:
            print(f"  Tokenizing batch {i // batch_size + 1}/{total_batches}")
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids.append(encoded["input_ids"])
        attention_mask.append(encoded["attention_mask"])

    return {
        "input_ids": torch.cat(input_ids, dim=0),
        "attention_mask": torch.cat(attention_mask, dim=0),
    }


def preprocess_text(texts, model_name="bert-base-uncased", max_length=256,
                    batch_size=512, verbose=True, tokenizer=None):
    """Clean then tokenize a list of raw texts. Convenience wrapper."""
    if verbose:
        print("Cleaning text data...")
    cleaned = [clean_text(t) for t in texts]
    return batch_tokenize(
        cleaned,
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
        verbose=verbose,
        tokenizer=tokenizer,
    )
