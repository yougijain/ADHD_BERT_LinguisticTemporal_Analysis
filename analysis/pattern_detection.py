"""Linguistic marker extraction and linguistic-x-temporal cross analysis.

The other half of the project name. Where timestamp_analysis asks *when* people
post, this asks *how they write*, and then crosses the two: does the language of
a 3am post differ measurably from a 3pm one?

Markers are lexicon and regex based on purpose -- no spacy or nltk model
download, so this runs on a fresh clone. They are crude proxies for writing
style, not clinical indicators, and nothing here diagnoses anything.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from data.preprocess import build_labels, clean_dataset
from training.config import FIGURE_DIR, TIMESTAMP_COLUMN
from utils.time_utils import add_temporal_features

# Small hand-built lexicons. Keep them visible and editable rather than hidden
# behind a model download -- they are the analysis, so they should be auditable.
LEXICONS = {
    "first_person": {"i", "im", "ive", "id", "me", "my", "mine", "myself"},
    "negation": {"not", "no", "never", "cant", "cannot", "dont", "doesnt",
                 "didnt", "wont", "wouldnt", "nothing", "nobody"},
    "absolutist": {"always", "never", "completely", "totally", "constantly",
                   "everything", "nothing", "every", "entirely", "impossible"},
    "executive_function": {"forgot", "forget", "forgetting", "distracted",
                           "procrastinate", "procrastinating", "focus", "focusing",
                           "overwhelmed", "deadline", "late", "remind", "reminder",
                           "task", "tasks", "organize", "routine"},
    "sleep": {"sleep", "asleep", "awake", "insomnia", "tired", "exhausted",
              "bed", "night", "nights", "3am", "2am", "am"},
    "hedging": {"maybe", "probably", "kind", "sort", "guess", "somewhat",
                "might", "perhaps", "possibly"},
}

_WORD_RE = re.compile(r"[a-z']+")
_SENTENCE_RE = re.compile(r"[.!?]+")


def tokenize_words(text):
    """Lowercase word tokens. Deliberately simple and dependency-free."""
    if not isinstance(text, str):
        return []
    return _WORD_RE.findall(text.lower())


def extract_markers(text):
    """Compute the linguistic markers for one post.

    Rates are per-word so that long and short posts stay comparable; the raw
    word count is reported separately.

    Returns:
        dict[str, float]
    """
    words = tokenize_words(text)
    n_words = len(words)
    if n_words == 0:
        return {f"{name}_rate": 0.0 for name in LEXICONS} | {
            "word_count": 0.0, "avg_word_length": 0.0, "sentence_count": 0.0,
            "avg_sentence_length": 0.0, "question_rate": 0.0,
            "exclamation_rate": 0.0, "lexical_diversity": 0.0,
        }

    markers = {
        f"{name}_rate": sum(w in vocab for w in words) / n_words
        for name, vocab in LEXICONS.items()
    }

    sentences = [s for s in _SENTENCE_RE.split(text) if s.strip()]
    n_sentences = max(len(sentences), 1)

    markers.update({
        "word_count": float(n_words),
        "avg_word_length": float(np.mean([len(w) for w in words])),
        "sentence_count": float(n_sentences),
        "avg_sentence_length": n_words / n_sentences,
        # Per-sentence so a long rambling post is not penalised for having one
        # question in it.
        "question_rate": text.count("?") / n_sentences,
        "exclamation_rate": text.count("!") / n_sentences,
        # Type-token ratio: how much vocabulary variety per word.
        "lexical_diversity": len(set(words)) / n_words,
    })
    return markers


MARKER_COLUMNS = list(extract_markers("placeholder text here.").keys())


def add_marker_columns(data, text_column="clean_text"):
    """Append one column per linguistic marker to the dataframe."""
    if text_column not in data.columns:
        raise KeyError(
            f"Column '{text_column}' not found. Run clean_dataset() first, which "
            "creates 'clean_text'."
        )
    markers = pd.DataFrame(
        [extract_markers(t) for t in data[text_column]], index=data.index
    )
    return pd.concat([data, markers], axis=1)


def compare_by_group(data, group_column, marker_columns=None):
    """Mean of each marker split by a binary/categorical column.

    Returns a frame indexed by marker with one column per group plus a `diff`
    column, sorted by the size of the gap -- so the markers that most separate
    the groups sit at the top.
    """
    marker_columns = marker_columns or MARKER_COLUMNS
    available = [c for c in marker_columns if c in data.columns]
    if not available:
        raise KeyError("No marker columns present. Call add_marker_columns() first.")

    grouped = data.groupby(group_column, observed=True)[available].mean().T
    if grouped.shape[1] == 2:
        left, right = grouped.columns
        grouped["diff"] = grouped[right] - grouped[left]
        grouped = grouped.reindex(grouped["diff"].abs().sort_values(ascending=False).index)
    return grouped


def marker_label_correlations(data, labels, marker_columns=None):
    """Pearson correlation between each marker and the binary label.

    A marker with near-zero correlation contributes nothing; this is the quick
    check on whether the linguistic side carries signal at all.
    """
    marker_columns = marker_columns or MARKER_COLUMNS
    available = [c for c in marker_columns if c in data.columns]
    target = pd.Series(np.asarray(labels, dtype="float64"), index=data.index)

    correlations = {}
    for column in available:
        series = data[column].astype("float64")
        # A constant column has zero variance and correlates with nothing;
        # np.corrcoef would return NaN and emit a divide-by-zero warning.
        if series.std(ddof=0) == 0:
            correlations[column] = 0.0
        else:
            correlations[column] = float(series.corr(target))

    result = pd.Series(correlations, name="correlation_with_label")
    return result.reindex(result.abs().sort_values(ascending=False).index)


def plot_marker_comparison(comparison, output_dir=FIGURE_DIR, title=None,
                           filename="linguistic_markers.png", top_n=8):
    """Horizontal bar chart of the largest between-group marker gaps."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if "diff" not in comparison.columns:
        raise ValueError("compare_by_group must have produced a 'diff' column "
                         "(it needs exactly two groups).")

    top = comparison.head(top_n).iloc[::-1]
    colors = ["#C44E52" if v > 0 else "#4C72B0" for v in top["diff"]]

    fig, ax = plt.subplots(figsize=(8, 0.5 * len(top) + 1.8))
    ax.barh(top.index, top["diff"], color=colors)
    ax.axvline(0, color="#444", linewidth=0.8)
    ax.set_xlabel("Difference in mean rate (group 1 minus group 0)")
    ax.set_title(title or "Linguistic markers by group")
    fig.tight_layout()

    path = output_dir / filename
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def run(dataset_path, output_dir=FIGURE_DIR, label_strategy="median"):
    """Full linguistic-temporal cross analysis on a dataset."""
    print(f"Loading {dataset_path}...")
    data = pd.read_csv(dataset_path)
    data = clean_dataset(data)
    data = add_temporal_features(data, TIMESTAMP_COLUMN)
    data = add_marker_columns(data)
    labels, label_summary = build_labels(data, strategy=label_strategy)

    print(f"\nLabel summary: {label_summary}")

    print("\nLinguistic markers, late-night vs daytime posts")
    by_night = compare_by_group(data, "is_late_night")
    print(by_night.round(4).to_string())

    print("\nMarker correlation with the engagement label")
    correlations = marker_label_correlations(data, labels)
    print(correlations.round(4).to_string())

    figures = []
    if "diff" in by_night.columns:
        figures.append(plot_marker_comparison(
            by_night, output_dir,
            title="Linguistic markers: late-night minus daytime posts",
            filename="markers_late_night.png",
        ))

    print("\nFigures written:")
    for path in figures:
        print(f"  {path}")

    return {
        "label_summary": label_summary,
        "late_night_comparison": by_night.round(6).to_dict(),
        "label_correlations": correlations.round(6).to_dict(),
    }


def main():
    parser = argparse.ArgumentParser(description="Linguistic pattern analysis.")
    parser.add_argument("--dataset", required=True, help="Path to the CSV.")
    parser.add_argument("--output-dir", default=str(FIGURE_DIR))
    parser.add_argument("--label-strategy", default="median",
                        choices=["median", "positive", "threshold"])
    args = parser.parse_args()
    run(args.dataset, args.output_dir, args.label_strategy)


if __name__ == "__main__":
    main()
