"""Map an arbitrary CSV onto the columns the pipeline expects.

The pipeline works in one schema -- `selftext`, `score`, `created_utc`, with
`title` used when present -- because every downstream module (cleaning,
labelling, temporal features, error slices) refers to those names. Real corpora
do not use them. Stack Exchange calls the body `body` and the timestamp
`creation_date`; a Hacker News export calls them `text` and `time`; a Kaggle
dump calls them whatever the uploader felt like.

Rather than teach every module to accept aliases, normalise once at load time.
Two layers, in this order:

1. An explicit `--column-map`, which always wins. Nothing is guessed.
2. Inference from a table of known aliases, used *only* to fill a canonical
   column that is genuinely absent. An existing `score` column is never
   silently replaced by a column named `points`.

Inference that only fires on a missing column is the important property. A
mapping layer that quietly reinterprets columns you already have is a way to
train on the wrong target and not find out.
"""

import pandas as pd

from training.config import (
    SCORE_COLUMN,
    TEXT_COLUMN,
    TIMESTAMP_COLUMN,
    TITLE_COLUMN,
)

# The columns the pipeline refers to by name. TITLE_COLUMN is optional --
# clean_dataset folds it into the body when it exists and ignores it otherwise.
REQUIRED_COLUMNS = (TEXT_COLUMN, SCORE_COLUMN, TIMESTAMP_COLUMN)
OPTIONAL_COLUMNS = (TITLE_COLUMN,)

# Known aliases, lowercased. Order matters: the first alias present in the frame
# wins, so put the least ambiguous name first.
COLUMN_ALIASES = {
    TEXT_COLUMN: ("selftext", "body", "text", "content", "post", "body_markdown",
                  "question_body", "description"),
    SCORE_COLUMN: ("score", "points", "ups", "upvotes", "votes", "vote_count",
                   "num_points", "reactions"),
    TIMESTAMP_COLUMN: ("created_utc", "creation_date", "created_at", "created",
                       "timestamp", "time", "date", "post_date", "published_at"),
    TITLE_COLUMN: ("title", "headline", "subject", "question_title"),
}


def parse_column_map(spec):
    """Parse a `canonical=source,canonical=source` string into a dict.

    Args:
        spec (str | None): e.g. "selftext=body,created_utc=creation_date".
    Returns:
        dict[str, str]: canonical column name -> source column name.
    Raises:
        ValueError: On a malformed pair or an unknown canonical name.
    """
    if not spec:
        return {}

    mapping = {}
    for pair in spec.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(
                f"Malformed --column-map entry {pair!r}. "
                "Expected canonical=source, e.g. selftext=body."
            )
        canonical, source = (part.strip() for part in pair.split("=", 1))
        known = REQUIRED_COLUMNS + OPTIONAL_COLUMNS
        if canonical not in known:
            raise ValueError(
                f"Unknown canonical column {canonical!r}. "
                f"Mappable columns are {list(known)}."
            )
        if not source:
            raise ValueError(f"No source column given for {canonical!r}.")
        mapping[canonical] = source
    return mapping


def infer_column_map(frame):
    """Guess a mapping for canonical columns the frame does not already have.

    Only missing columns are inferred. A frame that already has `score` keeps
    it, even if it also has `points`.

    Returns:
        dict[str, str]: canonical -> source, for inferred columns only. Empty
        when nothing needed inferring or nothing matched.
    """
    lowered = {str(c).lower(): str(c) for c in frame.columns}
    inferred = {}

    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in frame.columns:
            continue
        for alias in aliases:
            if alias in lowered:
                inferred[canonical] = lowered[alias]
                break
    return inferred


def normalize_columns(frame, column_map=None, verbose=True):
    """Rename `frame`'s columns onto the canonical schema.

    Args:
        frame (pd.DataFrame): The raw frame as read from disk.
        column_map (dict | str | None): Explicit mapping, or a spec string for
            parse_column_map. Applied before inference and never overridden.
        verbose (bool): Print what was renamed and how it was decided.
    Returns:
        pd.DataFrame: A copy with canonical column names.
    Raises:
        KeyError: If a required column is still missing afterwards, or if an
            explicit mapping names a source column that does not exist.
    """
    if isinstance(column_map, str) or column_map is None:
        column_map = parse_column_map(column_map)

    frame = frame.copy()
    explicit = dict(column_map)

    missing_sources = [src for src in explicit.values() if src not in frame.columns]
    if missing_sources:
        raise KeyError(
            f"--column-map refers to column(s) {missing_sources} that are not in "
            f"the CSV. It has: {list(frame.columns)}."
        )

    inferred = {k: v for k, v in infer_column_map(frame).items() if k not in explicit}
    renames = {**{src: canon for canon, src in inferred.items()},
               **{src: canon for canon, src in explicit.items()}}

    if renames:
        # Renaming onto a name the frame already uses for something else would
        # produce two columns with the same label, and every later .loc on it
        # returns a DataFrame instead of a Series. Drop the original first.
        collisions = [canon for canon in renames.values()
                      if canon in frame.columns and canon not in renames]
        if collisions:
            frame = frame.drop(columns=collisions)
        frame = frame.rename(columns=renames)

        if verbose:
            for canonical, source in sorted(explicit.items()):
                print(f"  column map: {source!r} -> {canonical!r} (explicit)")
            for canonical, source in sorted(inferred.items()):
                print(f"  column map: {source!r} -> {canonical!r} (inferred)")

    still_missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if still_missing:
        raise KeyError(
            f"Dataset is missing required column(s): {still_missing}. "
            f"The CSV has {list(frame.columns)}. Map them explicitly with "
            "--column-map, e.g. --column-map "
            f"'{still_missing[0]}=<your column name>'."
        )
    return frame


def read_dataset(path, column_map=None, verbose=True):
    """Read a CSV and normalise its columns. The only way in.

    Every entry point that loads a corpus goes through here. Reading with a
    bare `pd.read_csv` works right up until someone points the tool at a real
    dump, and then it fails deep inside cleaning with a message about a column
    the CSV never claimed to have. One reader means one behaviour, and no sixth
    call site drifting out of sync with the other five.

    Args:
        path: CSV to read.
        column_map (dict | str | None): Explicit mapping, applied before
            inference. See normalize_columns.
        verbose (bool): Print the renames that were applied.
    Returns:
        pd.DataFrame: with the canonical column names.
    """
    frame = pd.read_csv(path)
    return normalize_columns(frame, column_map, verbose=verbose)


def describe_schema(frame):
    """One-line-per-column summary of a normalised frame, for the run log."""
    lines = []
    for column in REQUIRED_COLUMNS + OPTIONAL_COLUMNS:
        if column not in frame.columns:
            lines.append(f"  {column:<14} (absent)")
            continue
        series = frame[column]
        non_null = int(series.notna().sum())
        detail = f"{non_null}/{len(frame)} non-null"
        if pd.api.types.is_numeric_dtype(series) and non_null:
            detail += f", range [{series.min():g}, {series.max():g}]"
        lines.append(f"  {column:<14} {detail}")
    return "\n".join(lines)
