"""Generate a synthetic, forum-shaped corpus for tests and smoke runs.

This is a **testing utility**, not a data source. It fabricates a CSV with the
columns the pipeline expects and a *known, planted* signal, so the test suite
and a fresh clone can exercise every code path without a download. For results
that mean anything, fetch a real corpus -- see `data/fetch_dataset.py` and the
Dataset section of the README.

THIS IS NOT REAL DATA. It is assembled from templates. Nothing measured on it
says anything about any forum, any topic, or anyone.

The planted structure, so you know what a correct run should recover:
  * Engagement (score) rises with posting hour-of-day, peaking in the evening
    when more people are online, and drops in the small hours.
  * Engagement also rises when a post asks a direct question.
  * Off-hours posts draw on a different phrase pool than daytime posts.
Text alone therefore gets you part of the way; adding the timestamp should get
you further. That gap is exactly what the temporal ablation measures, which is
why the generator plants a signal in both channels rather than one.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from training.config import DATASET_DIR

# Template fragments: mundane, topic-neutral help-forum phrasing. They exist to
# give the tokenizer realistic structure, nothing more.
OPENERS_DAY = [
    "Ran into this while setting up a new project today",
    "Following the documented steps and got a different result",
    "Picked this up from a colleague and it works but I do not know why",
    "Spent the afternoon narrowing this down to one function",
    "Back on this after shipping the last change",
    "Reduced it to a minimal example before posting",
]
OPENERS_NIGHT = [
    "Still at this well past midnight and out of ideas",
    "Third attempt tonight and the build keeps failing",
    "Everyone else has logged off so posting here instead",
    "Been staring at the same stack trace for hours",
    "Deploying late and hit something I have never seen",
    "Cannot leave this alone until it makes sense",
]
MIDDLES = [
    "and the error only shows up on the second run",
    "but the logs stop right before the interesting part",
    "and rolling back the last change did not help",
    "though it works fine on a clean checkout",
    "and the same input gives two different outputs",
    "but pinning the version made it go away",
    "and I cannot reproduce it outside the test suite",
    "so I am trying to isolate which step actually fails",
]
CLOSERS_STATEMENT = [
    "Leaving this here in case it helps someone later.",
    "Writing it up mostly to get the details straight.",
    "Anyway, that is where I have got to so far.",
    "Posting the workaround I settled on.",
]
CLOSERS_QUESTION = [
    "Has anyone hit this and found a real fix?",
    "What would you check next?",
    "Is this expected behaviour or a bug?",
    "Any pointers on where to start looking?",
]
TITLES = [
    "Unexpected result from a documented call", "Build fails only on CI",
    "Small fix that took all day", "Need a second opinion on this trace",
    "Late night debugging notes", "Trying a different approach",
    "Following up on an earlier thread", "Is this the intended behaviour",
]


def _make_post(rng, is_night, asks_question):
    """Assemble one post body from the template pools."""
    opener = rng.choice(OPENERS_NIGHT if is_night else OPENERS_DAY)
    n_middles = int(rng.integers(1, 4))
    middles = " ".join(rng.choice(MIDDLES, size=n_middles, replace=False))
    closer = rng.choice(CLOSERS_QUESTION if asks_question else CLOSERS_STATEMENT)
    return f"{opener} {middles}. {closer}"


def generate_dataset(n_rows=1200, seed=42, start="2023-01-01", days=365,
                     placeholder_rate=0.06, short_rate=0.04):
    """Build the synthetic DataFrame.

    Args:
        n_rows (int): Number of posts to generate.
        seed (int): RNG seed.
        start (str): First possible post date.
        days (int): Span of dates to spread posts across.
        placeholder_rate (float): Share of rows whose body is [removed]/[deleted].
            Real dumps are full of these, and the cleaning step must drop them.
        short_rate (float): Share of rows that are too short to be usable.
    Returns:
        pd.DataFrame: Columns id, title, selftext, score, num_comments, created_utc.
    """
    rng = np.random.default_rng(seed)
    start_ts = int(pd.Timestamp(start).timestamp())
    span = days * 24 * 3600

    # Posting hours: bimodal, with a real late-night tail rather than a uniform
    # spread, because a flat hour distribution would make the temporal features
    # carry no information at all.
    hour_weights = np.array([
        4.0, 3.0, 2.5, 2.0, 1.5, 1.2,      # 00-05 late night
        1.5, 2.5, 4.0, 5.0, 5.5, 5.5,      # 06-11 morning
        6.0, 6.0, 5.5, 5.5, 6.0, 7.0,      # 12-17 afternoon
        8.0, 8.5, 8.0, 7.0, 6.0, 5.0,      # 18-23 evening
    ])
    hour_weights = hour_weights / hour_weights.sum()
    hours = rng.choice(24, size=n_rows, p=hour_weights)

    day_offsets = rng.integers(0, days, size=n_rows)
    minutes = rng.integers(0, 3600, size=n_rows)
    created_utc = start_ts + day_offsets * 86400 + hours * 3600 + minutes
    created_utc = np.clip(created_utc, start_ts, start_ts + span)

    is_night = (hours < 5)
    asks_question = rng.random(n_rows) < 0.45

    texts = [_make_post(rng, bool(n), bool(q)) for n, q in zip(is_night, asks_question)]
    titles = list(rng.choice(TITLES, size=n_rows))

    # Planted score signal: an evening bump, a late-night penalty, a bonus for
    # asking a question, plus heavy noise so the task stays non-trivial.
    hour_effect = 1.6 * np.sin(2 * np.pi * (hours - 3) / 24)
    latent = (
        1.0
        + hour_effect
        + 0.9 * asks_question.astype(float)
        - 0.7 * is_night.astype(float)
        + rng.normal(0.0, 1.0, size=n_rows)
    )
    scores = np.maximum(0, np.round(np.exp(latent)).astype(int))
    num_comments = np.maximum(0, (scores * rng.uniform(0.2, 0.8, n_rows)).astype(int))

    frame = pd.DataFrame({
        "id": [f"sy{i:06d}" for i in range(n_rows)],
        "title": titles,
        "selftext": texts,
        "score": scores,
        "num_comments": num_comments,
        "created_utc": created_utc,
    })

    # Sprinkle in the junk a real dump contains, so the cleaning step has
    # something to actually remove.
    n_placeholder = int(n_rows * placeholder_rate)
    n_short = int(n_rows * short_rate)
    junk_idx = rng.choice(n_rows, size=n_placeholder + n_short, replace=False)
    frame.loc[junk_idx[:n_placeholder], "selftext"] = rng.choice(
        ["[removed]", "[deleted]"], size=n_placeholder
    )
    frame.loc[junk_idx[n_placeholder:], "selftext"] = "ok thanks"

    return frame.sort_values("created_utc").reset_index(drop=True)


def write_dataset(path=None, **kwargs):
    """Generate and write the CSV, creating the directory if needed."""
    path = Path(path) if path else DATASET_DIR / "sample_posts.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = generate_dataset(**kwargs)
    frame.to_csv(path, index=False)
    print(f"Wrote {len(frame)} synthetic rows to {path}")
    print(f"  score: min {frame['score'].min()} median {frame['score'].median():.0f} "
          f"max {frame['score'].max()}")
    print("  REMINDER: this is generated template text, not real data.")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=None, help="Output CSV path.")
    parser.add_argument("--rows", type=int, default=1200, help="Number of posts.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    args = parser.parse_args()
    write_dataset(path=args.out, n_rows=args.rows, seed=args.seed)


if __name__ == "__main__":
    main()
