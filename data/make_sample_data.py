"""Generate a synthetic, Reddit-shaped dataset so the pipeline runs out of the box.

The real dataset this project was built around is a Kaggle dump that has to be
downloaded by hand, which means a fresh clone cannot run anything. This module
fabricates a CSV with the same columns and a *known, planted* signal, so you can
exercise the full pipeline, verify the plumbing, and sanity-check that the
temporal branch is wired up.

THIS IS NOT REAL DATA. It is generated from templates. Nothing learned from it
says anything about ADHD, about Reddit, or about anyone. Use it to test the
code, then point --dataset at the real CSV for results that mean something.

The planted structure, so you know what a correct run should recover:
  * Engagement (score) rises with posting hour-of-day, peaking in the evening
    when more people are online, and drops in the small hours.
  * Engagement also rises when a post asks a direct question.
  * Late-night posts draw on a different phrase pool than daytime posts.
Text alone therefore gets you part of the way; adding the timestamp should get
you further. That gap is the point of the temporal ablation.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from training.config import DATASET_DIR

# Template fragments. Deliberately mundane -- these exist to give the tokenizer
# something with realistic structure, not to characterise anyone.
OPENERS_DAY = [
    "Started the morning with a long list",
    "Trying a new routine this week",
    "Had my checkup yesterday",
    "Finally cleared my inbox",
    "Back at work after a break",
    "Been using a timer for focus blocks",
]
OPENERS_NIGHT = [
    "Cant sleep again so here I am",
    "Its almost 3am and my brain will not stop",
    "Lying awake thinking about tomorrow",
    "Everyone is asleep and I am still going",
    "Another night of scrolling instead of sleeping",
    "Wide awake and hyperfocused on something pointless",
]
MIDDLES = [
    "and I keep losing track of what I was doing",
    "but the task switching is what gets me",
    "and the paperwork has been sitting there for weeks",
    "though the structure helps more than I expected",
    "and I forgot two appointments this month",
    "but writing things down has been working",
    "and I start five things before finishing one",
    "so I am trying to break it into smaller steps",
]
CLOSERS_STATEMENT = [
    "Just wanted to write it down somewhere.",
    "Posting mostly to get it out of my head.",
    "Anyway, that is where things are.",
    "Figured someone else might relate.",
]
CLOSERS_QUESTION = [
    "Has anyone found something that actually works?",
    "What do you all do about this?",
    "Is this something you deal with too?",
    "Any advice on where to start?",
]
TITLES = [
    "Routine question", "Focus struggles", "Small win today", "Need some advice",
    "Late night thoughts", "Trying something new", "Checking in", "Does this sound familiar",
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
    path = Path(path) if path else DATASET_DIR / "ADHD_sample.csv"
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
