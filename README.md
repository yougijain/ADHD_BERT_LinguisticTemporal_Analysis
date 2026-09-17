# ADHD Linguistic-Temporal Analysis

Binary classification over Reddit posts using **both** what was written and
**when** it was written, benchmarked against a linear baseline.

Two models over identical data and splits:

- **TF-IDF + logistic regression** — word and character n-grams, optionally
  stacked with the temporal features. Fast, interpretable, and a genuinely
  strong competitor on short text.
- **BERT + temporal fusion** — a BERT encoder for the text; cyclical
  hour-of-day, day-of-week, weekend and late-night features (all **UTC**, see
  [Timezones](#timezones)) through a small MLP; the two concatenated before the
  classification head.

Both claims the project makes are measurable rather than assumed. `--no-temporal`
gives the text-only ablation on either architecture, and `benchmark.py` runs the
full 2x2 grid so you can see whether the transformer earns its cost and whether
the timestamps contribute anything.

## Status

Working end to end. `python main.py --synthetic` trains, evaluates, writes
figures and a `results.json`, and reports accuracy against the majority-class
baseline. The test suite runs offline in a few seconds.

| Component | State |
|---|---|
| Data cleaning and labelling | Done |
| Temporal feature engineering | Done |
| BERT + temporal model, attention pooling | Done |
| Training loop (warmup, clipping, AMP, checkpoint selection) | Done |
| Evaluation with baseline comparison | Done |
| TF-IDF + logistic regression baseline | Done |
| Benchmark grid (model x feature set) | Done |
| Descriptive analysis + figures | Done |
| Test suite (153 tests, offline) | Done |
| Results on the real Kaggle dataset | Not run — see [Dataset](#dataset) |

## Install

```bash
python -m venv adhd_env && source adhd_env/bin/activate   # Windows: adhd_env\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` installs torch from PyPI, which on Linux pulls the CUDA
build (~2.5GB). For a much smaller CPU-only install, do torch first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Swap `/cpu` for `/cu124` (or whatever matches your driver) to pick a CUDA build.

## Quick start

The baseline needs no network and no downloads — it generates sample data,
fits, and prints metrics plus the words driving each class:

```bash
python main.py --synthetic --model tfidf --split-strategy random
```

The neural path, on a miniature model so nothing is downloaded:

```bash
python main.py --synthetic --tiny-model --epochs 8 --learning-rate 1e-3 \
    --split-strategy random --max-length 64
```

Both, side by side:

```bash
python benchmark.py --synthetic --tiny-model --epochs 6 --learning-rate 1e-3 \
    --max-length 64 --batch-size 32
python benchmark.py --synthetic --skip-bert     # linear half only, seconds
```

The real thing, once you have a dataset (downloads `bert-base-uncased`):

```bash
python main.py --dataset datasets/ADHD.csv --epochs 3
```

Analysis on its own, no training:

```bash
python -m analysis.timestamp_analysis --dataset datasets/ADHD_sample.csv
python -m analysis.pattern_detection  --dataset datasets/ADHD_sample.csv
python -m analysis.token_stats        --dataset datasets/ADHD_sample.csv --offline
```

Tests:

```bash
pytest tests/ -q
```

## What a run produces

```
outputs/
├── benchmark.json                    # the comparison grid
├── checkpoints/bert_adhd_model.pth   # best epoch by validation accuracy
├── figures/hourly_activity.png       # posts per hour + engagement rate overlay
├── figures/weekly_heatmap.png        # weekday x hour posting volume
├── figures/loss_curve.png            # batch loss + moving average
├── results.json                      # config, metrics, history, correlations
└── results_tfidf.json                # baseline metrics and top features
```

## Results on the sample data

Full grid, `benchmark.py --synthetic --tiny-model --epochs 6 --learning-rate 1e-3`,
majority-class baseline 0.500:

| Model | Features | Accuracy | Macro F1 | Lift |
|---|---|---|---|---|
| TF-IDF | text only | 0.5856 | 0.5850 | +0.0856 |
| TF-IDF | text + temporal | **0.8243** | 0.8239 | +0.3243 |
| BERT (tiny, random) | text only | 0.6081 | 0.6052 | +0.1081 |
| BERT (tiny, random) | text + temporal | 0.6667 | 0.6657 | +0.1667 |

Two things to read off this, and one trap:

**Temporal features help in both architectures** — +0.24 for the linear model,
+0.06 for the neural one. That is the project's central claim, and it holds on
both.

**The linear baseline wins by a wide margin.** Do not read that as "TF-IDF beats
BERT": the BERT row is a randomly initialised miniature model with no pretrained
weights, because this environment could not reach huggingface.co. It is a
plumbing check, not a competitor. Re-run without `--tiny-model` before drawing
any conclusion — `benchmark.py` prints this caveat itself when it detects the
tiny model.

**All of it is synthetic.** Generated template data with a deliberately planted
time-of-day signal. The numbers demonstrate the pipeline works; they say nothing
about real posts.

The baseline's coefficients recover the planted structure directly: `hour_cos`
at -1.41 and `hour_sin` at +1.37 dominate every word feature, with
`is_late_night` at -0.48 — an evening bump and a late-night penalty, which is
exactly what the generator plants. (In the synthetic data the clock is
unambiguous because the generator defines it; on a real dump read these as UTC
bands, per [Timezones](#timezones).)

## Dataset

The pipeline expects a CSV with at least `selftext`, `score`, and `created_utc`
(Unix epoch seconds; millisecond timestamps are detected and handled). `title`
is used when present.

**No real dataset ships with this repo.** Earlier README revisions pointed at a
`kaggle.com/your-dataset-link` placeholder and a `download_dataset.py` that was
never committed, so a fresh clone could not run anything. Until a real source is
pinned here, use `--synthetic`:

```bash
python -m data.make_sample_data --rows 2000 --out datasets/ADHD_sample.csv
```

That generates Reddit-shaped template text with a documented planted signal
(evening engagement bump, late-night penalty, bonus for asking a question) plus
the `[removed]`/`[deleted]` rows a real dump is full of, so the cleaning step
has something to remove. It is fabricated text. It is not data about anyone.

## Timezones

**Every temporal feature here is UTC, and that is a real limit on what they
mean.**

`created_utc` is all Reddit gives us. There is no per-author timezone in the
dataset, so a poster's local clock time cannot be recovered. Someone in
California writing at 2am local shows up at 09:00–10:00 UTC and is *not* flagged
late-night; someone in Berlin writing at 2am local is.

So `is_late_night` does not mean "written in the small hours". It means "written
in the 00:00–05:00 UTC band", which selects for a mix of local times that
depends on where the posters live. The feature is still predictive — UTC hour
correlates with both local hour and how many people are awake to vote — but the
circadian interpretation is not supported by this data. Don't describe it as a
sleep or chronotype measure.

Fixing it properly means inferring each author's timezone from the distribution
of their own posting times across a long history. That is real work and is not
implemented. Until it is, call these features UTC posting hour and say so in any
write-up.

## Labels

Post `score` is turned into a binary target. The default is `--label-strategy
median`, which picks the score threshold splitting the data closest to 50/50 —
read it as *did this post land above typical engagement for this subreddit*.

`--label-strategy positive` reproduces the original `score > 0` rule and is kept
only for comparison. Do not use it for results: Reddit posts start at a score of
1, so it puts over 90% of any real dump in one class and the model learns to
answer "1" every time. That is why evaluation always prints the majority-class
baseline next to accuracy.

## Splits

`--split-strategy temporal` (default) sorts chronologically and validates on the
most recent posts — the honest setup when the features include the timestamp,
because it does not let the model see the future. `--split-strategy random`
shuffles, which is the right choice when you only care about linguistic content
and want an i.i.d. split. Class balance for both halves is printed, with a
warning if either lands single-class.

## Running without network access

`--tiny-model` and `--offline-tokenizer` train a WordPiece tokenizer on your own
corpus and size a miniature randomly-initialised BERT to match, so nothing is
downloaded. This is for exercising the pipeline and for CI — the model carries
no pretrained knowledge, so its accuracy is not a result.

## Useful flags

| Flag | Effect |
|---|---|
| `--model` | `bert` (default) or `tfidf` |
| `--synthetic` | Generate and use sample data |
| `--tiny-model` | Random miniature BERT, no download |
| `--offline-tokenizer` | Corpus-trained tokenizer instead of downloading one |
| `--no-temporal` | Text-only ablation |
| `--no-attention-pooling` | Mean pooling instead of learned attention |
| `--freeze-bert` | Train only the head; much faster on CPU |
| `--max-rows N` | Cap rows for a quick run |
| `--label-strategy` | `median` (default), `threshold`, `positive` |
| `--split-strategy` | `temporal` (default) or `random` |
| `--device` | `auto`, `cpu`, `cuda` |

`python main.py --help` lists all of them.

## Layout

```
.
├── main.py                        # CLI pipeline: load -> analyse -> train -> evaluate
├── benchmark.py                   # TF-IDF vs BERT x temporal grid, as one table
├── analysis/
│   ├── pattern_detection.py       # linguistic markers, crossed with posting hour
│   ├── timestamp_analysis.py      # temporal EDA and figures
│   └── token_stats.py             # token-length distribution, truncation rates
├── data/
│   ├── data_loader.py             # Dataset and train/val splitting
│   ├── inspect_dataset.py         # quick look at a raw CSV
│   ├── make_sample_data.py        # synthetic dataset generator
│   └── preprocess.py              # cleaning, labelling, tokenization
├── models/
│   ├── attention_layer.py         # attention pooling over token states
│   ├── bert_adhd_model.py         # BERT + temporal fusion classifier
│   ├── model_utils.py             # seeding, devices, checkpoints
│   └── tfidf_baseline.py          # TF-IDF + logistic regression baseline
├── training/
│   ├── config.py                  # every tunable, in one dataclass
│   ├── evaluate.py                # metrics with baseline comparison
│   └── train.py                   # training loop
├── utils/
│   ├── loss_utils.py              # loss smoothing
│   └── time_utils.py              # timestamp parsing, temporal features
└── tests/                         # offline test suite
```

## Interpreting results

Accuracy on its own means little on an imbalanced split, so every evaluation
prints the majority-class baseline and the lift over it. **Lift at or below zero
means the model learned nothing**, whatever the accuracy says — the run tells
you so explicitly.

Two ways to inspect what a model keyed on. The neural path's attention weights
(`predict(..., return_attention=True)`) give per-token relevance. The baseline's
`top_features()` gives signed per-word coefficients, which is considerably more
legible — literal words with weights, rather than a distribution over
wordpieces. If the top features look like artefacts, the label is leaking.

Run the baseline first. It takes seconds, it needs no GPU and no downloads, and
if it already gets most of the available accuracy then the transformer is
carrying very little and the honest write-up says so.

## Scope and limits

This predicts post engagement from text and UTC timestamp. It is not a
diagnostic tool, it does not detect ADHD, and it says nothing about any
individual. The temporal features carry the timezone caveat above, so they are
not evidence about anyone's sleep. The
linguistic markers in `analysis/pattern_detection.py` are hand-built keyword
lists — crude proxies for writing style, deliberately kept visible and editable
rather than hidden behind a model download, so you can audit exactly what is
being counted.
