# ADHD Linguistic-Temporal Analysis

Binary classification over Reddit posts using **both** what was written and
**when** it was written. A BERT encoder handles the text; engineered timestamp
features (cyclical hour-of-day, day-of-week, weekend and late-night flags) run
through a small MLP; the two are concatenated before the classification head.

The point of the architecture is measurable: a `--no-temporal` flag gives you
the text-only ablation, so you can see what the timestamp is actually worth
rather than assuming it helps.

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
| Descriptive analysis + figures | Done |
| Test suite (130 tests, offline) | Done |
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

No dataset and no network needed — this generates sample data, trains a
miniature model, and prints real metrics:

```bash
python main.py --synthetic --tiny-model --epochs 8 --learning-rate 1e-3 \
    --split-strategy random --max-length 64
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
├── checkpoints/bert_adhd_model.pth   # best epoch by validation accuracy
├── figures/hourly_activity.png       # posts per hour + engagement rate overlay
├── figures/weekly_heatmap.png        # weekday x hour posting volume
├── figures/loss_curve.png            # batch loss + moving average
└── results.json                      # config, metrics, history, correlations
```

## The ablation

On the bundled synthetic data (tiny random model, 8 epochs, random split):

| Configuration | Accuracy | Lift over majority baseline |
|---|---|---|
| Text + temporal | 0.698 | +0.198 |
| Text only (`--no-temporal`) | 0.608 | +0.108 |

The temporal branch is carrying about 9 points. **These numbers are from
generated template data with a deliberately planted time-of-day signal.** They
demonstrate the plumbing works; they say nothing about real posts. Re-run the
comparison on a real dataset before quoting any of it.

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
│   └── model_utils.py             # seeding, devices, checkpoints
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

The attention weights (`predict(..., return_attention=True)`) give per-token
relevance, which is what makes the linguistic side inspectable rather than just
a number.

## Scope and limits

This predicts post engagement from text and timestamp. It is not a diagnostic
tool, it does not detect ADHD, and it says nothing about any individual. The
linguistic markers in `analysis/pattern_detection.py` are hand-built keyword
lists — crude proxies for writing style, deliberately kept visible and editable
rather than hidden behind a model download, so you can audit exactly what is
being counted.
