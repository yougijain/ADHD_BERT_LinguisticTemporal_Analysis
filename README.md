# Text vs. Timing

**Does *when* a post goes up add anything over *what* it says?**

Everyone accepts that posting time affects engagement — it is folklore on every
platform with a "best time to post" blog article. Almost nobody measures it
against the text. This project does: same rows, same labels, same split, two
architectures, each run with and without the timestamp, and one table at the end
saying how much the clock was worth.

The target is binary: did a post land above typical engagement for its venue.
The comparison is a 2x2 grid.

|                  | text only | text + temporal |
|------------------|-----------|-----------------|
| **TF-IDF + LR**  | ablation  | full            |
| **BERT**         | ablation  | full            |

- **TF-IDF + logistic regression** — word and character n-grams, optionally
  stacked with the temporal features. Fast, interpretable, and a genuinely
  strong competitor on short text. It is the control, not a straw man.
- **BERT + temporal fusion** — a BERT encoder for the text; cyclical
  hour-of-day, day-of-week, month, weekend and late-night features (all
  **UTC**, see [Timezones](#timezones)) through a small MLP; the two
  concatenated before the classification head.

Running the same ablation on both architectures is the point. If the timestamp
helps a linear model and a transformer alike, the signal is in the data. If it
only helps one, what you are measuring is the fusion head.

## Status

Working end to end, offline. `python main.py --synthetic` trains, evaluates,
writes figures and a `results.json`, and reports accuracy against the
majority-class baseline. The test suite runs in a few seconds with no network.

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
| Error analysis (slices, calibration, model comparison) | Done |
| Test suite (193 tests, offline) | Done |
| Results on a real corpus | **Not run — see [Dataset](#dataset)** |

That last row is the honest one. Everything below the line marked *sample data*
was produced on generated text and is a check that the plumbing works, not a
finding.

## Install

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
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

The real thing, once you have a corpus (downloads `bert-base-uncased`):

```bash
python main.py --dataset datasets/posts.csv --epochs 3
```

Analysis on its own, no training:

```bash
python -m analysis.timestamp_analysis --dataset datasets/sample_posts.csv
python -m analysis.pattern_detection  --dataset datasets/sample_posts.csv
python -m analysis.token_stats        --dataset datasets/sample_posts.csv --offline
python -m analysis.error_analysis     --dataset datasets/sample_posts.csv
python -m analysis.error_analysis     --dataset datasets/sample_posts.csv --compare-feature-sets
```

Tests:

```bash
pytest tests/ -q
```

## What a run produces

```
outputs/
├── benchmark.json                          # the comparison grid
├── checkpoints/bert_temporal_model.pth     # best epoch by validation accuracy
├── figures/hourly_activity.png             # posts per hour + engagement rate overlay
├── figures/weekly_heatmap.png              # weekday x hour posting volume
├── figures/loss_curve.png                  # batch loss + moving average
├── figures/error_rate_by_hour.png          # where the model fails
├── figures/calibration.png                 # confidence vs observed accuracy
├── results.json                            # config, metrics, history, correlations
└── results_tfidf.json                      # baseline metrics and top features
```

## Dataset

The pipeline expects a CSV with at least `selftext`, `score`, and `created_utc`
(Unix epoch seconds; millisecond timestamps are detected and handled). `title`
is used when present.

**No real corpus ships with this repo**, and no results have been produced on
one yet. Until then, the sample generator stands in:

```bash
python -m data.make_sample_data --rows 2000 --out datasets/sample_posts.csv
```

That is a **testing utility**, not a data source. It writes forum-shaped
template text with a documented planted signal (evening engagement bump,
off-hours penalty, bonus for asking a question) plus the `[removed]`/`[deleted]`
rows a real dump is full of, so the cleaning step has something to remove. It is
fabricated text. It is not data about anyone, and nothing measured on it is a
result.

## Results on the sample data

> **Sample data.** Generated template text with a deliberately planted
> time-of-day signal. These numbers demonstrate that the pipeline works end to
> end and that the ablation is wired up correctly. They say nothing about real
> posts, and they are not the project's result.

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
+0.06 for the neural one. That the gap appears on both is the structure the
generator planted, recovered. On real data, whether it appears on both is the
question.

**The linear baseline wins by a wide margin.** Do not read that as "TF-IDF beats
BERT": the BERT row is a randomly initialised miniature model with no pretrained
weights, because the environment it ran in could not reach huggingface.co. It is
a plumbing check, not a competitor. Re-run without `--tiny-model` before drawing
any conclusion — `benchmark.py` prints this caveat itself when it detects the
tiny model.

The baseline's coefficients recover the planted structure directly: `hour_cos`
at -1.41 and `hour_sin` at +1.37 dominate every word feature, with
`is_late_night` at -0.48 — an evening bump and a late-night penalty, which is
exactly what the generator plants. (In the synthetic data the clock is
unambiguous because the generator defines it; on a real dump read these as UTC
bands, per [Timezones](#timezones).)

## Error analysis

An accuracy number says how often a model is right. It does not say whether the
mistakes are spread evenly or piled into one slice, and that difference decides
whether a result is usable.

```bash
python -m analysis.error_analysis --dataset datasets/sample_posts.csv
```

Reports:

- **Error rate by slice** — posting hour, weekday, text length, stylistic
  markers. Buckets under 15 samples are excluded from the ranking, so a 3-row
  bucket at 100% error cannot pose as the model's biggest weakness.
- **Error asymmetry** — mistakes running almost entirely one direction mean a
  threshold problem, not an accuracy problem. Different fix.
- **Calibration** — a reliability table and expected calibration error. Above
  ~0.1 the probabilities do not mean what they say, so they can rank but cannot
  be thresholded for a precision target. The report says so.
- **The most confident mistakes** — a model that is 95% sure and wrong has
  learned something false. These explain an error rate faster than any statistic.

On the sample data the baseline sits at 17.6% error with ECE 0.053, and its
most confident mistakes are all late-night posts that *did* well, called low
engagement at 0.95+. It learned the planted late-night penalty hard enough to
override everything else — which is the kind of thing only error analysis
surfaces.

### Does the ablation fix errors or just move them?

```bash
python -m analysis.error_analysis --dataset datasets/sample_posts.csv --compare-feature-sets
```

The benchmark table says the temporal features raise accuracy. It cannot say
whether they fix predictions or shuffle errors around — a model can gain overall
while getting worse where it matters. This compares the two runs example by
example:

| | Count |
|---|---|
| Both correct | 116 |
| Both wrong | 25 |
| Only text-only correct | 14 |
| Only text+temporal correct | **67** |

67 fixed against 14 broken, a net gain of 53. The features are adding
information, not reshuffling it — a stronger claim than the accuracy delta alone
supports.

`compare_predictions()` does the same for any two models, so TF-IDF and BERT can
be compared the same way once real BERT weights are available.

## Timezones

**Every temporal feature here is UTC, and that is a real limit on what they
mean.**

`created_utc` is all most public dumps give us. There is no per-author timezone,
so a poster's local clock time cannot be recovered. Someone in California
writing at 2am local shows up at 09:00–10:00 UTC and is *not* flagged
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
read it as *did this post land above typical engagement for this venue*.

`--label-strategy positive` reproduces a naive `score > 0` rule and is kept only
for comparison. Do not use it for results: on platforms where posts start at a
score of 1, it puts over 90% of any real dump in one class and the model learns
to answer "1" every time. That is why evaluation always prints the
majority-class baseline next to accuracy.

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
│   ├── pattern_detection.py       # stylistic markers, crossed with posting hour
│   ├── error_analysis.py          # error slices, calibration, model comparison
│   ├── timestamp_analysis.py      # temporal EDA and figures
│   └── token_stats.py             # token-length distribution, truncation rates
├── data/
│   ├── data_loader.py             # Dataset and train/val splitting
│   ├── inspect_dataset.py         # quick look at a raw CSV
│   ├── make_sample_data.py        # synthetic generator (testing utility)
│   └── preprocess.py              # cleaning, labelling, tokenization
├── models/
│   ├── attention_layer.py         # attention pooling over token states
│   ├── bert_temporal_model.py     # BERT + temporal fusion classifier
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

Then run the error analysis. Accuracy is one number; where a model fails is the
part that decides whether it is usable, and it is usually the more interesting
half of a write-up.

## Scope and limits

This predicts post engagement from text and a UTC timestamp. That is the whole
claim. It is not a diagnostic tool of any kind, it makes no inference about any
author, and it says nothing about any individual.

The temporal features carry the timezone caveat above, so they are not evidence
about anyone's sleep or circadian rhythm. The stylistic markers in
`analysis/pattern_detection.py` are hand-built keyword lists — crude proxies for
writing style, deliberately kept visible and editable rather than hidden behind
a model download, so you can audit exactly what is being counted.

Engagement is also not quality. A score is a measure of what an audience
rewarded at a particular hour on a particular platform, and a model that
predicts it is modelling that audience's behaviour, not the merit of the writing.
