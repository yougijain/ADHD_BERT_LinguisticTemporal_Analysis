"""Central configuration for the text-vs-timing pipeline.

Every tunable lives here so that `main.py`, the training loop, and the analysis
scripts all agree on paths, hyperparameters, and feature definitions.
"""

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List

# Repo root, resolved from this file so the paths work no matter where you cd to.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "datasets"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
FIGURE_DIR = OUTPUT_DIR / "figures"

# Columns the pipeline expects in the raw CSV.
TEXT_COLUMN = "selftext"
TITLE_COLUMN = "title"
SCORE_COLUMN = "score"
TIMESTAMP_COLUMN = "created_utc"
REQUIRED_COLUMNS = (TEXT_COLUMN, SCORE_COLUMN, TIMESTAMP_COLUMN)

# Temporal features fed into the model alongside BERT's text representation.
# Order matters: the model's temporal branch is sized from this list.
TEMPORAL_FEATURES: List[str] = [
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "is_late_night",
]

# Forum dumps use these placeholders for posts whose body was taken down. They
# are not text, and training on them teaches the model nothing.
PLACEHOLDER_BODIES = ("[removed]", "[deleted]", "[removed by reddit]", "")


@dataclass
class Config:
    """Runtime configuration for a single train/evaluate run."""

    # Data
    dataset_path: Path = DATASET_DIR / "posts.csv"
    label_strategy: str = "median"  # median | positive | threshold
    label_threshold: int = 1  # only used when label_strategy == "threshold"
    min_tokens: int = 5  # drop posts with fewer whitespace tokens than this
    # "selftext=body,created_utc=creation_date" for a CSV that does not already
    # use the canonical names. Empty means infer only; see data/schema.py.
    column_map: str = ""
    max_rows: int = 0  # 0 means "use everything"; >0 subsamples for quick runs

    # Tokenization
    model_name: str = "bert-base-uncased"
    # Encoder for the frozen-embedding baseline. Never fine-tuned, so it is a
    # separate knob from model_name -- the point of that row is to hold
    # task-specific training fixed while varying the representation.
    encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    # Model for the zero-shot LLM row. Scaffolded, opt-in, and billed per row
    # of the validation split -- see models/llm_baseline.py.
    llm_model: str = "claude-opus-5"
    max_length: int = 256
    tokenize_batch_size: int = 512

    # Model
    num_labels: int = 2
    dropout: float = 0.1
    temporal_hidden_size: int = 32
    use_temporal_features: bool = True
    use_attention_pooling: bool = True
    freeze_bert: bool = False

    # Training
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    val_split: float = 0.2
    split_strategy: str = "temporal"  # temporal | random
    seed: int = 42
    num_workers: int = 0

    # Runtime
    device: str = "auto"  # auto | cpu | cuda
    use_amp: bool = True  # only takes effect on CUDA
    checkpoint_dir: Path = CHECKPOINT_DIR
    figure_dir: Path = FIGURE_DIR
    save_every_epoch: bool = False
    log_every: int = 10

    # Derived / non-init
    temporal_features: List[str] = field(default_factory=lambda: list(TEMPORAL_FEATURES))

    def __post_init__(self):
        self.dataset_path = Path(self.dataset_path)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.figure_dir = Path(self.figure_dir)

        if self.label_strategy not in ("median", "positive", "threshold"):
            raise ValueError(
                f"label_strategy must be median|positive|threshold, got {self.label_strategy!r}"
            )
        if self.split_strategy not in ("temporal", "random"):
            raise ValueError(
                f"split_strategy must be temporal|random, got {self.split_strategy!r}"
            )
        if not 0.0 < self.val_split < 1.0:
            raise ValueError(f"val_split must be in (0, 1), got {self.val_split}")

    @property
    def num_temporal_features(self) -> int:
        return len(self.temporal_features) if self.use_temporal_features else 0

    def ensure_dirs(self) -> None:
        """Create the output directories this run will write to."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.figure_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(self).items()}
