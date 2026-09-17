"""End-to-end pipeline: load, clean, analyse, train, evaluate.

Examples:
    # Train on the bundled synthetic data (generates it if missing).
    python main.py --synthetic --epochs 2

    # Train on a real dump.
    python main.py --dataset datasets/ADHD.csv --epochs 3

    # Text-only ablation, to measure what the temporal branch is worth.
    python main.py --synthetic --no-temporal

    # Quick smoke run on a tiny random model -- no 440MB download.
    python main.py --synthetic --tiny-model --epochs 1 --max-rows 200
"""

import argparse
import os
import sys

import pandas as pd

# Silence the HF symlink warning before transformers is imported, otherwise the
# setting arrives too late to have any effect.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from analysis.pattern_detection import (  # noqa: E402
    add_marker_columns,
    marker_label_correlations,
)
from analysis.timestamp_analysis import (  # noqa: E402
    plot_hourly_activity,
    plot_loss_curve,
    plot_weekly_heatmap,
    summarize_temporal,
)
from data.data_loader import build_dataloaders  # noqa: E402
from data.preprocess import (  # noqa: E402
    batch_tokenize,
    build_labels,
    build_local_tokenizer,
    clean_dataset,
)
from models.bert_adhd_model import BertTemporalClassifier  # noqa: E402
from models.model_utils import (  # noqa: E402
    describe_model,
    resolve_device,
    save_metrics,
    set_seed,
)
from training.config import Config, TIMESTAMP_COLUMN  # noqa: E402
from training.evaluate import evaluate_model  # noqa: E402
from training.train import build_optimizer, train_model  # noqa: E402
from utils.time_utils import add_temporal_features, temporal_feature_matrix  # noqa: E402


def load_and_prepare_data(config, run_analysis=True, offline_tokenizer=False):
    """Load, clean, featurise, tokenize, and split the dataset.

    Args:
        config (Config): Run configuration.
        run_analysis (bool): Produce the descriptive analysis and figures.
        offline_tokenizer (bool): Train a WordPiece tokenizer on this corpus
            instead of downloading one. Lets the pipeline run with no network.
    Returns:
        tuple: (train_loader, val_loader, info_dict)
    """
    print(f"Loading dataset from {config.dataset_path}...")
    if not config.dataset_path.exists():
        raise FileNotFoundError(
            f"No dataset at {config.dataset_path}. Either point --dataset at a "
            "real CSV, or pass --synthetic to generate a sample one."
        )
    data = pd.read_csv(config.dataset_path)
    print(f"  {len(data)} raw rows, columns: {list(data.columns)}")

    data = clean_dataset(data, min_tokens=config.min_tokens)

    print("Deriving temporal features...")
    data = add_temporal_features(data, TIMESTAMP_COLUMN)

    # Sort chronologically so a temporal split really is train-on-past,
    # validate-on-future. The old code split on whatever order the CSV happened
    # to be in and called it a split.
    if config.split_strategy == "temporal":
        data = data.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)

    if config.max_rows and config.max_rows < len(data):
        print(f"  Subsampling to {config.max_rows} rows (--max-rows).")
        data = data.head(config.max_rows).reset_index(drop=True)

    labels, label_summary = build_labels(
        data, strategy=config.label_strategy, threshold=config.label_threshold
    )
    print(f"  Labels: {label_summary}")

    info = {"label_summary": label_summary, "num_rows": int(len(data))}

    if run_analysis:
        print("\nRunning descriptive analysis...")
        config.ensure_dirs()
        temporal_summary = summarize_temporal(data, labels)
        for key, value in temporal_summary.items():
            print(f"  {key}: {value}")
        info["temporal_summary"] = temporal_summary

        with_markers = add_marker_columns(data)
        correlations = marker_label_correlations(with_markers, labels)
        print("\n  Top linguistic markers by label correlation:")
        for name, value in correlations.head(5).items():
            print(f"    {name:<26} {value:+.4f}")
        info["marker_correlations"] = correlations.round(6).to_dict()

        figures = [
            plot_hourly_activity(data, config.figure_dir, labels),
            plot_weekly_heatmap(data, config.figure_dir),
        ]
        info["figures"] = [str(p) for p in figures]
        print("  Figures: " + ", ".join(str(p.name) for p in figures))

    print("\nTokenizing...")
    tokenizer = None
    if offline_tokenizer:
        print("  Training a WordPiece tokenizer on this corpus (offline mode).")
        tokenizer = build_local_tokenizer(data["clean_text"].tolist())
        info["tokenizer"] = f"local-wordpiece (vocab {len(tokenizer)})"
        print(f"  Vocabulary size: {len(tokenizer)}")
    encodings = batch_tokenize(
        data["clean_text"].tolist(),
        model_name=config.model_name,
        batch_size=config.tokenize_batch_size,
        max_length=config.max_length,
        tokenizer=tokenizer,
    )
    info["vocab_size"] = int(len(tokenizer)) if tokenizer is not None else None

    temporal_matrix = None
    if config.use_temporal_features:
        temporal_matrix = temporal_feature_matrix(data, config.temporal_features)
        print(f"  Temporal feature matrix: {temporal_matrix.shape}")

    print("\nBuilding dataloaders...")
    train_loader, val_loader = build_dataloaders(
        encodings,
        labels,
        temporal_features=temporal_matrix,
        batch_size=config.batch_size,
        val_split=config.val_split,
        strategy=config.split_strategy,
        seed=config.seed,
        num_workers=config.num_workers,
    )
    return train_loader, val_loader, info


def build_model(config, tiny=False, vocab_size=None):
    """Construct the classifier, optionally as the tiny test model.

    `vocab_size` must match the tokenizer actually in use -- the offline path
    trains its own vocabulary, which is far smaller than BERT's 30522.
    """
    n_features = config.num_temporal_features
    if tiny:
        print("Building a randomly initialised tiny BERT (smoke-test mode).")
        return BertTemporalClassifier.tiny_for_testing(
            num_temporal_features=n_features,
            num_labels=config.num_labels,
            use_attention_pooling=config.use_attention_pooling,
            vocab_size=vocab_size or 2048,
            max_position_embeddings=max(config.max_length, 64),
        )
    print(f"Loading encoder '{config.model_name}'...")
    return BertTemporalClassifier.from_config(config)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="ADHD linguistic-temporal classification pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    data_group = parser.add_argument_group("data")
    data_group.add_argument("--dataset", default=None, help="Path to the CSV dataset.")
    data_group.add_argument("--synthetic", action="store_true",
                            help="Use (and generate if missing) the synthetic sample dataset.")
    data_group.add_argument("--max-rows", type=int, default=0,
                            help="Cap the number of rows used. 0 uses everything.")
    data_group.add_argument("--label-strategy", default="median",
                            choices=["median", "positive", "threshold"],
                            help="How to turn score into a binary label.")
    data_group.add_argument("--split-strategy", default="temporal",
                            choices=["temporal", "random"],
                            help="Chronological or shuffled train/val split.")

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--model-name", default="bert-base-uncased")
    model_group.add_argument("--max-length", type=int, default=256)
    model_group.add_argument("--no-temporal", action="store_true",
                             help="Text-only ablation: drop the temporal branch.")
    model_group.add_argument("--no-attention-pooling", action="store_true",
                             help="Use mean pooling instead of learned attention.")
    model_group.add_argument("--freeze-bert", action="store_true",
                             help="Train only the head. Much faster on CPU.")
    model_group.add_argument("--tiny-model", action="store_true",
                             help="Random miniature BERT for smoke tests. No download.")
    model_group.add_argument("--offline-tokenizer", action="store_true",
                             help="Train a WordPiece tokenizer on this corpus instead "
                                  "of downloading one. Implied by --tiny-model.")

    train_group = parser.add_argument_group("training")
    train_group.add_argument("--epochs", type=int, default=3)
    train_group.add_argument("--batch-size", type=int, default=16)
    train_group.add_argument("--learning-rate", type=float, default=2e-5)
    train_group.add_argument("--seed", type=int, default=42)
    train_group.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    train_group.add_argument("--no-amp", action="store_true",
                             help="Disable mixed precision even on CUDA.")
    train_group.add_argument("--skip-analysis", action="store_true",
                             help="Skip the descriptive analysis and figures.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    dataset_path = args.dataset
    if args.synthetic or dataset_path is None:
        from training.config import DATASET_DIR
        sample_path = DATASET_DIR / "ADHD_sample.csv"
        if dataset_path is None and not args.synthetic and not sample_path.exists():
            print(
                "No --dataset given. Falling back to synthetic sample data.\n"
                "Pass --dataset path/to/ADHD.csv for results that mean something.\n"
            )
        if not sample_path.exists():
            from data.make_sample_data import write_dataset
            print("Generating synthetic sample dataset...")
            write_dataset(sample_path)
        dataset_path = sample_path

    config = Config(
        dataset_path=dataset_path,
        label_strategy=args.label_strategy,
        split_strategy=args.split_strategy,
        max_rows=args.max_rows,
        model_name=args.model_name,
        max_length=args.max_length,
        use_temporal_features=not args.no_temporal,
        use_attention_pooling=not args.no_attention_pooling,
        freeze_bert=args.freeze_bert,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=args.device,
        use_amp=not args.no_amp,
    )
    config.ensure_dirs()
    set_seed(config.seed)

    if config.use_temporal_features:
        print(f"Temporal branch: ON ({config.num_temporal_features} features)")
    else:
        print("Temporal branch: OFF (text-only ablation)")

    offline_tokenizer = args.offline_tokenizer or args.tiny_model
    train_loader, val_loader, info = load_and_prepare_data(
        config,
        run_analysis=not args.skip_analysis,
        offline_tokenizer=offline_tokenizer,
    )

    model = build_model(config, tiny=args.tiny_model, vocab_size=info.get("vocab_size"))
    print(describe_model(model))

    device = resolve_device(config.device)
    optimizer = build_optimizer(model, config.learning_rate, config.weight_decay)

    checkpoint_path = config.checkpoint_dir / "bert_adhd_model.pth"
    history = train_model(
        train_loader,
        model,
        optimizer=optimizer,
        epochs=config.epochs,
        save_path=checkpoint_path,
        val_loader=val_loader,
        device=device,
        use_amp=config.use_amp,
        max_grad_norm=config.max_grad_norm,
        log_every=config.log_every,
        config=config,
        save_every_epoch=config.save_every_epoch,
    )

    if history["batch_losses"]:
        window = min(25, max(len(history["batch_losses"]) // 4, 1))
        loss_figure = plot_loss_curve(history["batch_losses"], config.figure_dir, window)
        print(f"Loss curve: {loss_figure}")

    print("\nFinal evaluation")
    metrics = evaluate_model(val_loader, model, device=device, verbose=True)

    results = {
        "config": config.to_dict(),
        "data": info,
        "history": history,
        "final_metrics": metrics,
    }
    metrics_path = save_metrics(results, config.checkpoint_dir.parent / "results.json")
    print(f"\nResults written to {metrics_path}")

    # The number that actually matters: beating the majority-class baseline.
    lift = metrics.get("lift_over_baseline", 0.0)
    if lift <= 0:
        print(
            f"\nNOTE: accuracy {metrics['accuracy']:.4f} is at or below the "
            f"majority-class baseline {metrics['majority_baseline']:.4f}. The "
            "model has not learned anything useful -- train longer, unfreeze the "
            "encoder, or check the label strategy."
        )
    else:
        print(f"\nBeat the majority-class baseline by {lift:+.4f} accuracy.")

    return results


if __name__ == "__main__":
    sys.exit(0 if main() else 0)
