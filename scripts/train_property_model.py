#!/usr/bin/env python3
"""Property-model entry point reserved for prepared labeled Parquet input."""

import argparse
import json
from pathlib import Path

import torch

from amp_ctmc_2027.pipeline.property_training_pipeline import PropertyTrainingPipeline


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=root / "configs/train_property.json"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=root / "artifacts/property")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    pipeline = PropertyTrainingPipeline(
        args.data if args.data.is_absolute() else root / args.data,
        args.output,
        config["model_name"],
        config.get("revision"),
        epochs=int(config.get("epochs", 10)),
        batch_size=int(config.get("batch_size", 16)),
        max_length=int(config.get("max_length", 64)),
        seed=int(config.get("seed", 42)),
        unfreeze_last_n_layers=int(config.get("unfreeze_last_n_layers", 6)),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 1)),
        repository_root=root,
        ranking_strains=config.get("ranking_strains"),
        strain_groups=config.get("strain_groups"),
        sequence_clustering_threshold=float(
            config.get("sequence_clustering_threshold", 0.4)
        ),
        activity_threshold_log2_mic=float(config["activity_threshold_log2_mic"]),
        activity_temperature=float(config["activity_temperature"]),
        broad_spectrum_aggregation=config["broad_spectrum_aggregation"],
        early_stopping=bool(config.get("early_stopping", True)),
        early_stopping_patience=int(config.get("early_stopping_patience", 2)),
        mixed_precision=bool(config.get("mixed_precision", True)),
        pretrained_model_id=config.get("pretrained_model_id"),
        pretrained_revision=config.get("pretrained_revision"),
    )
    try:
        pipeline.run()
    except torch.cuda.OutOfMemoryError as exc:
        if int(config.get("batch_size", 16)) >= 64:
            raise RuntimeError(
                "CUDA ran out of memory at property batch size 64; edit/copy the config and retry with batch_size 32."
            ) from exc
        raise


if __name__ == "__main__":
    main()
