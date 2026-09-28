#!/usr/bin/env python3
"""Property-model entry point reserved for prepared labeled Parquet input."""

import argparse
import json
from pathlib import Path

from amp_ctmc_2027.pipeline.property_training_pipeline import PropertyTrainingPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/train_property.json")
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/property"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    pipeline = PropertyTrainingPipeline(
        args.data,
        args.output,
        config["model_name"],
        config.get("revision"),
        epochs=int(config.get("epochs", 10)),
        batch_size=int(config.get("batch_size", 16)),
        max_length=int(config.get("max_length", 64)),
        seed=int(config.get("seed", 42)),
        unfreeze_last_n_layers=int(config.get("unfreeze_last_n_layers", 6)),
        gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 1)),
    )
    pipeline.run()


if __name__ == "__main__":
    main()
