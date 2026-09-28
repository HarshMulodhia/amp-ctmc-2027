from __future__ import annotations

import argparse
from pathlib import Path

from amp_ctmc_2027.config import AMPConfig
from amp_ctmc_2027.data.fasta_io import FastaRepository
from amp_ctmc_2027.pipeline.training_pipeline import TrainingPipeline


def main() -> None:
    """Train with a checked-in production config or an explicitly selected config."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train_ctmc.json"))
    parser.add_argument(
        "--resume", type=Path, help="Resume model and optimizer from training_state.pt"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use a small non-production overfit subset",
    )
    args = parser.parse_args()
    config = AMPConfig.from_json_file(args.config)
    if args.resume:
        config.resume_from = args.resume
    if args.smoke_test:
        config.n_epochs = min(config.n_epochs, 2)
        config.batch_size = min(config.batch_size, 64)
        config.debug_overfit_samples = 64
    pipeline = TrainingPipeline(config=config, fasta_repo=FastaRepository(Path.cwd()))
    pipeline.run()


if __name__ == "__main__":
    main()
