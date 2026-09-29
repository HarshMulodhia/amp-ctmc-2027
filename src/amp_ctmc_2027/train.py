from __future__ import annotations

import argparse
from pathlib import Path

from amp_ctmc_2027.config import AMPConfig
from amp_ctmc_2027.data.fasta_io import FastaRepository
from amp_ctmc_2027.pipeline.training_pipeline import TrainingPipeline


def main() -> None:
    """Train with a checked-in production config or an explicitly selected config."""
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--config", type=Path, default=root / "configs/train_ctmc.json")
    parser.add_argument(
        "--resume", type=Path, help="Resume model and optimizer from training_state.pt"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use a small non-production overfit subset",
    )
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = AMPConfig.from_json_file(config_path)
    if not config.training_fasta_path.is_absolute():
        config.training_fasta_path = root / config.training_fasta_path
    if config.background_fasta_path and not config.background_fasta_path.is_absolute():
        config.background_fasta_path = root / config.background_fasta_path
    if (
        config.cluster_split_manifest_path
        and not config.cluster_split_manifest_path.is_absolute()
    ):
        config.cluster_split_manifest_path = root / config.cluster_split_manifest_path
    if not config.checkpoint_dir.is_absolute():
        config.checkpoint_dir = root / config.checkpoint_dir
    for field in (
        "condition_table_path",
        "condition_manifest_path",
        "property_metadata_path",
    ):
        value = getattr(config, field)
        if value is not None and not value.is_absolute():
            setattr(config, field, root / value)
    if args.resume:
        config.resume_from = args.resume
    if args.smoke_test:
        config.n_epochs = min(config.n_epochs, 2)
        config.batch_size = min(config.batch_size, 64)
        config.debug_overfit_samples = 64
    pipeline = TrainingPipeline(config=config, fasta_repo=FastaRepository(root))
    pipeline.run()


if __name__ == "__main__":
    main()
