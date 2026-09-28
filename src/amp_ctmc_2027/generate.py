from __future__ import annotations

import argparse
from pathlib import Path

from amp_ctmc_2027.config import AMPConfig
from amp_ctmc_2027.data.fasta_io import FastaRepository
from amp_ctmc_2027.pipeline.generation_pipeline import GenerationPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate broad-spectrum AMP challenge outputs")
    parser.add_argument("--config", type=Path, default=Path("configs/generate.json"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def generate_broad_spectrum() -> None:
    """Challenge entry point for broad-spectrum generation."""
    args = parse_args()
    config = AMPConfig.from_json_file(args.config)
    if args.checkpoint is not None:
        config.checkpoint_dir = args.checkpoint
    if args.output_dir is not None:
        config.generate_dir = args.output_dir
    if args.device is not None:
        config.device = args.device
    if args.seed is not None:
        config.seed = args.seed
    pipeline = GenerationPipeline(config=config, fasta_repo=FastaRepository(Path.cwd()))
    pipeline.run()


def main() -> None:
    generate_broad_spectrum()


if __name__ == "__main__":
    generate_broad_spectrum()
