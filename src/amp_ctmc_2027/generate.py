from __future__ import annotations

import argparse
from pathlib import Path

from amp_ctmc_2027.config import AMPConfig
from amp_ctmc_2027.data.fasta_io import FastaRepository
from amp_ctmc_2027.pipeline.generation_pipeline import GenerationPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate broad-spectrum AMP challenge outputs")
    parser.add_argument("--n-sequences", type=int, default=50000)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoint"))
    parser.add_argument("--known-amps-fasta", type=Path, default=Path("data/training/training.fasta"))
    parser.add_argument("--antibacterial-fasta", type=Path, default=Path("data/antibacterial.fasta"))
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    return parser.parse_args()


def generate_broad_spectrum() -> None:
    """Challenge entry point for broad-spectrum generation."""
    args = parse_args()
    config = AMPConfig(
        n_sequences=args.n_sequences,
        top_k=args.top_k,
        seed=args.seed,
        generation_batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint,
        training_fasta_path=args.known_amps_fasta,
        antibacterial_fasta_path=args.antibacterial_fasta,
        device=args.device,
    )
    pipeline = GenerationPipeline(config=config, fasta_repo=FastaRepository(Path.cwd()))
    pipeline.run()


def main() -> None:
    generate_broad_spectrum()


if __name__ == "__main__":
    generate_broad_spectrum()
