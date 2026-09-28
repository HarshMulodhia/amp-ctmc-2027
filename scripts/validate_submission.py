#!/usr/bin/env python3
"""Validate final files using a pinned organizer identity function."""
import argparse
from pathlib import Path

from amp_ctmc_2027.compliance import load_official_identity, validate_submission
from amp_ctmc_2027.data.fasta_io import FastaRepository


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, default=Path("generate_broad_spectrum/library.fasta"))
    parser.add_argument("--top", type=Path, default=Path("generate_broad_spectrum/top.fasta"))
    parser.add_argument("--references", type=Path, default=Path("data/antibacterial.fasta"))
    parser.add_argument("--identity-function", required=True, help="Python import spec module:function from pinned official validator")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--output", type=Path, default=Path("generate_broad_spectrum/compliance_report.json"))
    args = parser.parse_args()
    repo = FastaRepository(args.library.parent)
    report = validate_submission(repo.read_sequences(args.library), repo.read_sequences(args.top),
                                 FastaRepository(args.references.parent).read_sequences(args.references),
                                 identity_function=load_official_identity(args.identity_function), threshold=args.threshold)
    repo.write_json_atomic(args.output, report)


if __name__ == "__main__":
    main()
