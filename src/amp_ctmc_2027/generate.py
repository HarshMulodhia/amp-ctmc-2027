from __future__ import annotations

import argparse
import os
from pathlib import Path

from amp_ctmc_2027.submission import DEFAULT_MANIFEST, generate_from_manifest

REPOSITORY_ROOT = Path(
    os.environ.get("AMP_REPOSITORY_ROOT", Path(__file__).resolve().parents[2])
).resolve()


def main(argv: list[str] | None = None) -> None:
    """Official no-required-arguments entry point."""
    parser = argparse.ArgumentParser(
        description="Generate AMP Challenge submission FASTA files"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)
    generate_from_manifest(
        REPOSITORY_ROOT,
        args.manifest,
        output_dir=args.output_dir,
        seed=args.seed,
    )


def generate_broad_spectrum() -> None:
    main()


if __name__ == "__main__":
    main()
