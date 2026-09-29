#!/usr/bin/env python3
"""Run the official entry point twice offline and compare validated output hashes."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from amp_ctmc_2027.compliance import validate_fasta_files
from amp_ctmc_2027.data.fasta_io import FastaRepository
from amp_ctmc_2027.submission import DEFAULT_MANIFEST, _root_path, sha256_file
from amp_ctmc_2027.official_identity import official_identity

ROOT = Path(__file__).resolve().parents[1]


def rehearse(root: Path, manifest_path: Path) -> tuple[str, str]:
    root = root.resolve()
    manifest = json.loads(
        _root_path(root, str(manifest_path)).read_text(encoding="utf-8")
    )
    rehearsal_dir = root / "generate/.rehearsal"
    if rehearsal_dir.exists():
        shutil.rmtree(rehearsal_dir)
    env = {
        **os.environ,
        "AMP_REPOSITORY_ROOT": str(root),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "WANDB_MODE": "disabled",
    }
    output_dirs = [Path("generate/.rehearsal/run1"), Path("generate/.rehearsal/run2")]
    for output_dir in output_dirs:
        subprocess.run(
            [
                "uv",
                "run",
                "--project",
                str(ROOT),
                "--no-sync",
                "generate",
                "--manifest",
                str(manifest_path),
                "--output-dir",
                str(output_dir),
            ],
            cwd=root,
            env=env,
            check=True,
        )
    repository = FastaRepository(root)
    training = repository.read_sequences(
        _root_path(root, manifest["artifacts"]["training_sequences"]["path"])
    )
    references = repository.read_sequences(
        _root_path(root, manifest["artifacts"]["reference_sequences"]["path"])
    )
    hashes: list[str] = []
    for output_dir in output_dirs:
        library_path = output_dir / "library.fasta"
        top_path = output_dir / "top.fasta"
        validate_fasta_files(
            root / library_path,
            root / top_path,
            references,
            training,
            identity_function=official_identity,
            expected_library_count=int(manifest["generation"]["library_count"]),
            expected_top_count=int(manifest["generation"]["top_count"]),
        )
        hashes.extend([sha256_file(root / library_path), sha256_file(root / top_path)])
    if hashes[:2] != hashes[2:]:
        raise RuntimeError(
            f"Generation is not deterministic: output hashes differ: {hashes}"
        )
    return hashes[0], hashes[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    library_hash, top_hash = rehearse(args.root, args.manifest)
    print(f"Deterministic rehearsal passed: library={library_hash} top={top_hash}")


if __name__ == "__main__":
    main()
