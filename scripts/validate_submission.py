#!/usr/bin/env python3
"""Validate generated FASTA files with the vendored challenge validator."""

import argparse
from pathlib import Path

from amp_ctmc_2027.compliance import validate_fasta_files
from amp_ctmc_2027.data.fasta_io import FastaRepository
from amp_ctmc_2027.official_identity import official_identity
from amp_ctmc_2027.submission import DEFAULT_MANIFEST, _root_path, validate_manifest

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--library", type=Path, default=Path("generate/library.fasta"))
    parser.add_argument("--top", type=Path, default=Path("generate/top.fasta"))
    args = parser.parse_args()
    manifest = validate_manifest(ROOT, args.manifest)
    repo = FastaRepository(ROOT)
    report = validate_fasta_files(
        _root_path(ROOT, str(args.library)),
        _root_path(ROOT, str(args.top)),
        repo.read_sequences(
            _root_path(ROOT, manifest["artifacts"]["reference_sequences"]["path"])
        ),
        repo.read_sequences(
            _root_path(ROOT, manifest["artifacts"]["training_sequences"]["path"])
        ),
        identity_function=official_identity,
        expected_library_count=int(manifest["generation"]["library_count"]),
        expected_top_count=int(manifest["generation"]["top_count"]),
    )
    repo.write_json_atomic(Path("generate/compliance_report.json"), report)
    print("Submission files pass the local validator.")


if __name__ == "__main__":
    main()
