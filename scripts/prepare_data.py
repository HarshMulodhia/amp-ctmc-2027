#!/usr/bin/env python3
"""Prepare a canonical Parquet observation table from a disclosed CSV export."""

import argparse
import csv
import json
from pathlib import Path

from amp_ctmc_2027.data.manifests import sha256_file, write_manifest
from amp_ctmc_2027.data.preprocess import clean_observations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--reference-fasta", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/observations.parquet")
    )
    args = parser.parse_args()
    references: set[str] = set()
    if args.reference_fasta:
        from amp_ctmc_2027.data.fasta_io import FastaRepository

        references = set(
            FastaRepository(args.reference_fasta.parent).read_sequences(
                args.reference_fasta
            )
        )
    with args.input_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    result = clean_observations(rows, references)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Install pyarrow to write the canonical Parquet dataset"
        ) from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist([item.as_dict() for item in result.observations]),
        args.output,
    )
    rejected_path = args.output.with_name("rejected.json")
    conflicts_path = args.output.with_name("conflicts.json")
    rejected_path.write_text(
        json.dumps(result.rejected, indent=2) + "\n", encoding="utf-8"
    )
    conflicts_path.write_text(
        json.dumps(result.conflicts, indent=2) + "\n", encoding="utf-8"
    )
    write_manifest(
        args.output.with_name("dataset_manifest.json"),
        {
            "input": str(args.input_csv),
            "input_sha256": sha256_file(args.input_csv),
            "output": str(args.output),
            "output_sha256": sha256_file(args.output),
            "rows_retained": len(result.observations),
            "exact_duplicate_count": result.duplicate_count,
            "rejected_count": len(result.rejected),
            "conflict_count": len(result.conflicts),
            "reference_sha256": sha256_file(args.reference_fasta)
            if args.reference_fasta
            else None,
            "aggregation": "none; observations and censoring are preserved row-wise",
        },
    )


if __name__ == "__main__":
    main()
