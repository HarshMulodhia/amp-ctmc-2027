from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


class FastaRepository:
    """Repository-style access to FASTA files and atomic artifact writes."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir

    def resolve(self, path: Path) -> Path:
        """Resolve a path relative to the repository root."""
        return path if path.is_absolute() else (self.base_dir / path)

    def read_sequences(self, path: Path) -> list[str]:
        """Read peptide sequences from FASTA."""
        resolved = self.resolve(path)
        if not resolved.exists():
            raise FileNotFoundError(f"Missing FASTA file: {resolved}")
        sequences: list[str] = []
        for record in SeqIO.parse(str(resolved), "fasta"):
            sequences.append(str(record.seq).upper().strip())
        return sequences

    def write_fasta(
        self, path: Path, sequences: Iterable[str], id_prefix: str = "amp"
    ) -> None:
        """Write peptide sequences as deterministic FASTA records."""
        resolved = self.resolve(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = resolved.with_suffix(resolved.suffix + ".tmp")
        records = [
            SeqRecord(
                Seq(sequence),
                id=f"seq{idx}" if id_prefix == "seq" else f"{id_prefix}_{idx:06d}",
                description="",
            )
            for idx, sequence in enumerate(sequences, start=1)
        ]
        with open(tmp_path, "w", encoding="utf-8", newline="\n") as handle:
            SeqIO.write(records, handle, "fasta")
        tmp_path.replace(resolved)

    def write_json_atomic(self, path: Path, payload: dict) -> None:
        """Write JSON atomically to disk."""
        resolved = self.resolve(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = resolved.with_suffix(resolved.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2)
        tmp_path.replace(resolved)
