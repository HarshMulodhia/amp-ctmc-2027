"""Pinned MMseqs2 clustering adapter; each cluster remains indivisible in splits."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from amp_ctmc_2027.data.fasta_io import FastaRepository


def cluster_fasta(
    input_fasta: Path, output_dir: Path, identity: float = 0.5, coverage: float = 0.8
) -> dict[str, str]:
    if not 0.0 < identity <= 1.0 or not 0.0 < coverage <= 1.0:
        raise ValueError("identity and coverage must be in (0, 1]")
    exe = shutil.which("mmseqs")
    if exe is None:
        raise RuntimeError(
            "MMseqs2 is required for homology-aware splits (install the pinned version from configs)"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    repo = FastaRepository(input_fasta.parent)
    sequences = repo.read_sequences(input_fasta)
    ids = {f"seq{i}": seq for i, seq in enumerate(sequences)}
    fasta = output_dir / "cluster_input.fasta"
    fasta.write_text(
        "".join(f">seq{i}\n{seq}\n" for i, seq in enumerate(sequences)),
        encoding="utf-8",
    )
    db, clu, tmp, tsv = (
        output_dir / name for name in ("db", "cluster", "tmp", "clusters.tsv")
    )
    subprocess.run(
        [exe, "createdb", str(fasta), str(db)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            exe,
            "cluster",
            str(db),
            str(clu),
            str(tmp),
            "--min-seq-id",
            str(identity),
            "-c",
            str(coverage),
            "--cov-mode",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [exe, "createtsv", str(db), str(db), str(clu), str(tsv)],
        check=True,
        capture_output=True,
        text=True,
    )
    assignments: dict[str, str] = {}
    for line in tsv.read_text().splitlines():
        rep_id, member_id = line.split("\t")[:2]
        assignments[member_id] = rep_id
    return {
        ids[key]: cluster_id for key, cluster_id in assignments.items() if key in ids
    }
