"""Compatibility wrapper for the manifest-driven production generator."""

from pathlib import Path

from amp_ctmc_2027.config import AMPConfig
from amp_ctmc_2027.data.fasta_io import FastaRepository
from amp_ctmc_2027.submission import generate_from_manifest


class GenerationPipeline:
    """Delegate legacy API callers to the strict offline submission path."""

    def __init__(self, config: AMPConfig, fasta_repo: FastaRepository) -> None:
        self.config = config
        self.repo = fasta_repo

    def run(self) -> tuple[Path, Path]:
        return generate_from_manifest(self.repo.base_dir)
