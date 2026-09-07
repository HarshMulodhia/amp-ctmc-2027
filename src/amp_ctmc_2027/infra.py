"""Infrastructure: validation, experiment tracking, and runtime utilities."""
from __future__ import annotations

import logging
import os
from typing import Any, Iterable

import torch

from amp_ctmc_2027.config import AMPConfig

logger = logging.getLogger(__name__)

try:
    import wandb
except Exception:  # pragma: no cover - runtime dependency fallback
    wandb = None


class ConstraintValidator:
    """Hard assignment constraint validator for peptide sequences."""

    def __init__(self, alphabet: str, min_len: int, max_len: int, forbidden: set[str]) -> None:
        self.alphabet = set(alphabet)
        self.min_len = min_len
        self.max_len = max_len
        self.forbidden = forbidden

    def is_valid(self, sequence: str) -> bool:
        """Check whether one sequence satisfies hard constraints."""
        if not (self.min_len <= len(sequence) <= self.max_len):
            return False
        if any(char not in self.alphabet for char in sequence):
            return False
        if sequence in self.forbidden:
            return False
        return True

    def filter_valid(self, sequences: Iterable[str], seen: set[str] | None = None) -> list[str]:
        """Return valid ordered-unique sequences."""
        output: list[str] = []
        local_seen: set[str] = set() if seen is None else seen
        for sequence in sequences:
            seq = sequence.strip().upper()
            if seq in local_seen:
                continue
            if self.is_valid(seq):
                local_seen.add(seq)
                output.append(seq)
        return output

    def assert_library(self, library: list[str], expected_size: int | None = None) -> None:
        """Raise AssertionError if any hard constraint is violated."""
        if expected_size is not None and len(library) != expected_size:
            raise AssertionError(f"Expected {expected_size} sequences, found {len(library)}")
        if len(library) != len(set(library)):
            raise AssertionError("Library contains duplicates")
        invalid = [sequence for sequence in library if not self.is_valid(sequence)]
        if invalid:
            raise AssertionError(f"Invalid sequence(s) found: {invalid[:3]}")


class WandbTracker:
    """Light wrapper around wandb with safe no-op behavior."""

    def __init__(self, config: AMPConfig, *, job_type: str) -> None:
        self._run = None
        if not config.wandb_enabled:
            return
        if wandb is None:
            logger.warning("wandb is not available; skipping experiment tracking")
            return
        mode = os.getenv("WANDB_MODE", config.wandb_mode)
        try:
            self._run = wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                config=config.model_dump(mode="json"),
                job_type=job_type,
                mode=mode,
            )
        except Exception as exc:  # pragma: no cover - network/login dependent
            logger.warning("Failed to initialize wandb (%s); continuing without tracking", exc)
            self._run = None

    @property
    def enabled(self) -> bool:
        return self._run is not None

    def log(self, metrics: dict[str, Any], *, step: int | None = None) -> None:
        if self._run is None:
            return
        if step is None:
            self._run.log(metrics)
            return
        self._run.log(metrics, step=step)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()


def resolve_device(config: AMPConfig) -> torch.device:
    """Resolve runtime torch device from config preference."""
    preference = config.device
    if preference == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("AMPConfig.device='cuda' but CUDA is not available")
    if preference == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        raise RuntimeError("AMPConfig.device='mps' but MPS is not available")
    if preference == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def log_device(device: torch.device) -> None:
    """Log selected device and accelerator details."""
    if device.type == "cuda":
        current_idx = torch.cuda.current_device()
        logger.info(
            "Using CUDA device %s (%s)",
            current_idx,
            torch.cuda.get_device_name(current_idx),
        )
        return
    logger.info("Using %s device", device.type.upper())
