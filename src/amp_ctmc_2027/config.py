from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from pydantic import BaseModel, Field, model_validator


class AMPConfig(BaseModel):
    """Typed configuration for training and generation."""

    vocab: str = "ACDEFGHIKLMNPQRSTVWY"
    max_length: int = 50
    min_length: int = 8

    d_model: int = 512
    n_heads: int = 16
    n_layers: int = 10
    d_ff: int = 2048
    dropout: float = 0.1

    batch_size: int = 256
    learning_rate: float = 1e-3
    learning_rate_min: float = 5e-7
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    n_epochs: int = 50
    early_stopping_patience: int = 10
    early_stopping_threshold: float = 5e-5
    scheduler_type: str = "cosine"
    val_fraction: float = 0.1
    log_masking_diagnostics: bool = True
    debug_overfit_samples: int | None = None

    reverse_steps: int = 64
    generation_batch_size: int = 256
    generation_temperatures: tuple[float, ...] = (0.85, 1.0, 1.1)
    generation_step_counts: tuple[int, ...] = (96, 64, 48)
    generation_shares: tuple[float, ...] = (0.4, 0.4, 0.2)
    confidence_reveal_fraction: float = 0.0
    dfm_stochasticity: float = 0.0
    candidate_pool_size: int = 120000
    max_generation_attempts: int = 300

    n_sequences: int = 50000
    top_k: int = 100
    mmr_lambda_diversity: float = 0.5
    score_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "discriminator": 0.18,
            "realism": 0.16,
            "conformity": 0.14,
            "novelty": 0.12,
            "quality": 0.12,
            "activity_hemolysis": 0.18,
            "esm2_pseudo_perplexity": 0.10,
        }
    )
    score_combine_mode: Literal["arithmetic", "geometric"] = "geometric"
    discriminator_ensemble_size: int = 5

    seed: int = 42
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    wandb_enabled: bool = True
    wandb_project: str = "AMPGen"
    wandb_entity: str | None = None
    wandb_mode: Literal["online", "offline", "disabled"] = "online"

    training_fasta_path: Path = Path("data/training/training.fasta")
    antibacterial_fasta_path: Path = Path("data/antibacterial.fasta")
    background_fasta_path: Path = Path("data/generic/background.fasta")
    checkpoint_dir: Path = Path("checkpoint")
    generate_dir: Path = Path("generate_broad_spectrum")

    novelty_similarity_ceiling: float = 0.60
    diversity_similarity_ceiling: float = 0.85
    top_identity_ceiling: float = 0.80

    external_scorer_project_dir: Path = Path("scorer")
    external_scorer_timeout_sec: int = 600

    @model_validator(mode="after")
    def validate_fields(self) -> "AMPConfig":
        if self.min_length < 1 or self.max_length < self.min_length:
            raise ValueError("Invalid min/max length")
        if self.max_length != 50:
            raise ValueError("The assignment requires a fixed canvas length of 50")
        if len(set(self.vocab)) != 20:
            raise ValueError("Vocab must contain exactly 20 unique amino acids")
        if len(self.generation_temperatures) != len(self.generation_step_counts):
            raise ValueError("generation_temperatures and generation_step_counts must match")
        if len(self.generation_shares) != len(self.generation_temperatures):
            raise ValueError("generation_shares must match generation_temperatures")
        if not np.isclose(sum(self.generation_shares), 1.0, atol=1e-6):
            raise ValueError("generation_shares must sum to 1")
        if self.n_sequences < 1:
            raise ValueError("n_sequences must be positive")
        if self.top_k < 1 or self.top_k > self.n_sequences:
            raise ValueError("top_k must be in [1, n_sequences]")
        if any(weight < 0 for weight in self.score_weights.values()):
            raise ValueError("score weights must be nonnegative")
        for value, name in [
            (self.novelty_similarity_ceiling, "novelty_similarity_ceiling"),
            (self.diversity_similarity_ceiling, "diversity_similarity_ceiling"),
            (self.top_identity_ceiling, "top_identity_ceiling"),
        ]:
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0, 1]")
        if self.external_scorer_timeout_sec < 1:
            raise ValueError("external_scorer_timeout_sec must be positive")
        if self.debug_overfit_samples is not None and self.debug_overfit_samples < 1:
            raise ValueError("debug_overfit_samples must be positive when set")
        return self

    def to_json_file(self, path: Path) -> None:
        """Serialize config to disk."""
        path.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    @classmethod
    def from_json_file(cls, path: Path) -> "AMPConfig":
        """Load config from a JSON file."""
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)


def set_global_determinism(seed: int) -> None:
    """Enable deterministic behavior across Python, NumPy, and PyTorch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)