from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class TokenMetadata:
    """Tokenizer metadata for checkpoint serialization."""

    vocab: str
    max_length: int
    eos_token: str
    mask_token: str
    pad_token: str
    version: int = 2
    canvas_semantics: str = "residues_plus_eos"


class AMPCanvasEncoder:
    """Encodes/decodes peptides to/from fixed-length token canvases."""

    eos_token: str = "<EOS>"
    mask_token: str = "<MASK>"
    pad_token: str = "<PAD>"

    def __init__(
        self, max_length: int = 50, vocab: str = "ACDEFGHIKLMNPQRSTVWY"
    ) -> None:
        self.max_length = max_length
        self.vocab = vocab
        self.alphabet = set(vocab)
        self.tokens = list(vocab) + [self.eos_token, self.mask_token, self.pad_token]
        self.token_to_idx = {token: idx for idx, token in enumerate(self.tokens)}
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}

        self.eos_idx = self.token_to_idx[self.eos_token]
        self.mask_idx = self.token_to_idx[self.mask_token]
        self.pad_idx = self.token_to_idx[self.pad_token]
        self.amino_indices = [self.token_to_idx[a] for a in vocab]

    def to_metadata(self) -> dict:
        """Return serializable metadata."""
        return TokenMetadata(
            vocab=self.vocab,
            max_length=self.max_length,
            eos_token=self.eos_token,
            mask_token=self.mask_token,
            pad_token=self.pad_token,
        ).__dict__

    @classmethod
    def from_metadata(cls, metadata: dict) -> AMPCanvasEncoder:
        """Create encoder from checkpoint metadata."""
        if (
            metadata.get("version", 1) < 2
            or metadata.get("canvas_semantics") != "residues_plus_eos"
        ):
            raise ValueError(
                "Legacy tokenizer canvas is incompatible; retrain or migrate this checkpoint explicitly"
            )
        return cls(max_length=int(metadata["max_length"]), vocab=str(metadata["vocab"]))

    @property
    def vocab_size(self) -> int:
        """Return tokenizer vocabulary size."""
        return len(self.tokens)

    @property
    def canvas_length(self) -> int:
        """Maximum residues plus one always-reserved EOS position."""
        return self.max_length + 1

    def encode(self, sequence: str) -> torch.Tensor:
        """Encode a peptide sequence into a fixed-length tensor."""
        sequence = sequence.strip().upper()
        if len(sequence) > self.max_length:
            raise ValueError("Sequence too long for canvas")
        if any(char not in self.alphabet for char in sequence):
            raise ValueError("Sequence contains invalid amino acid")

        canvas = [self.token_to_idx[aa] for aa in sequence]
        canvas.append(self.eos_idx)
        canvas.extend([self.pad_idx] * (self.max_length - len(sequence)))
        return torch.tensor(canvas, dtype=torch.long)

    def decode(self, canvas: torch.Tensor) -> str:
        """Decode a token canvas into a peptide sequence."""
        values = canvas.detach().cpu().tolist()
        if len(values) != self.canvas_length:
            raise ValueError("Canvas length mismatch")

        sequence: list[str] = []
        eos_seen = False
        for idx in values:
            token = self.idx_to_token.get(int(idx))
            if token is None:
                raise ValueError("Unknown token index")
            if token == self.pad_token:
                if eos_seen:
                    continue
            elif token == self.eos_token:
                eos_seen = True
            else:
                if eos_seen:
                    raise ValueError("Malformed canvas: non-PAD token after EOS")
                if token in {self.mask_token, self.pad_token, self.eos_token}:
                    raise ValueError("Malformed content token")
                sequence.append(token)
        return "".join(sequence)

    def mask_canvas(
        self, canvas: torch.Tensor, t: float, rng: np.random.Generator
    ) -> torch.Tensor:
        """Apply factorized masking corruption using kappa(t)=sin^2(pi t/2)."""
        corrupted = canvas.clone()
        kappa = math.sin(math.pi * float(t) / 2.0) ** 2
        non_pad = corrupted != self.pad_idx
        reveal_draws = torch.from_numpy(rng.random(self.canvas_length)).to(
            dtype=torch.float32
        )
        keep = reveal_draws < kappa
        to_mask = non_pad & ~keep
        if not torch.any(to_mask) and torch.any(non_pad):
            valid_idx = torch.nonzero(non_pad, as_tuple=False).flatten()
            chosen = int(valid_idx[int(rng.integers(0, valid_idx.numel()))].item())
            to_mask = torch.zeros_like(non_pad)
            to_mask[chosen] = True
        corrupted[to_mask] = self.mask_idx
        return corrupted


class AMPDataset(Dataset[torch.Tensor]):
    """Dataset of encoded AMP sequences."""

    def __init__(self, sequences: list[str], encoder: AMPCanvasEncoder) -> None:
        self.items = [encoder.encode(sequence) for sequence in sequences]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.items[idx]


class ConditionedAMPDataset(Dataset[dict]):
    """Encoded sequences joined to validated canonical condition rows."""

    def __init__(
        self,
        sequences: list[str],
        encoder: AMPCanvasEncoder,
        rows: list[dict],
        normalization: dict | None = None,
    ):
        from amp_ctmc_2027.core import CONDITION_NAMES

        by_sequence = {row["sequence"]: row for row in rows}
        missing = set(sequences) - set(by_sequence)
        if missing:
            raise ValueError(
                f"Condition table lacks {len(missing)} requested sequences"
            )
        self.items = []
        for sequence in sequences:
            row = by_sequence[sequence]
            if row.get("condition_names") != list(CONDITION_NAMES):
                raise ValueError("Condition row ordering differs from CONDITION_NAMES")
            values = torch.tensor(
                [row["values"][n] for n in CONDITION_NAMES], dtype=torch.float32
            )
            observed = torch.tensor(
                [row["observed"][n] for n in CONDITION_NAMES], dtype=torch.bool
            )
            if normalization is not None:
                from amp_ctmc_2027.data.conditions import normalize_condition_values

                values = normalize_condition_values(
                    values.unsqueeze(0), observed.unsqueeze(0), normalization
                ).squeeze(0)
            self.items.append(
                {
                    "tokens": encoder.encode(sequence),
                    "condition_values": values,
                    "condition_observed": observed,
                    "condition_provenance": [
                        row["provenance"][n] for n in CONDITION_NAMES
                    ],
                    "sequence": sequence,
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


def collate_conditioned(batch: list[dict]) -> dict:
    """Stack tensors and retain identifiers/provenance as per-example lists."""
    return {
        "tokens": torch.stack([item["tokens"] for item in batch]),
        "condition_values": torch.stack([item["condition_values"] for item in batch]),
        "condition_observed": torch.stack(
            [item["condition_observed"] for item in batch]
        ),
        "condition_provenance": [item["condition_provenance"] for item in batch],
        "sequence_ids": [item["sequence"] for item in batch],
    }
