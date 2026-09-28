"""Canonical observation schema for AMP sequences and assay labels."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class AMPObservation:
    sequence: str
    sequence_sha256: str
    source: str
    source_record_id: str | None = None
    is_amp: float | None = None
    is_hemolytic: float | None = None
    organism: str | None = None
    strain: str | None = None
    panel_target: str | None = None
    mic_value: float | None = None
    mic_unit_original: str | None = None
    mic_uM: float | None = None
    mic_upper_uM: float | None = None
    mic_censor: Literal["none", "left", "right", "interval"] | None = None
    hc50_uM: float | None = None
    hc50_upper_uM: float | None = None
    hc50_censor: Literal["none", "left", "right", "interval"] | None = None
    terminal_modification: str | None = None
    other_modification: str | None = None
    is_linear: bool | None = None
    split_cluster_id: str | None = None
    split: Literal["train", "val", "test"] | None = None
    label_provenance: Literal["measured", "aggregated", "pseudo", "missing"] = "missing"

    def as_dict(self) -> dict:
        return asdict(self)
