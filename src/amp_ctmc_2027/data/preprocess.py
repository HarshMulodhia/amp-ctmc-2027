"""Deterministic canonicalization and audit helpers for source observations."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from amp_ctmc_2027.data.schema import AMPObservation

CANONICAL = frozenset("ACDEFGHIKLMNPQRSTVWY")
MODIFICATION_TERMS = (
    "cyclic",
    "branched",
    "stapled",
    "lipid",
    "glycosyl",
    "peg",
    "dendrimer",
    "terminal",
    "noncanonical",
)


@dataclass(frozen=True)
class CleaningResult:
    observations: tuple[AMPObservation, ...]
    rejected: tuple[dict, ...]
    conflicts: tuple[dict, ...]
    duplicate_count: int


def concentration_to_um(
    value: float, unit: str, molecular_weight_g_mol: float | None = None
) -> float:
    """Convert molar concentration to µM; mass units require explicit molecular weight.

    Supported mass units are mg/L, µg/mL, g/L, and mg/mL. This helper never
    infers molecular weight from an amino-acid string or silently converts mass.
    """
    normalized = (
        unit.strip().lower().replace("μ", "u").replace("µ", "u").replace(" ", "")
    )
    molar_factors = {"m": 1_000_000.0, "mm": 1_000.0, "um": 1.0, "nm": 1e-3, "pm": 1e-6}
    if normalized in molar_factors:
        return float(value) * molar_factors[normalized]
    mass_as_g_l = {"mg/l": 1e-3, "ug/ml": 1e-3, "g/l": 1.0, "mg/ml": 1.0}
    if normalized not in mass_as_g_l:
        raise ValueError(f"Unsupported concentration unit: {unit}")
    if molecular_weight_g_mol is None or molecular_weight_g_mol <= 0:
        raise ValueError(
            "Mass concentration conversion requires a positive molecular weight in g/mol"
        )
    return (
        float(value)
        * mass_as_g_l[normalized]
        / float(molecular_weight_g_mol)
        * 1_000_000.0
    )


def clean_observations(
    rows: Iterable[Mapping], reference_sequences: set[str] | None = None
) -> CleaningResult:
    """Clean records in documented order, retaining every valid observation/provenance row.

    Reference sequences are excluded from analysis rows; their separate compliance input
    must never be used as a training source by this function.
    """
    reference_sequences = {s.strip().upper() for s in (reference_sequences or set())}
    kept: list[AMPObservation] = []
    rejected: list[dict] = []
    for row_no, source_row in enumerate(rows, start=1):
        row = dict(source_row)
        seq = "".join(str(row.get("sequence", "")).split()).upper()
        why = None
        if not seq:
            why = "empty"
        elif any(char not in CANONICAL for char in seq):
            why = "noncanonical_residue"
        elif not 8 <= len(seq) <= 50:
            why = "length_out_of_range"
        else:
            metadata = " ".join(
                str(row.get(k) or "")
                for k in (
                    "terminal_modification",
                    "other_modification",
                    "modification",
                    "is_linear",
                )
            ).lower()
            linear_value = row.get("is_linear")
            non_linear = linear_value is False or str(linear_value).strip().lower() in {
                "false",
                "0",
                "no",
                "nonlinear",
                "non-linear",
            }
            if any(term in metadata for term in MODIFICATION_TERMS) or non_linear:
                why = "modified_or_non_linear"
            elif seq in reference_sequences:
                why = "compliance_reference_overlap"
        if why:
            rejected.append({"row": row_no, "sequence": seq, "reason": why})
            continue
        digest = hashlib.sha256(seq.encode("ascii")).hexdigest()
        kept.append(
            AMPObservation(
                sequence=seq,
                sequence_sha256=digest,
                source=str(row.get("source", "unspecified")),
                source_record_id=None
                if row.get("source_record_id") is None
                else str(row["source_record_id"]),
                is_amp=None if row.get("is_amp") is None else float(row["is_amp"]),
                is_hemolytic=None
                if row.get("is_hemolytic") is None
                else float(row["is_hemolytic"]),
                organism=row.get("organism"),
                strain=row.get("strain"),
                panel_target=row.get("panel_target"),
                mic_value=row.get("mic_value"),
                mic_unit_original=row.get("mic_unit_original"),
                mic_uM=row.get("mic_uM"),
                mic_upper_uM=row.get("mic_upper_uM"),
                mic_censor=row.get("mic_censor"),
                hc50_uM=row.get("hc50_uM"),
                hc50_censor=row.get("hc50_censor"),
                hc50_upper_uM=row.get("hc50_upper_uM"),
                terminal_modification=row.get("terminal_modification"),
                other_modification=row.get("other_modification"),
                is_linear=row.get("is_linear"),
                label_provenance=str(row.get("label_provenance", "measured")),
            )
        )
    grouped: dict[str, list[AMPObservation]] = defaultdict(list)
    for item in kept:
        grouped[item.sequence].append(item)
    conflicts = []
    for seq, items in sorted(grouped.items()):
        labels = {i.is_amp for i in items if i.is_amp is not None}
        if len(labels) > 1:
            conflicts.append(
                {
                    "sequence": seq,
                    "kind": "contradictory_amp_labels",
                    "labels": sorted(labels),
                }
            )
        # Keep all original records: aggregation must preserve censoring and strain.
    deduplicated = tuple(item for seq in sorted(grouped) for item in grouped[seq])
    return CleaningResult(
        deduplicated, tuple(rejected), tuple(conflicts), len(kept) - len(grouped)
    )
