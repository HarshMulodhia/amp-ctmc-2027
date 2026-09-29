"""Canonical, integrity checked conditioning tables for CTMC training."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from amp_ctmc_2027.core import CONDITION_NAMES

SCHEMA_VERSION = 1
NORMALIZATION_SCHEMA_VERSION = 1
PROVENANCE = {"measured", "teacher", "derived", "missing"}
PROBABILITY_CONDITIONS = {
    name for name in CONDITION_NAMES if name.endswith("_probability")
}


def panel_semantic_definition(
    ranking_strains: list[str],
    strain_groups: dict[str, list[str]],
    activity_threshold_log2_mic: float,
    activity_temperature: float,
    broad_spectrum_aggregation: dict | None = None,
) -> dict:
    """Canonical, serializable definitions shared by inference and conditioning."""
    if not ranking_strains or len(set(ranking_strains)) != len(ranking_strains):
        raise ValueError("Ranking strain panel must be nonempty and unique")
    if not math.isfinite(float(activity_threshold_log2_mic)):
        raise ValueError("Activity threshold must be finite")
    if not math.isfinite(float(activity_temperature)) or activity_temperature <= 0:
        raise ValueError("Activity temperature must be finite and positive")
    groups = {
        key: list(strain_groups.get(key, []))
        for key in ("gram_negative", "gram_positive", "mdr")
    }
    for name, members in groups.items():
        if len(set(members)) != len(members) or set(members) - set(ranking_strains):
            raise ValueError(f"Invalid {name} group membership")
    aggregation = broad_spectrum_aggregation or {"method": "minimum"}
    if aggregation.get("method") not in {"minimum", "quantile"}:
        raise ValueError("Broad-spectrum aggregation must be minimum or quantile")
    if aggregation["method"] == "quantile":
        q = float(aggregation.get("quantile", -1))
        if not 0 <= q <= 0.5:
            raise ValueError("Conservative broad-spectrum quantile must be in [0, 0.5]")
        aggregation = {"method": "quantile", "quantile": q}
    else:
        aggregation = {"method": "minimum"}
    return {
        "schema_version": 1,
        "ranking_strains": list(ranking_strains),
        "strain_groups": groups,
        "activity_threshold_log2_mic": float(activity_threshold_log2_mic),
        "activity_temperature": float(activity_temperature),
        "broad_spectrum_aggregation": aggregation,
    }


def compute_panel_condition_semantics(
    predicted_log2_mic: dict[str, float], semantic_definition: dict
) -> dict:
    """Convert strain MIC predictions into the canonical condition and audit fields."""
    panel = semantic_definition["ranking_strains"]
    missing = set(panel) - set(predicted_log2_mic)
    if missing:
        raise ValueError(f"Missing ranking strain predictions: {sorted(missing)}")
    mic = {strain: float(predicted_log2_mic[strain]) for strain in panel}
    if not all(math.isfinite(value) for value in mic.values()):
        raise ValueError("Per-strain predicted log2 MIC values must be finite")
    threshold = float(semantic_definition["activity_threshold_log2_mic"])
    temperature = float(semantic_definition["activity_temperature"])
    probabilities = {
        strain: 1.0
        / (1.0 + math.exp(max(-700.0, min(700.0, (value - threshold) / temperature))))
        for strain, value in mic.items()
    }
    groups = semantic_definition["strain_groups"]
    aggregation = semantic_definition["broad_spectrum_aggregation"]
    activity_values = list(probabilities.values())
    if aggregation["method"] == "minimum":
        broad = min(activity_values)
    else:
        broad = float(np.quantile(activity_values, float(aggregation["quantile"])))
    conditions = {
        "broad_spectrum_probability": broad,
        "mean_activity_probability": sum(activity_values) / len(activity_values),
        "predicted_log2_mic_summary": sum(mic.values()) / len(mic),
    }
    for group, condition in (
        ("gram_negative", "mean_gram_negative_activity_probability"),
        ("gram_positive", "mean_gram_positive_activity_probability"),
        ("mdr", "mean_mdr_activity_probability"),
    ):
        members = groups[group]
        conditions[condition] = (
            sum(probabilities[strain] for strain in members) / len(members)
            if members
            else None
        )
    for name, value in conditions.items():
        if value is not None and name.endswith("_probability") and not 0 <= value <= 1:
            raise ValueError(f"Invalid probability condition {name}={value}")
    return {
        "conditions": conditions,
        "audit": {
            "worst_predicted_log2_mic": max(mic.values()),
            "minimum_activity_probability": min(probabilities.values()),
            "panel_coverage": len(mic) / len(panel),
            "per_strain_predicted_log2_mic": mic,
            "per_strain_activity_probability": probabilities,
        },
    }


def fit_condition_normalization(
    rows: list[dict],
    train_sequences: list[str],
    fitting_split_sha256: str,
    lower_clip: float = -5.0,
    upper_clip: float = 5.0,
) -> dict:
    """Fit z-score parameters from observed entries on CTMC train sequences only."""
    if (
        not math.isfinite(lower_clip)
        or not math.isfinite(upper_clip)
        or lower_clip >= upper_clip
    ):
        raise ValueError("Normalization clipping bounds must be finite and ordered")
    train = set(train_sequences)
    train_rows = [row for row in rows if row["sequence"] in train]
    if {row["sequence"] for row in train_rows} != train:
        raise ValueError(
            "Normalization train sequences do not have complete condition rows"
        )
    stats = {}
    for name in CONDITION_NAMES:
        values = [
            float(row.get("values", {}).get(name, 0.0))
            for row in train_rows
            if row.get("observed", {}).get(name, False)
        ]
        if not values:
            mean, std = 0.0, 1.0
        else:
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            std = math.sqrt(variance)
        # A constant train feature is represented on its native scale (unit scale).
        stats[name] = {
            "mean": mean,
            "std": std if std > 0 else 1.0,
            "count": len(values),
            "lower_clip": float(lower_clip),
            "upper_clip": float(upper_clip),
        }
    return {
        "schema_version": NORMALIZATION_SCHEMA_VERSION,
        "condition_names": list(CONDITION_NAMES),
        "fit_split": "train",
        "fitting_split_sha256": fitting_split_sha256,
        "statistics": stats,
    }


def normalize_condition_values(values, observed, normalization: dict):
    """Apply stored z-score/clipping to observed entries only; masks are unchanged."""
    import torch

    if normalization.get(
        "schema_version"
    ) != NORMALIZATION_SCHEMA_VERSION or normalization.get("condition_names") != list(
        CONDITION_NAMES
    ):
        raise ValueError("Condition normalization schema mismatch")
    out = values.clone()
    mask = observed.bool()
    for index, name in enumerate(CONDITION_NAMES):
        stat = normalization.get("statistics", {}).get(name, {})
        mean, std = (
            float(stat.get("mean", float("nan"))),
            float(stat.get("std", float("nan"))),
        )
        if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
            raise ValueError(f"Invalid normalization parameters for {name}")
        transformed = ((out[..., index] - mean) / std).clamp(
            float(stat["lower_clip"]), float(stat["upper_clip"])
        )
        out[..., index] = torch.where(
            mask[..., index], transformed, torch.zeros_like(transformed)
        )
    return out


def canonical_sequence(sequence: str) -> str:
    value = "".join(sequence.split()).upper()
    if not value or any(aa not in "ACDEFGHIKLMNPQRSTVWY" for aa in value):
        raise ValueError(f"Invalid canonical peptide sequence: {sequence!r}")
    return value


def table_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def measured_overrides(teacher_values: dict, measured_row: dict | None) -> dict:
    """Apply only explicitly observed measured fields over teacher predictions."""
    merged = {
        "values": dict(teacher_values["values"]),
        "observed": dict(teacher_values["observed"]),
        "provenance": dict(teacher_values["provenance"]),
        "uncertainty": dict(teacher_values.get("uncertainty", {})),
    }
    if measured_row is None:
        return merged
    if measured_row.get("condition_names") != list(CONDITION_NAMES):
        raise ValueError("Measured condition schema/order mismatch")
    for name in CONDITION_NAMES:
        if measured_row.get("observed", {}).get(name, False):
            value = float(measured_row["values"][name])
            if not math.isfinite(value):
                raise ValueError(f"Nonfinite measured value for {name}")
            merged["values"][name] = value
            merged["observed"][name] = True
            merged["provenance"][name] = "measured"
            merged["uncertainty"][name] = measured_row.get("uncertainty", {}).get(name)
    return merged


def validate_rows(
    rows: list[dict], expected_sequences: list[str] | None = None
) -> list[dict]:
    """Validate wide rows; condition names/order are explicit schema, never inferred."""
    expected = {canonical_sequence(s) for s in (expected_sequences or [])}
    normalized: dict[str, dict] = {}
    allowed = set(CONDITION_NAMES)
    for row in rows:
        sequence = canonical_sequence(str(row["sequence"]))
        values, observed, provenance = (
            row.get("values", {}),
            row.get("observed", {}),
            row.get("provenance", {}),
        )
        if (
            set(values) - allowed
            or set(observed) - allowed
            or set(provenance) - allowed
        ):
            raise ValueError("Condition table contains unknown condition names")
        # Require explicit ordered schema metadata; JSON object iteration is not a contract.
        if row.get("condition_names") != list(CONDITION_NAMES):
            raise ValueError(
                "Condition schema/order must exactly match CONDITION_NAMES"
            )
        result = {
            "sequence": sequence,
            "condition_names": list(CONDITION_NAMES),
            "values": {},
            "observed": {},
            "provenance": {},
            "uncertainty": {},
        }
        for name in CONDITION_NAMES:
            value = values.get(name)
            mask = bool(observed.get(name, False))
            source = provenance.get(name, "missing")
            if source not in PROVENANCE:
                raise ValueError(f"Invalid provenance for {name}: {source}")
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"Nonfinite condition value for {sequence}/{name}")
            if (
                name in PROBABILITY_CONDITIONS
                and value is not None
                and not 0 <= float(value) <= 1
            ):
                raise ValueError(f"Probability condition {name} must be in [0, 1]")
            if source == "missing" and mask:
                raise ValueError("Missing condition cannot be observed")
            if source != "missing" and not mask:
                raise ValueError("Condition provenance requires an observed value")
            if mask and value is None:
                raise ValueError("Observed condition has no value")
            result["values"][name] = float(value) if value is not None else 0.0
            result["observed"][name] = mask
            result["provenance"][name] = source
            uncertainty = row.get("uncertainty", {}).get(name)
            if isinstance(uncertainty, (int, float)) and not math.isfinite(
                float(uncertainty)
            ):
                raise ValueError(f"Nonfinite uncertainty for {sequence}/{name}")
            result["uncertainty"][name] = uncertainty
        if "audit" in row:
            result["audit"] = row["audit"]
        old = normalized.get(sequence)
        if old is not None and old != result:
            raise ValueError(
                f"Duplicate sequence rows have incompatible values: {sequence}"
            )
        normalized[sequence] = result
    if expected and set(normalized) != expected:
        missing, extra = expected - set(normalized), set(normalized) - expected
        raise ValueError(
            f"Condition table sequence coverage differs (missing={len(missing)}, extra={len(extra)})"
        )
    return [normalized[s] for s in sorted(normalized)]


def load_condition_table(
    table_path: Path, manifest_path: Path, expected_sequences: list[str] | None = None
) -> tuple[list[dict], dict]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get(
        "condition_names"
    ) != list(CONDITION_NAMES):
        raise ValueError("Condition manifest schema/order mismatch")
    for field in ("property_checkpoint_sha256", "property_metadata_sha256"):
        value = manifest.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"Condition manifest has invalid {field}")
        try:
            int(value, 16)
        except ValueError as exc:
            raise ValueError(f"Condition manifest has invalid {field}") from exc
    semantic = manifest.get("semantic_definition")
    if not isinstance(semantic, dict):
        raise ValueError("Condition manifest lacks semantic_definition")
    canonical = panel_semantic_definition(
        semantic.get("ranking_strains", []),
        semantic.get("strain_groups", {}),
        semantic.get("activity_threshold_log2_mic", float("nan")),
        semantic.get("activity_temperature", float("nan")),
        semantic.get("broad_spectrum_aggregation"),
    )
    if canonical != semantic:
        raise ValueError("Condition manifest semantic definition is invalid")
    actual = table_sha256(table_path)
    if actual != manifest.get("table_sha256"):
        raise ValueError("Condition table SHA-256 does not match manifest")
    rows = [
        json.loads(line)
        for line in table_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    return validate_rows(rows, expected_sequences), manifest


def write_condition_table(
    table_path: Path, rows: list[dict], metadata: dict
) -> tuple[Path, Path]:
    """Write a create-once immutable JSONL table and SHA-256 sidecar manifest."""
    rows = validate_rows(rows)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = table_path.with_suffix(table_path.suffix + ".manifest.json")
    if table_path.exists() or manifest_path.exists():
        raise FileExistsError(
            "Condition cache artifacts are immutable and already exist"
        )
    table_path.write_text(
        "".join(
            json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "condition_names": list(CONDITION_NAMES),
        "table_sha256": table_sha256(table_path),
        **metadata,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return table_path, manifest_path
