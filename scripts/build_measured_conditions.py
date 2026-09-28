#!/usr/bin/env python3
"""Aggregate train-split prepared assays into measured CTMC condition rows.

Replicates use the median. Censored MICs are represented by conservative activity
probability bounds: left censoring uses the lower probability bound, right
censoring uses zero, and interval censoring uses the interval midpoint. Censor
states and bounds remain in the audit output; censoring boundaries are never
relabeled as exact. HC50 contributes a numeric condition only from exact assays.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from amp_ctmc_2027.core import CONDITION_NAMES
from amp_ctmc_2027.data.conditions import (
    canonical_sequence,
    compute_panel_condition_semantics,
    panel_semantic_definition,
    table_sha256,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(max(-700.0, min(700.0, -x))))


def _mic_probability_interval(
    row: dict, semantics: dict
) -> tuple[float, float, float, str, dict]:
    value = row.get("mic_uM")
    censor = row.get("mic_censor") or "none"
    if value is None or float(value) <= 0:
        raise ValueError(
            "MIC observation must include a positive mic_uM boundary/value"
        )
    boundary = math.log2(float(value))
    threshold, temp = (
        semantics["activity_threshold_log2_mic"],
        semantics["activity_temperature"],
    )
    p_boundary = _sigmoid((threshold - boundary) / temp)
    if censor == "none":
        return p_boundary, p_boundary, p_boundary, censor, {"log2_mic": boundary}
    if censor == "left":
        return p_boundary, p_boundary, 1.0, censor, {"upper_log2_mic": boundary}
    if censor == "right":
        return 0.0, 0.0, p_boundary, censor, {"lower_log2_mic": boundary}
    if censor == "interval":
        upper = row.get("mic_upper_uM")
        if upper is None or float(upper) <= float(value):
            raise ValueError("Interval-censored MIC requires mic_upper_uM > mic_uM")
        high = math.log2(float(upper))
        low_p = _sigmoid((threshold - high) / temp)
        return (
            (p_boundary + low_p) / 2.0,
            low_p,
            p_boundary,
            censor,
            {"lower_log2_mic": boundary, "upper_log2_mic": high},
        )
    raise ValueError(f"Unsupported MIC censor state: {censor}")


def _robust(values: list[float]) -> tuple[float, float]:
    values = sorted(values)
    mid = len(values) // 2
    median = values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2
    deviations = sorted(abs(v - median) for v in values)
    dmid = len(deviations) // 2
    mad = (
        deviations[dmid]
        if len(deviations) % 2
        else (deviations[dmid - 1] + deviations[dmid]) / 2
    )
    return median, mad


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--observations",
        type=Path,
        required=True,
        help="Prepared row-level Parquet table",
    )
    p.add_argument(
        "--split-manifest",
        type=Path,
        required=True,
        help="CSV with sequence, split_cluster_id, split",
    )
    p.add_argument("--property-metadata", type=Path, required=True)
    p.add_argument(
        "--fasta",
        type=Path,
        help="Optional conditioning corpus FASTA; restrict output rows to these sequences",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Create-once measured JSONL condition table",
    )
    p.add_argument("--min-panel-coverage", type=float, default=0.8)
    p.add_argument("--min-group-coverage", type=float, default=1.0)
    args = p.parse_args()
    if not 0 < args.min_panel_coverage <= 1 or not 0 < args.min_group_coverage <= 1:
        p.error("coverage thresholds must be in (0, 1]")
    metadata = json.loads(args.property_metadata.read_text(encoding="utf-8"))
    semantics = metadata.get("semantic_definition")
    if not isinstance(semantics, dict):
        raise ValueError("Property metadata must define panel activity semantics")
    semantics = panel_semantic_definition(
        semantics["ranking_strains"],
        semantics["strain_groups"],
        semantics["activity_threshold_log2_mic"],
        semantics["activity_temperature"],
        semantics["broad_spectrum_aggregation"],
    )
    split_rows = list(
        csv.DictReader(args.split_manifest.open(encoding="utf-8", newline=""))
    )
    split_by_sequence: dict[str, tuple[str, str]] = {}
    cluster_splits: dict[str, str] = {}
    for row in split_rows:
        seq = canonical_sequence(row["sequence"])
        cluster, split = str(row["split_cluster_id"]), str(row["split"])
        if seq in split_by_sequence and split_by_sequence[seq] != (cluster, split):
            raise ValueError(f"Conflicting sequence identity/split assignment: {seq}")
        if cluster in cluster_splits and cluster_splits[cluster] != split:
            raise ValueError(f"Homology cluster crosses splits: {cluster}")
        split_by_sequence[seq] = (cluster, split)
        cluster_splits[cluster] = split
    table = pq.read_table(args.observations).to_pylist()
    grouped: dict[str, list[dict]] = defaultdict(list)
    identity_by_hash: dict[str, str] = {}
    for row in table:
        seq = canonical_sequence(row["sequence"])
        digest = row.get("sequence_sha256")
        if not digest:
            raise ValueError(f"Prepared observations require sequence_sha256: {seq}")
        if digest != _sha(seq):
            raise ValueError(f"Sequence SHA-256 mismatch: {seq}")
        if digest in identity_by_hash and identity_by_hash[digest] != seq:
            raise ValueError("Conflicting sequence identities share a hash")
        identity_by_hash[digest] = seq
        if seq not in split_by_sequence:
            raise ValueError(f"Observation sequence missing from split manifest: {seq}")
        cluster, split = split_by_sequence[seq]
        if row.get("split") and row["split"] != split:
            raise ValueError(f"Observation/split-manifest mismatch: {seq}")
        if row.get("split_cluster_id") and row["split_cluster_id"] != cluster:
            raise ValueError(f"Observation cluster mismatch: {seq}")
        if split == "train":
            grouped[seq].append(row)
    if args.fasta:
        from amp_ctmc_2027.data.fasta_io import FastaRepository

        allowed = set(FastaRepository(args.fasta.parent).read_sequences(args.fasta))
        grouped = {
            sequence: values
            for sequence, values in grouped.items()
            if sequence in allowed
        }
    rows, audit_rows = [], []
    panel = semantics["ranking_strains"]
    for seq, replicates in sorted(grouped.items()):
        values = {name: None for name in CONDITION_NAMES}
        observed = {name: False for name in CONDITION_NAMES}
        provenance = {name: "missing" for name in CONDITION_NAMES}
        uncertainty = {name: None for name in CONDITION_NAMES}
        audit = {
            "assay_count": len(replicates),
            "strain_coverage": {},
            "censoring": {},
            "fields": {},
        }
        amp_values = [
            float(r["is_amp"])
            for r in replicates
            if r.get("is_amp") is not None
            and r.get("label_provenance", "measured_public_database")
            == "measured_public_database"
        ]
        if amp_values:
            values["amp_probability"], uncertainty["amp_probability"] = _robust(
                amp_values
            )
            observed["amp_probability"], provenance["amp_probability"] = (
                True,
                "measured",
            )
        hemo_values = [
            float(r["is_hemolytic"])
            for r in replicates
            if r.get("is_hemolytic") is not None
            and r.get("label_provenance", "measured_public_database")
            == "measured_public_database"
        ]
        if hemo_values:
            values["hemolysis_probability"], uncertainty["hemolysis_probability"] = (
                _robust(hemo_values)
            )
            observed["hemolysis_probability"], provenance["hemolysis_probability"] = (
                True,
                "measured",
            )
        mic_group: dict[str, list[tuple]] = defaultdict(list)
        for row in replicates:
            strain = row.get("strain") or row.get("panel_target")
            if (
                strain in panel
                and row.get("mic_uM") is not None
                and row.get("label_provenance", "measured_public_database")
                == "measured_public_database"
            ):
                mic_group[strain].append(
                    (_mic_probability_interval(row, semantics), row)
                )
        strain_p, strain_mic, strain_uncertainty, strain_exact_mic = {}, {}, {}, {}
        for strain, observations in mic_group.items():
            pmedian, pmad = _robust([x[0][0] for x in observations])
            lower, _ = _robust([x[0][1] for x in observations])
            upper, _ = _robust([x[0][2] for x in observations])
            strain_p[strain] = pmedian
            epsilon = 1e-12
            logit = math.log(
                max(epsilon, min(1 - epsilon, pmedian))
                / max(epsilon, 1 - min(1 - epsilon, pmedian))
            )
            strain_mic[strain] = (
                semantics["activity_threshold_log2_mic"]
                - semantics["activity_temperature"] * logit
            )
            strain_uncertainty[strain] = {
                "mad": pmad,
                "probability_lower": lower,
                "probability_upper": upper,
                "assay_count": len(observations),
                "censoring_counts": {
                    state: sum(item[0][3] == state for item in observations)
                    for state in ("none", "left", "right", "interval")
                },
                "observations": [
                    {
                        "censor": item[3],
                        "bounds": item[4],
                        "mic_uM": row.get("mic_uM"),
                        "mic_upper_uM": row.get("mic_upper_uM"),
                        "source_record_id": row.get("source_record_id"),
                    }
                    for item, row in observations
                ],
            }
            exact_logs = [
                math.log2(float(row["mic_uM"]))
                for _, row in mic_group[strain]
                if (row.get("mic_censor") or "none") == "none"
            ]
            if len(exact_logs) == len(observations):
                strain_exact_mic[strain] = _robust(exact_logs)[0]
        audit["strain_coverage"] = {
            "observed": len(strain_p),
            "panel": len(panel),
            "fraction": len(strain_p) / len(panel),
        }
        audit["censoring"] = {
            strain: data["censoring_counts"]
            for strain, data in strain_uncertainty.items()
        }
        audit["group_coverage"] = {
            group: {
                "observed": sum(strain in strain_p for strain in members),
                "total": len(members),
                "fraction": sum(strain in strain_p for strain in members) / len(members)
                if members
                else 0.0,
            }
            for group, members in semantics["strain_groups"].items()
        }
        audit["fields"]["mic_by_strain"] = strain_uncertainty
        panel_coverage = len(strain_p) / len(panel)
        if strain_p:
            semantic_result = compute_panel_condition_semantics(
                strain_mic,
                {
                    **semantics,
                    "ranking_strains": list(strain_p),
                    "strain_groups": {
                        g: [s for s in semantics["strain_groups"][g] if s in strain_p]
                        for g in ("gram_negative", "gram_positive", "mdr")
                    },
                },
            )
            audit.update(semantic_result["audit"])
            if panel_coverage >= args.min_panel_coverage:
                values["broad_spectrum_probability"] = semantic_result["conditions"][
                    "broad_spectrum_probability"
                ]
                (
                    observed["broad_spectrum_probability"],
                    provenance["broad_spectrum_probability"],
                ) = True, "derived"
                uncertainty["broad_spectrum_probability"] = max(
                    (x["mad"] for x in strain_uncertainty.values()), default=0.0
                )
            if panel_coverage == 1.0:
                values["mean_activity_probability"] = semantic_result["conditions"][
                    "mean_activity_probability"
                ]
                (
                    observed["mean_activity_probability"],
                    provenance["mean_activity_probability"],
                ) = True, "derived"
                uncertainty["mean_activity_probability"] = max(
                    (x["mad"] for x in strain_uncertainty.values()), default=0.0
                )
                if len(strain_exact_mic) == len(panel):
                    values["predicted_log2_mic_summary"] = sum(
                        strain_exact_mic.values()
                    ) / len(panel)
                    (
                        observed["predicted_log2_mic_summary"],
                        provenance["predicted_log2_mic_summary"],
                    ) = True, "derived"
                    uncertainty["predicted_log2_mic_summary"] = max(
                        (data["mad"] for data in strain_uncertainty.values()),
                        default=0.0,
                    )
        for group, name in (
            ("gram_negative", "mean_gram_negative_activity_probability"),
            ("gram_positive", "mean_gram_positive_activity_probability"),
            ("mdr", "mean_mdr_activity_probability"),
        ):
            members = semantics["strain_groups"][group]
            coverage = (
                sum(member in strain_p for member in members) / len(members)
                if members
                else 0
            )
            if members and coverage >= args.min_group_coverage:
                vals = [
                    compute_panel_condition_semantics(
                        {s: strain_mic[s] for s in members if s in strain_mic},
                        {
                            **semantics,
                            "ranking_strains": [s for s in members if s in strain_mic],
                            "strain_groups": {
                                g: [] for g in ("gram_negative", "gram_positive", "mdr")
                            },
                        },
                    )["conditions"]["mean_activity_probability"]
                ]
                values[name] = vals[0]
                observed[name], provenance[name] = True, "derived"
                uncertainty[name] = sum(
                    strain_uncertainty[s]["mad"] for s in members
                ) / len(members)
        # HC50 is a numeric condition only for exact measurements; censored records stay in audit.
        hc_exact = [
            math.log2(float(r["hc50_uM"]))
            for r in replicates
            if r.get("hc50_uM")
            and float(r["hc50_uM"]) > 0
            and (r.get("hc50_censor") or "none") == "none"
            and r.get("label_provenance", "measured_public_database")
            == "measured_public_database"
        ]
        audit["fields"]["amp"] = {
            "assay_count": len(amp_values),
            "mad": uncertainty["amp_probability"],
        }
        audit["fields"]["hemolysis"] = {
            "assay_count": len(hemo_values),
            "mad": uncertainty["hemolysis_probability"],
        }
        audit["fields"]["hc50"] = {
            "assay_count": sum(r.get("hc50_uM") is not None for r in replicates),
            "censoring_counts": {
                state: sum(
                    (r.get("hc50_censor") or "none") == state
                    for r in replicates
                    if r.get("hc50_uM") is not None
                )
                for state in ("none", "left", "right", "interval")
            },
            "observations": [
                {
                    "value_uM": r.get("hc50_uM"),
                    "upper_uM": r.get("hc50_upper_uM"),
                    "censor": r.get("hc50_censor") or "none",
                    "source_record_id": r.get("source_record_id"),
                }
                for r in replicates
                if r.get("hc50_uM") is not None
            ],
        }
        if hc_exact:
            values["predicted_log2_hc50"], uncertainty["predicted_log2_hc50"] = _robust(
                hc_exact
            )
            observed["predicted_log2_hc50"], provenance["predicted_log2_hc50"] = (
                True,
                "measured",
            )
        row = {
            "sequence": seq,
            "condition_names": list(CONDITION_NAMES),
            "values": values,
            "observed": observed,
            "provenance": provenance,
            "uncertainty": uncertainty,
            "audit": audit,
        }
        rows.append(row)
        audit_rows.append(
            {"sequence": seq, **audit, "condition_censoring": audit["censoring"]}
        )
    if not rows:
        raise ValueError("No train-split measurements produced conditions")
    audit_path = args.output.with_suffix(args.output.suffix + ".audit.jsonl")
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    for path in (args.output, audit_path, manifest_path):
        if path.exists():
            raise FileExistsError(
                f"Measured condition artifacts are create-once: {path}"
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8"
    )
    audit_path.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in audit_rows),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "condition_names": list(CONDITION_NAMES),
        "semantic_definition": semantics,
        "source_table_sha256": table_sha256(args.observations),
        "split_manifest_sha256": table_sha256(args.split_manifest),
        "property_metadata_sha256": table_sha256(args.property_metadata),
        "table_sha256": table_sha256(args.output),
        "audit_sha256": table_sha256(audit_path),
        "fit_split": "train",
        "train_sequence_count": len(rows),
        "min_panel_coverage": args.min_panel_coverage,
        "min_group_coverage": args.min_group_coverage,
        "replicate_aggregation": "median with median absolute deviation uncertainty",
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
