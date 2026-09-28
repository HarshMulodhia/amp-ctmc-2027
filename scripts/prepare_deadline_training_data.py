#!/usr/bin/env python3
"""Prepare the pinned deadline sources into observations, splits and FASTAs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from amp_ctmc_2027.data.deadline import (
    DEADLINE_STRAIN_COLUMNS,
    deterministic_sequence_split,
    grampa_log10_mic_to_um,
    hc50_log10_to_um,
    hemopi2_hemolysis_label,
)

ROOT = Path(__file__).resolve().parents[1]
AA = set("ACDEFGHIKLMNPQRSTVWY")
PANEL = DEADLINE_STRAIN_COLUMNS


def norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def field(row: dict, *candidates: str) -> str | None:
    normalized = {norm(str(key)): value for key, value in row.items()}
    for candidate in candidates:
        value = normalized.get(norm(candidate))
        if value not in (None, ""):
            return str(value).strip()
    return None


def fasta(path: Path) -> list[str]:
    seqs, current = [], []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(">"):
            if current:
                seqs.append("".join(current).upper())
            current = []
        else:
            current.append("".join(line.split()))
    if current:
        seqs.append("".join(current).upper())
    return seqs


def valid_sequence(value: str | None) -> tuple[str | None, str | None]:
    seq = "" if value is None else "".join(value.split()).upper()
    if not seq:
        return None, "empty"
    if any(char not in AA for char in seq):
        return None, "noncanonical_residue_or_modification"
    if not 8 <= len(seq) <= 50:
        return None, "length_out_of_range"
    return seq, None


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    raw = ROOT / "data/raw/deadline"
    download_manifest_path = raw / "download_manifest.json"
    if not download_manifest_path.is_file():
        raise FileNotFoundError("Run scripts/bootstrap_deadline_data.py first")
    download_manifest = json.loads(download_manifest_path.read_text())
    for item in download_manifest["sources"]:
        if (
            item.get("expected_sha256")
            and item["actual_sha256"] != item["expected_sha256"]
        ):
            raise ValueError(f"Unverified downloaded source: {item['filename']}")
        if (
            item["id"] != "organizer_reference"
            and sha(raw / item["filename"]) != item["expected_sha256"]
        ):
            raise ValueError(
                f"Downloaded file changed after verification: {item['filename']}"
            )

    rows: list[dict] = []
    rejected: list[dict] = []
    uncertain_mods: list[dict] = []
    counts = Counter()

    def add(row: dict, original: dict | None = None) -> None:
        seq, reason = valid_sequence(row.get("sequence"))
        if reason:
            rejected.append(
                {
                    "source": row.get("source"),
                    "sequence": row.get("sequence"),
                    "reason": reason,
                }
            )
            return
        row["sequence"] = seq
        if original:
            row["source_record_id"] = original.get("id") or original.get("entry_id")
        rows.append(row)
        counts[row["source"]] += 1

    amp_rows = read_csv(raw / "amp_diffusion_train_eval.csv")
    for idx, source in enumerate(amp_rows):
        seq = field(source, "sequence", "seq", "peptide")
        for strain_id, source_name in PANEL.items():
            mic = field(source, source_name)
            if mic is None:
                # Source uses AIG spelling; this alias is explicit and audited below.
                continue
            try:
                value = float(mic)
            except ValueError:
                continue
            add(
                {
                    "sequence": seq,
                    "source": "amp_diffusion_train_eval",
                    "source_record_id": str(idx),
                    "is_amp": 1.0,
                    "strain": strain_id,
                    "panel_target": strain_id,
                    "mic_value": value,
                    "mic_unit_original": "uM source pseudo-prediction",
                    "mic_uM": value,
                    "mic_censor": "none",
                    "label_provenance": "pseudo_apex_or_source_model",
                }
            )

    grampa_rows = read_csv(raw / "grampa.csv")
    for idx, source in enumerate(grampa_rows):
        seq = field(source, "sequence", "seq")
        modified = field(source, "is_modified")
        amidated = field(source, "has_cterminal_amidation")
        unusual_flag = field(source, "has_unusual_modification", "unusual_modification")
        modification_list = field(source, "modification", "modifications")

        def known_false(value: str | None) -> bool:
            return value is not None and value.strip().lower() in {
                "false",
                "0",
                "no",
            }

        unusual_ok = (
            known_false(unusual_flag)
            if unusual_flag is not None
            else modification_list is not None
            and modification_list.strip().lower() in {"none", "nan", "", "[]"}
        )
        if not known_false(modified) or not known_false(amidated) or not unusual_ok:
            uncertain_mods.append(
                {
                    "source": "grampa",
                    "row": idx,
                    "sequence": seq,
                    "is_modified": modified,
                    "has_cterminal_amidation": amidated,
                    "modification": modification_list,
                    "has_unusual_modification": unusual_flag,
                }
            )
            continue
        value = field(source, "value")
        try:
            original_value = float(value)
            mic_um = grampa_log10_mic_to_um(original_value)
        except (TypeError, ValueError, OverflowError):
            continue
        if not math.isfinite(mic_um) or mic_um <= 0:
            continue
        add(
            {
                "sequence": seq,
                "source": "grampa",
                "source_record_id": str(idx),
                "is_amp": 1.0,
                "organism": field(source, "species", "organism", "bacterium"),
                "strain": field(source, "strain"),
                "mic_value": original_value,
                "mic_unit_original": "log10(MIC in micromolar)",
                "mic_uM": mic_um,
                "mic_censor": "none",
                "label_provenance": "measured_public_database",
                "grampa_transform": "mic_uM=10**value",
            }
        )

    hc_rows = read_csv(raw / "Cleaned_hemolytic_data.csv")
    for idx, source in enumerate(hc_rows):
        seq = field(source, "sequence", "seq")
        value = field(source, "log10_HC50")
        try:
            hc50 = hc50_log10_to_um(float(value))
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(hc50) and hc50 > 0:
            add(
                {
                    "sequence": seq,
                    "source": "cleaned_hc50",
                    "source_record_id": str(idx),
                    "hc50_uM": hc50,
                    "hc50_censor": "none",
                    "label_provenance": "measured_public_database",
                    "hc50_original_log10_uM": float(value),
                    "hc50_transform": "hc50_uM=10**log10_HC50",
                }
            )

    for source_id, filename in (
        ("hemopi2_cross_val", "hemopi2_cross_val.csv"),
        ("hemopi2_independent", "hemopi2_independent.csv"),
    ):
        for idx, source in enumerate(read_csv(raw / filename)):
            seq = field(source, "SEQUENCE")
            concentration = field(source, "μM", "µM", "uM", "micromolar")
            label = field(source, "label")
            try:
                hc50 = float(concentration) if concentration is not None else None
                hemo = (
                    hemopi2_hemolysis_label(float(label)) if label is not None else None
                )
            except ValueError:
                continue
            # HemoPI2's documented label is preserved verbatim as its hemolysis label.
            add(
                {
                    "sequence": seq,
                    "source": source_id,
                    "source_record_id": str(idx),
                    "hc50_uM": hc50 if hc50 and hc50 > 0 else None,
                    "hc50_censor": "none" if hc50 and hc50 > 0 else None,
                    "is_hemolytic": hemo,
                    "label_provenance": "measured_public_database",
                    "hemopi2_label_direction": "source label retained without reversal",
                }
            )

    for filename, label, source_id in (
        ("AMPlify_AMP_train_common.fa", 1.0, "amplify_amp_positive"),
        ("AMPlify_non_AMP_train_balanced.fa", 0.0, "amplify_non_amp"),
    ):
        for idx, seq in enumerate(fasta(raw / filename)):
            canonical, reason = valid_sequence(seq)
            if reason:
                rejected.append(
                    {"source": source_id, "sequence": seq, "reason": reason}
                )
            else:
                add(
                    {
                        "sequence": canonical,
                        "source": source_id,
                        "source_record_id": str(idx),
                        "is_amp": label,
                        "label_provenance": "public_dataset_label",
                    }
                )

    all_sequences = sorted({row["sequence"] for row in rows})

    # Global sequence-level hash split. Every observation for a sequence inherits it.
    split_map = {seq: deterministic_sequence_split(seq, 42) for seq in all_sequences}
    for row in rows:
        row["sequence_sha256"] = hashlib.sha256(row["sequence"].encode()).hexdigest()
        row["split_cluster_id"] = row["sequence_sha256"]
        row["split"] = split_map[row["sequence"]]
        row.setdefault("is_linear", True)

    output = ROOT / "data/processed/deadline_observations.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet_columns = sorted({key for row in rows for key in row})
    complete_rows = [{key: row.get(key) for key in parquet_columns} for row in rows]
    pq.write_table(pa.Table.from_pylist(complete_rows), output)
    split_path = ROOT / "data/processed/deadline_split_manifest.csv"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    with split_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sequence", "split_cluster_id", "split"])
        writer.writerows(
            (seq, hashlib.sha256(seq.encode()).hexdigest(), split_map[seq])
            for seq in all_sequences
        )
    negative = sorted(
        {
            row["sequence"]
            for row in rows
            if row.get("is_amp") == 0.0 and row["source"] == "amplify_non_amp"
        }
    )
    labels_by_sequence: dict[str, set[float]] = defaultdict(set)
    for row in rows:
        if row.get("is_amp") is not None:
            labels_by_sequence[row["sequence"]].add(float(row["is_amp"]))
    conflicts = [
        {
            "sequence": sequence,
            "kind": "contradictory_amp_labels",
            "labels": sorted(labels),
        }
        for sequence, labels in sorted(labels_by_sequence.items())
        if len(labels) > 1
    ]
    conflicted = {item["sequence"] for item in conflicts}
    negative = sorted(set(negative) - conflicted)
    positive = sorted(
        {
            row["sequence"]
            for row in rows
            if row.get("is_amp") == 1.0 and row["source"] != "amplify_non_amp"
        }
        - set(negative)
        - conflicted
    )

    def write_fasta(path: Path, seqs: list[str], label: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(f">{label}_{i:06d}\n{seq}\n" for i, seq in enumerate(seqs)),
            encoding="utf-8",
        )

    write_fasta(ROOT / "data/training/training.fasta", positive, "deadline_amp")
    write_fasta(ROOT / "data/training/background.fasta", negative, "deadline_non_amp")
    rejected_path = ROOT / "data/audit/deadline_rejected.csv"
    conflicts_path = ROOT / "data/audit/deadline_conflicts.csv"
    for path, records in ((rejected_path, rejected), (conflicts_path, conflicts)):
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({key for rec in records for key in rec}) or [
            "source",
            "sequence",
            "reason",
        ]
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(records)
    (ROOT / "data/audit/deadline_uncertain_modifications.csv").write_text(
        "source,row,sequence,is_modified,has_cterminal_amidation,modification\n"
        + "".join(
            ",".join(
                str(item.get(key, ""))
                for key in (
                    "source",
                    "row",
                    "sequence",
                    "is_modified",
                    "has_cterminal_amidation",
                    "modification",
                )
            )
            + "\n"
            for item in uncertain_mods
        ),
        encoding="utf-8",
    )
    (ROOT / "data/audit/deadline_source_counts.json").write_text(
        json.dumps(
            {
                "rows_by_source": dict(counts),
                "observation_rows": len(rows),
                "unique_sequences": len(all_sequences),
                "positive_sequences": len(positive),
                "background_sequences": len(negative),
                "rejected_rows": len(rejected),
                "uncertain_modification_rows": len(uncertain_mods),
                "strain_mapping": PANEL,
                "explicit_aliases": {
                    "E. coli AIG221": "E_coli_AIC221",
                    "E. coli AIG222": "E_coli_AIC222",
                },
                "panel_note": "11-strain proxy panel; not the complete official 20-strain wet-lab panel",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    source_hashes = {
        item["id"]: item["actual_sha256"] for item in download_manifest["sources"]
    }
    dataset_manifest = {
        "schema_version": 1,
        "source_hashes": source_hashes,
        "observation_table_sha256": sha(output),
        "split_manifest_sha256": sha(split_path),
        "split_method": "deterministic_exact_sequence_hash",
        "homology_split": False,
        "seed": 42,
        "split_proportions": {"train": 0.8, "val": 0.1, "test": 0.1},
        "split_counts": dict(Counter(split_map.values())),
        "observation_rows": len(rows),
        "unique_sequences": len(all_sequences),
        "positive_sequences": len(positive),
        "background_sequences": len(negative),
        "strain_groups": {
            "gram_negative": list(PANEL)[:7],
            "gram_positive": list(PANEL)[7:],
            "mdr": [
                "E_coli_AIC222",
                "S_aureus_ATCC_BAA1556",
                "E_faecalis_ATCC700802",
                "E_faecium_ATCC700221",
            ],
        },
        "panel_note": "11-strain proxy panel; not the complete official 20-strain wet-lab panel",
        "alias_audit": "AIG221/AIG222 source names explicitly mapped to official AIC221/AIC222 identifiers",
        "measured_precedence": "measured_public_database supersedes pseudo_apex_or_source_model at condition aggregation; raw observations retained",
    }
    (ROOT / "data/training/dataset_manifest.json").write_text(
        json.dumps(dataset_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "observation_rows": len(rows),
                "unique_sequences": len(all_sequences),
                "positive_sequences": len(positive),
                "background_sequences": len(negative),
                "paths": [
                    str(output),
                    str(split_path),
                    "data/training/training.fasta",
                    "data/training/background.fasta",
                    "data/training/dataset_manifest.json",
                    str(rejected_path),
                    str(conflicts_path),
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
