#!/usr/bin/env python3
"""Build immutable offline conditioning pseudo-labels from a local property model."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from amp_ctmc_2027.core import CONDITION_NAMES
from amp_ctmc_2027.data.conditions import (
    canonical_sequence,
    measured_overrides,
    write_condition_table,
)
from amp_ctmc_2027.data.fasta_io import FastaRepository
from amp_ctmc_2027.data.manifests import sha256_file
from amp_ctmc_2027.models.esm_multitask import ESMMultiTaskPredictor
from amp_ctmc_2027.submission import panel_component_scores


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Exported local property artifact directory",
    )
    p.add_argument("--fasta", type=Path, required=True)
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL path; artifacts are create-once",
    )
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument(
        "--measured",
        type=Path,
        help="Optional JSONL condition rows; measured values override predictions",
    )
    args = p.parse_args()
    if args.batch_size < 1:
        p.error("--batch-size must be positive")
    checkpoint = args.checkpoint
    metadata_path, checkpoint_path = (
        checkpoint / "model_metadata.json",
        checkpoint / "model.pt",
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    ranking_strains = metadata.get("ranking_strains", [])
    if not ranking_strains or set(ranking_strains) - set(metadata["strain_ids"]):
        raise ValueError("Export metadata must contain the configured ranking_strains")
    semantic_definition = metadata.get("semantic_definition")
    if not isinstance(semantic_definition, dict):
        raise ValueError(
            "Property metadata lacks the canonical activity semantic definition"
        )
    sequences = [
        canonical_sequence(s)
        for s in FastaRepository(Path(".")).read_sequences(args.fasta)
    ]
    if len(set(sequences)) != len(sequences):
        raise ValueError("Training FASTA contains duplicate canonical sequences")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = ESMMultiTaskPredictor.from_local_artifacts(
        checkpoint_path,
        checkpoint / "backbone",
        checkpoint / "tokenizer",
        metadata_path,
        device,
    )
    per_strain: dict[str, list[float]] = {strain: [] for strain in ranking_strains}
    per_strain_std: dict[str, list[float]] = {strain: [] for strain in ranking_strains}
    amp, hemo, hc50, hc50_std = [], [], [], []
    with torch.inference_mode():
        for start in range(0, len(sequences), args.batch_size):
            batch = sequences[start : start + args.batch_size]
            encoded = tokenizer(
                batch,
                truncation=True,
                max_length=int(metadata.get("inference_max_length", 64)),
                padding=True,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            panel_ids = torch.tensor(
                [int(metadata["strain_ids"][name]) for name in ranking_strains],
                dtype=torch.long,
                device=device,
            )
            out = model.forward_panel(
                encoded["input_ids"], encoded["attention_mask"], panel_ids
            )
            panel_mic = out.mic_mean.cpu().numpy()
            panel_std = out.mic_log_std.exp().cpu().numpy()
            for index, strain_name in enumerate(ranking_strains):
                per_strain[strain_name].extend(panel_mic[:, index].tolist())
                per_strain_std[strain_name].extend(panel_std[:, index].tolist())
            amp.extend(out.amp_probability.cpu().tolist())
            hemo.extend(out.hemolysis_probability.cpu().tolist())
            hc50.extend(out.hc50_mean.cpu().tolist())
            hc50_std.extend(out.hc50_log_std.exp().cpu().tolist())
    groups = metadata.get("strain_groups", {})
    measured = {}
    if args.measured:
        measured_manifest_path = args.measured.with_suffix(
            args.measured.suffix + ".manifest.json"
        )
        measured_manifest = json.loads(
            measured_manifest_path.read_text(encoding="utf-8")
        )
        if measured_manifest.get("table_sha256") != sha256_file(args.measured):
            raise ValueError("Measured condition table SHA-256 mismatch")
        if measured_manifest.get("semantic_definition") != semantic_definition:
            raise ValueError("Measured and teacher semantic definitions differ")
        for line in args.measured.read_text(encoding="utf-8").splitlines():
            if line:
                row = json.loads(line)
                seq = canonical_sequence(row["sequence"])
                if seq in measured:
                    raise ValueError(f"Duplicate measured condition row: {seq}")
                measured[seq] = row
        sequence_set = set(sequences)
        measured = {seq: row for seq, row in measured.items() if seq in sequence_set}
        
    rows = []
    for i, seq in enumerate(sequences):
        mic = {
            name: np.asarray(per_strain[name][i], dtype=np.float64)
            for name in ranking_strains
        }
        panel = panel_component_scores(
            {name: np.asarray([mic[name]]) for name in ranking_strains},
            np.asarray([amp[i]]),
            np.asarray([hemo[i]]),
            np.asarray([hc50[i]]),
            ranking_strains,
            groups,
            semantic_definition,
        )
        vals = {
            "amp_probability": float(amp[i]),
            **{
                name: float(panel[name][0]) if name in panel else None
                for name in CONDITION_NAMES
                if name != "amp_probability"
            },
            "hemolysis_probability": float(hemo[i]),
            "predicted_log2_hc50": float(hc50[i]),
        }
        derived_fields = set(CONDITION_NAMES) - {
            "amp_probability",
            "hemolysis_probability",
            "predicted_log2_hc50",
        }
        sources = {
            name: (
                "missing"
                if value is None
                else "derived"
                if name in derived_fields
                else "teacher"
            )
            for name, value in vals.items()
        }
        obs = {name: value is not None for name, value in vals.items()}
        uncertainty = {name: None for name in CONDITION_NAMES}
        uncertainty["amp_probability"] = float(np.sqrt(amp[i] * (1 - amp[i])))
        uncertainty["hemolysis_probability"] = float(np.sqrt(hemo[i] * (1 - hemo[i])))
        uncertainty["predicted_log2_hc50"] = float(hc50_std[i])
        for name in derived_fields:
            if name.endswith("_probability"):
                pvalue = vals[name]
                sigma = max(per_strain_std[s][i] for s in ranking_strains)
                uncertainty[name] = float(
                    pvalue
                    * (1 - pvalue)
                    * sigma
                    / semantic_definition["activity_temperature"]
                )
            elif name == "predicted_log2_mic_summary":
                uncertainty[name] = float(
                    np.sqrt(
                        np.mean(
                            np.square([per_strain_std[s][i] for s in ranking_strains])
                        )
                    )
                    / len(ranking_strains)
                )
        audit = {
            "worst_predicted_log2_mic": float(panel["worst_log2_mic"][0]),
            "minimum_activity_probability": float(
                panel["minimum_activity_probability"][0]
            ),
            "panel_coverage": float(panel["panel_coverage"][0]),
        }
        audit["per_strain_predicted_log2_mic"] = {
            s: float(mic[s]) for s in ranking_strains
        }
        audit["per_strain_activity_probability"] = {
            s: float(panel[f"activity_probability__{s}"][0]) for s in ranking_strains
        }
        audit["per_strain_mic_uncertainty"] = {
            s: float(per_strain_std[s][i]) for s in ranking_strains
        }
        merged = measured_overrides(
            {
                "values": vals,
                "observed": obs,
                "provenance": sources,
                "uncertainty": uncertainty,
            },
            measured.get(seq),
        )
        if seq in measured:
            audit["measured_audit"] = measured[seq].get("audit", {})
        rows.append(
            {
                "sequence": seq,
                "condition_names": list(CONDITION_NAMES),
                **merged,
                "audit": audit,
            }
        )
    panel_hash = hashlib.sha256(
        json.dumps(ranking_strains, separators=(",", ":")).encode()
    ).hexdigest()
    write_condition_table(
        args.output,
        rows,
        {
            "kind": "conditioning_pseudo_labels",
            "property_checkpoint_sha256": sha256_file(checkpoint_path),
            "property_metadata_sha256": sha256_file(metadata_path),
            "ranking_strains": ranking_strains,
            "semantic_definition": semantic_definition,
            "strain_panel_sha256": panel_hash,
            "source_fasta_sha256": sha256_file(args.fasta),
            "measured_table_sha256": sha256_file(args.measured)
            if args.measured
            else None,
        },
    )


if __name__ == "__main__":
    main()
