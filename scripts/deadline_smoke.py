#!/usr/bin/env python3
"""Bounded end-to-end smoke workflow using isolated temporary artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def run(*args: str) -> None:
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    source_table = ROOT / "data/processed/deadline_observations.parquet"
    if not source_table.is_file():
        raise FileNotFoundError("Run the bootstrap stage before smoke")
    all_rows = pq.read_table(source_table).to_pylist()
    split_rows = list(
        csv.DictReader((ROOT / "data/processed/deadline_split_manifest.csv").open())
    )
    split_map = {row["sequence"]: row["split"] for row in split_rows}
    # Read FASTA without dependencies beyond the repository's normal parser.
    from amp_ctmc_2027.data.fasta_io import FastaRepository

    train_sequences = set(
        FastaRepository(ROOT).read_sequences(Path("data/training/training.fasta"))
    )
    train_ids = sorted(seq for seq in train_sequences if split_map.get(seq) == "train")[
        :12
    ]
    val_ids = sorted(
        {
            row["sequence"]
            for row in all_rows
            if row.get("is_amp") == 1.0 and split_map.get(row["sequence"]) == "val"
        }
    )[:3]
    selected = set(train_ids + val_ids)
    if len(selected) > 256 or len(train_ids) < 2 or not val_ids:
        raise RuntimeError(
            "Deadline smoke needs at least two train and one validation positive sequence"
        )
    rows = [row for row in all_rows if row["sequence"] in selected]
    with tempfile.TemporaryDirectory(prefix="amp-ctmc-deadline-smoke-") as directory:
        tmp = Path(directory)
        rows_path = tmp / "observations.parquet"
        pq.write_table(pa.Table.from_pylist(rows), rows_path)
        split_path = tmp / "splits.csv"
        with split_path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["sequence", "split_cluster_id", "split"])
            writer.writerows(
                (seq, hashlib.sha256(seq.encode()).hexdigest(), split_map[seq])
                for seq in sorted(selected)
            )
        fasta_path = tmp / "training.fasta"
        FastaRepository(tmp).write_fasta(fasta_path, train_ids, id_prefix="smoke")
        background_candidates = FastaRepository(ROOT).read_sequences(
            Path("data/training/background.fasta")
        )
        smoke_background_sequences = [
            sequence
            for sequence in background_candidates
            if split_map.get(sequence) == "train"
        ][:32]
        if not smoke_background_sequences:
            raise RuntimeError(
                "No train-split background sequences available for smoke"
            )
        background_path = tmp / "background.fasta"
        FastaRepository(tmp).write_fasta(
            background_path, smoke_background_sequences, id_prefix="background"
        )

        property_config = json.loads(
            (ROOT / "configs/deadline_property_35m.json").read_text()
        )
        property_config.update(
            {
                "model_name": str(ROOT / "data/pretrained/esm2_t12_35M_UR50D"),
                "epochs": 1,
                "batch_size": 8,
                "max_length": 64,
            }
        )
        property_config_path = tmp / "property.json"
        property_config_path.write_text(json.dumps(property_config))
        property_dir = tmp / "property"
        run(
            "scripts/train_property_model.py",
            "--config",
            str(property_config_path),
            "--data",
            str(rows_path),
            "--output",
            str(property_dir),
        )

        measured_path = tmp / "measured.jsonl"
        run(
            "scripts/build_measured_conditions.py",
            "--observations",
            str(rows_path),
            "--split-manifest",
            str(split_path),
            "--property-metadata",
            str(property_dir / "model_metadata.json"),
            "--fasta",
            str(fasta_path),
            "--output",
            str(measured_path),
            "--min-panel-coverage",
            "0.1",
            "--min-group-coverage",
            "0.1",
        )
        condition_path = tmp / "conditions.jsonl"
        run(
            "scripts/cache_property_conditions.py",
            "--checkpoint",
            str(property_dir),
            "--fasta",
            str(fasta_path),
            "--measured",
            str(measured_path),
            "--output",
            str(condition_path),
            "--batch-size",
            "8",
        )

        ctmc_config = json.loads(
            (ROOT / "configs/deadline_ctmc_conditioned.json").read_text()
        )
        ctmc_config.update(
            {
                "n_epochs": 1,
                "batch_size": 4,
                "device": "cpu",
                "training_fasta_path": str(fasta_path),
                "background_fasta_path": str(background_path),
                "cluster_split_manifest_path": str(split_path),
                "condition_table_path": str(condition_path),
                "condition_manifest_path": str(condition_path) + ".manifest.json",
                "property_metadata_path": str(property_dir / "model_metadata.json"),
                "checkpoint_dir": str(tmp / "ctmc"),
                "debug_overfit_samples": 8,
            }
        )
        ctmc_config_path = tmp / "ctmc.json"
        ctmc_config_path.write_text(json.dumps(ctmc_config))
        run("scripts/train_ctmc.py", "--config", str(ctmc_config_path))

        checkpoint = tmp / "ctmc/model.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError("CTMC smoke run did not write model.pt")
        # Build a test-only manifest recording all local hashes and the conditioning schema.
        cache_manifest = json.loads(
            Path(str(condition_path) + ".manifest.json").read_text()
        )
        smoke_manifest = {
            "smoke_schema_version": 1,
            "property_checkpoint_sha256": file_hash(property_dir / "model.pt"),
            "ctmc_checkpoint_sha256": file_hash(checkpoint),
            "condition_table_sha256": file_hash(condition_path),
            "condition_manifest": cache_manifest,
            "training_sequences": len(train_ids),
            "selected_sequences_at_most": 256,
        }
        (tmp / "smoke_manifest.json").write_text(
            json.dumps(smoke_manifest, indent=2) + "\n"
        )

        from amp_ctmc_2027.compliance import validate_fasta_files
        from amp_ctmc_2027.config import AMPConfig
        from amp_ctmc_2027.core import (
            AMPCanvasEncoder,
            CTMCDenoiser,
            ReverseGenerationConfig,
            SinSquaredSchedule,
            TauLeapingSampler,
        )
        from amp_ctmc_2027.data.fasta_io import FastaRepository
        from amp_ctmc_2027.official_identity import official_identity

        config = AMPConfig.from_json_file(ctmc_config_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = AMPCanvasEncoder()
        model = CTMCDenoiser.load(
            checkpoint, config, encoder.vocab_size, encoder.pad_idx, device
        ).eval()
        from amp_ctmc_2027.core import ConditionVector
        from amp_ctmc_2027.data.conditions import (
            load_condition_table,
            normalize_condition_values,
        )

        contract = torch.load(checkpoint, map_location="cpu", weights_only=True)[
            "conditioning_contract"
        ]
        condition_rows, _ = load_condition_table(
            condition_path, Path(str(condition_path) + ".manifest.json")
        )
        first = condition_rows[0]
        values = torch.tensor(
            [first["values"][name] or 0.0 for name in first["condition_names"]],
            dtype=torch.float32,
            device=device,
        )
        observed = torch.tensor(
            [first["observed"][name] for name in first["condition_names"]],
            dtype=torch.bool,
            device=device,
        )
        values = normalize_condition_values(
            values.unsqueeze(0),
            observed.unsqueeze(0),
            contract["condition_normalization"],
        ).squeeze(0)
        cond = ConditionVector(
            values.unsqueeze(0).expand(8, -1),
            observed.unsqueeze(0).expand(8, -1),
            provenance=("smoke",) * 8,
        )
        sampler = TauLeapingSampler(model, encoder, SinSquaredSchedule(), min_length=8)
        library = []
        for seed in range(42, 52):
            library.extend(
                sampler.sample_batch(
                    ReverseGenerationConfig(
                        steps=6,
                        temperature=1.0,
                        seed=seed,
                        conditions=cond,
                        cfg_scale=1.5,
                    ),
                    8,
                )
            )
            library = list(
                dict.fromkeys(
                    seq
                    for seq in library
                    if 8 <= len(seq) <= 50
                    and set(seq).issubset(set(encoder.vocab))
                    and seq not in train_sequences
                )
            )[:8]
            if len(library) == 8:
                break
        if len(library) < 4:
            raise RuntimeError(
                "Tiny CTMC sampler did not produce four unique valid smoke peptides"
            )
        output = tmp / "generate"
        repo = FastaRepository(tmp)
        repo.write_fasta(output / "library.fasta", library, id_prefix="seq")
        repo.write_fasta(output / "top.fasta", library[:2], id_prefix="seq")
        validate_fasta_files(
            output / "library.fasta",
            output / "top.fasta",
            FastaRepository(ROOT).read_sequences(Path("data/antibacterial.fasta")),
            list(train_sequences),
            identity_function=official_identity,
            expected_library_count=len(library),
            expected_top_count=2,
        )
        print(
            json.dumps(
                {
                    "smoke": "passed",
                    "property_epochs": 1,
                    "ctmc_epochs": 1,
                    "unique_sequences": len(selected),
                    "generated_library": len(library),
                    "top": 2,
                    "manifest": str(tmp / "smoke_manifest.json"),
                    "production_artifacts_modified": False,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
