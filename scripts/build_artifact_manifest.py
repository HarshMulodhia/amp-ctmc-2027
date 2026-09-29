#!/usr/bin/env python3
"""Build the fixed inference manifest from locally trained/exported artifacts."""

import json
import argparse
from pathlib import Path

from amp_ctmc_2027.data.dataset import AMPCanvasEncoder
from amp_ctmc_2027.submission import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def ref(path: Path, root: Path = ROOT) -> dict:
    digest = sha256_file(path)
    if len(digest) != 64 or digest == "0" * 64:
        raise ValueError(f"Refusing placeholder SHA-256 for {path}")
    return {"path": path.relative_to(root).as_posix(), "sha256": digest}


def main(root: Path = ROOT) -> None:
    root = root.resolve()
    template_path = root / "artifacts/manifest.example.json"
    manifest = json.loads(template_path.read_text(encoding="utf-8"))
    ctmc_checkpoint = root / "artifacts/ctmc/model.pt"
    ctmc_config_path = root / "artifacts/ctmc/config.json"
    property_dir = root / "artifacts/property"
    property_checkpoint = property_dir / "model.pt"
    property_metadata = property_dir / "model_metadata.json"
    training_path = root / "data/training/training.fasta"
    training_manifest = root / "data/training/dataset_manifest.json"
    reference_path = root / "data/antibacterial.fasta"
    backbone_files = [
        path for path in (property_dir / "backbone").rglob("*") if path.is_file()
    ]
    tokenizer_files = [
        path for path in (property_dir / "tokenizer").rglob("*") if path.is_file()
    ]
    required = [
        ctmc_checkpoint,
        ctmc_config_path,
        property_checkpoint,
        property_metadata,
        training_path,
        training_manifest,
        reference_path,
        *backbone_files,
        *tokenizer_files,
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Cannot build inference manifest; missing local artifacts: "
            + ", ".join(str(path.relative_to(root)) for path in missing)
        )
    if not backbone_files or not tokenizer_files:
        raise FileNotFoundError(
            "Local ESM backbone and tokenizer directories must not be empty"
        )
    artifacts = manifest["artifacts"]
    artifacts["ctmc_checkpoint"] = ref(ctmc_checkpoint, root)
    artifacts["property_checkpoint"] = ref(property_checkpoint, root)
    artifacts["property_metadata"] = ref(property_metadata, root)
    artifacts["property_files"] = [
        ref(path, root)
        for path in sorted(
            [p for p in (property_dir / "backbone").rglob("*") if p.is_file()]
            + [p for p in (property_dir / "tokenizer").rglob("*") if p.is_file()]
        )
    ]
    artifacts["training_sequences"] = ref(training_path, root)
    artifacts["training_data_manifest"] = ref(training_manifest, root)
    artifacts["reference_sequences"] = ref(reference_path, root)
    ctmc_config = json.loads(ctmc_config_path.read_text(encoding="utf-8"))
    ctmc_payload = __import__("torch").load(
        ctmc_checkpoint, map_location="cpu", weights_only=True
    )
    ctmc_contract = ctmc_payload.get("conditioning_contract", {})
    ctmc_config["vocab"] = AMPCanvasEncoder().vocab
    ctmc_config["max_length"] = 50
    manifest["ctmc"]["config"] = ctmc_config
    property_model = json.loads(property_metadata.read_text(encoding="utf-8"))
    semantic_definition = property_model.get("semantic_definition")
    if not isinstance(semantic_definition, dict):
        raise ValueError("Property metadata lacks semantic_definition")
    if ctmc_contract.get("semantic_definition") != semantic_definition:
        raise ValueError("Property semantic definition differs from CTMC checkpoint")
    normalization = ctmc_contract.get("condition_normalization")
    if not isinstance(normalization, dict):
        raise ValueError("CTMC checkpoint lacks condition normalization statistics")
    if ctmc_config.get("condition_normalization") != normalization:
        raise ValueError("CTMC config/checkpoint normalization mismatch")
    ctmc_config["condition_normalization"] = normalization
    ctmc_config["condition_semantics"] = semantic_definition
    strain_ids = property_model.get(
        "strain_to_id", property_model.get("strain_ids", {})
    )
    ranking_strains = property_model.get("ranking_strains", [])
    if not ranking_strains or set(ranking_strains).difference(strain_ids):
        raise ValueError(
            "Property metadata must define known, nonempty ranking_strains"
        )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(property_dir / "tokenizer"), local_files_only=True
    )
    manifest["property_model"].update(
        {
            "checkpoint_format_version": 1,
            "inference_max_length": int(property_model.get("max_tokenized_length", 64)),
            "inference_batch_size": int(
                property_model.get("inference_batch_size", 128)
            ),
            "strain_ids": strain_ids,
            "strain_to_id": strain_ids,
            "ranking_strains": ranking_strains,
            "strain_groups": property_model.get("strain_groups", {}),
            "semantic_definition": semantic_definition,
            "unknown_strain_id": property_model["unknown_strain_id"],
            "strain_dim": property_model["strain_dim"],
            "hidden_size": property_model["hidden_size"],
            "residue_token_ids": {
                residue: int(tokenizer.convert_tokens_to_ids(residue))
                for residue in AMPCanvasEncoder().vocab
            },
        }
    )
    manifest["ranking_strains"] = ranking_strains
    manifest["strain_groups"] = property_model.get("strain_groups", {})
    manifest["condition_semantics"] = semantic_definition
    manifest["conditions"]["semantic_definition"] = semantic_definition
    manifest["condition_cache"] = {
        "table_sha256": ctmc_contract.get("condition_table_sha256"),
        "manifest_sha256": ctmc_contract.get("condition_manifest_sha256"),
        "semantic_definition": semantic_definition,
    }
    manifest["scoring_weights"] = manifest.get("scoring_weights", {})
    if not manifest["scoring_weights"]:
        raise ValueError("manifest.example.json must configure scoring_weights")
    manifest["generation"]["library_count"] = 50000
    manifest["generation"]["top_count"] = 100
    generation_config = root / "configs/deadline_generation.json"
    if generation_config.is_file():
        deadline_generation = json.loads(generation_config.read_text(encoding="utf-8"))
        manifest["generation"].update(
            {
                "oversampling_factor": deadline_generation["oversampling_factor"],
                "batch_size": deadline_generation["generation_batch_size"],
                "steps": deadline_generation["ctmc_steps"],
                "cfg_scale": (
                    0.0
                    if ctmc_contract.get("training_mode") == "unconditional"
                    else deadline_generation["cfg_scale"]
                ),
            }
        )
        manifest["scoring_weights"] = deadline_generation["scoring_weights"]
        if ctmc_contract.get("training_mode") == "unconditional":
            manifest["conditions"]["values"] = [0.0] * len(
                manifest["conditions"]["condition_names"]
            )
            manifest["conditions"]["observed"] = [False] * len(
                manifest["conditions"]["condition_names"]
            )
        else:
            manifest["conditions"]["values"] = [
                deadline_generation["targets"][name]
                for name in manifest["conditions"]["condition_names"]
            ]
        manifest["default_seed"] = int(deadline_generation["seed"])
    else:
        manifest["default_seed"] = int(manifest.get("default_seed", 42))
    manifest["official_identity_source"]["repository_template_commit"] = (
        "987afe5c9f73c82d39b0def7acdc56dec31a0132"
    )
    manifest["training_data_manifest_reference"] = "artifacts.training_data_manifest"
    output = root / "artifacts/manifest.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    print(f"Wrote {output.relative_to(root)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    main(parser.parse_args().root)
