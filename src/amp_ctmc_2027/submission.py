"""Offline, manifest-validated production generation and submission validation."""

from __future__ import annotations

import hashlib
import csv
import json
import logging
import math
import os
import random
from pathlib import Path

import numpy as np
import torch

from amp_ctmc_2027.compliance import validate_fasta_files
from amp_ctmc_2027.config import AMPConfig, set_global_determinism
from amp_ctmc_2027.core import (
    CONDITION_NAMES,
    ConditionVector,
    CTMCDenoiser,
    ReverseGenerationConfig,
    SinSquaredSchedule,
    TauLeapingSampler,
)
from amp_ctmc_2027.data.dataset import AMPCanvasEncoder
from amp_ctmc_2027.data.conditions import (
    compute_panel_condition_semantics,
    normalize_condition_values,
    panel_semantic_definition,
)
from amp_ctmc_2027.data.fasta_io import FastaRepository
from amp_ctmc_2027.models.esm_multitask import ESMMultiTaskPredictor
from amp_ctmc_2027.official_identity import DEFAULT_THRESHOLD, official_identity

logger = logging.getLogger(__name__)
MANIFEST_SCHEMA_VERSION = 1
DEFAULT_MANIFEST = Path("artifacts/manifest.json")
CANONICAL_VOCAB = "ACDEFGHIKLMNPQRSTVWY"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _root_path(root: Path, value: str) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError(f"Artifact path escapes repository root: {value}")
    return resolved


def validate_manifest(root: Path, manifest_path: Path) -> dict:
    """Read and validate every referenced artifact/hash before model loading."""
    resolved_manifest = _root_path(root, str(manifest_path))
    if not resolved_manifest.is_file():
        raise FileNotFoundError(
            f"Missing artifact manifest: {resolved_manifest}. Copy and complete artifacts/manifest.example.json."
        )
    manifest = json.loads(resolved_manifest.read_text(encoding="utf-8"))
    if manifest.get("artifact_schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("Unsupported artifact manifest schema version")
    if manifest.get("default_seed") is None:
        raise ValueError("Artifact manifest must define default_seed")
    identity_source = manifest.get("official_identity_source", {})
    if (
        identity_source.get("repository_template_commit")
        != "987afe5c9f73c82d39b0def7acdc56dec31a0132"
        or identity_source.get("threshold") != DEFAULT_THRESHOLD
        or identity_source.get("reject_semantics") != "strictly greater than"
    ):
        raise ValueError("Manifest official identity rule/provenance is incompatible")
    refs: list[dict] = []
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Artifact manifest must contain an artifacts object")
    required_artifacts = {
        "ctmc_checkpoint",
        "property_checkpoint",
        "property_metadata",
        "property_files",
        "training_sequences",
        "training_data_manifest",
        "reference_sequences",
    }
    missing_entries = required_artifacts.difference(artifacts)
    if missing_entries:
        raise ValueError(
            f"Artifact manifest is missing required entries: {sorted(missing_entries)}"
        )
    fixed_artifact_paths = {
        "ctmc_checkpoint": "artifacts/ctmc/model.pt",
        "property_checkpoint": "artifacts/property/model.pt",
        "property_metadata": "artifacts/property/model_metadata.json",
        "training_sequences": "data/training/training.fasta",
        "training_data_manifest": "data/training/dataset_manifest.json",
        "reference_sequences": "data/antibacterial.fasta",
    }
    for name, expected_path in fixed_artifact_paths.items():
        if artifacts[name].get("path") != expected_path:
            raise ValueError(
                f"Artifact {name} must use the documented path {expected_path}"
            )
    if (
        manifest.get("property_model", {}).get("backbone_dir")
        != "artifacts/property/backbone"
    ):
        raise ValueError("Property backbone must use artifacts/property/backbone")
    if (
        manifest.get("property_model", {}).get("tokenizer_dir")
        != "artifacts/property/tokenizer"
    ):
        raise ValueError("Property tokenizer must use artifacts/property/tokenizer")
    for name, value in artifacts.items():
        values = value if isinstance(value, list) else [value]
        if not values:
            raise ValueError(f"Artifact entry {name!r} cannot be empty")
        for item in values:
            if not isinstance(item, dict) or not item.get("path"):
                raise ValueError(f"Artifact entry {name!r} requires path and sha256")
            if len(str(item.get("sha256", ""))) != 64:
                raise ValueError(f"Artifact entry {name!r} has no valid SHA-256")
            try:
                int(item["sha256"], 16)
            except ValueError as exc:
                raise ValueError(
                    f"Artifact entry {name!r} has a malformed SHA-256"
                ) from exc
            if name == "property_files" and not item["path"].startswith(
                ("artifacts/property/backbone/", "artifacts/property/tokenizer/")
            ):
                raise ValueError(
                    "Property file hashes must be under the fixed backbone/tokenizer paths"
                )
            refs.append(item)
    for item in refs:
        path = _root_path(root, item["path"])
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing required artifact {item['path']!r}: {path}"
            )
        actual = sha256_file(path)
        if actual.lower() != item["sha256"].lower():
            raise ValueError(
                f"SHA-256 mismatch for {item['path']}: expected {item['sha256']}, got {actual}"
            )
    if not isinstance(manifest.get("scorers"), list) or not manifest["scorers"]:
        raise ValueError("Manifest must enable at least one artifact-backed scorer")
    for scorer in manifest["scorers"]:
        if scorer.get("artifact") not in artifacts:
            raise ValueError(
                f"Scorer references unknown artifact: {scorer.get('artifact')}"
            )
    ctmc = manifest.get("ctmc", {})
    vocab = manifest.get("vocabulary", {})
    residues = vocab.get("residues")
    if residues != CANONICAL_VOCAB or vocab.get("max_length") != 50:
        raise ValueError("Manifest vocabulary or maximum length is incompatible")
    encoder = AMPCanvasEncoder(max_length=50, vocab=residues)
    if vocab.get("token_ids") != encoder.token_to_idx:
        raise ValueError("Manifest token IDs/order do not match the CTMC encoder")
    if ctmc.get("checkpoint_format_version") != 2:
        raise ValueError("Manifest CTMC checkpoint format must be version 2")
    if ctmc.get("config", {}).get("vocab") != residues:
        raise ValueError("CTMC model configuration vocabulary does not match manifest")
    if ctmc.get("config", {}).get("max_length") != 50:
        raise ValueError("CTMC model configuration maximum length must be 50")
    for key in ("d_model", "n_heads", "n_layers", "d_ff", "condition_dim"):
        if not isinstance(ctmc.get("config", {}).get(key), int):
            raise ValueError(f"CTMC model configuration is missing integer {key}")
    property_model = manifest.get("property_model", {})
    conditions = manifest.get("conditions", {})
    semantic_definition = manifest.get("condition_semantics")
    if not isinstance(semantic_definition, dict):
        raise ValueError("Manifest is missing condition semantic definition")
    canonical_semantics = panel_semantic_definition(
        semantic_definition.get("ranking_strains", []),
        semantic_definition.get("strain_groups", {}),
        semantic_definition.get("activity_threshold_log2_mic", float("nan")),
        semantic_definition.get("activity_temperature", float("nan")),
        semantic_definition.get("broad_spectrum_aggregation"),
    )
    if canonical_semantics != semantic_definition:
        raise ValueError("Manifest condition semantic definition is not canonical")
    if property_model.get("semantic_definition") != semantic_definition:
        raise ValueError("Property manifest condition semantics differ")
    if conditions.get("semantic_definition") != semantic_definition:
        raise ValueError("Generation condition semantics differ")
    cache_contract = manifest.get("condition_cache", {})
    if cache_contract.get("semantic_definition") != semantic_definition:
        raise ValueError("Condition-cache manifest semantics differ")
    if ctmc.get("config", {}).get("condition_semantics") != semantic_definition:
        raise ValueError("CTMC config condition semantics differ")
    if cache_contract.get("table_sha256") != ctmc.get("config", {}).get(
        "condition_table_sha256"
    ):
        raise ValueError("Condition-cache table hash differs from CTMC config")
    if cache_contract.get("manifest_sha256") != ctmc.get("config", {}).get(
        "condition_manifest_sha256"
    ):
        raise ValueError("Condition-cache manifest hash differs from CTMC config")
    if ctmc.get("config", {}).get("training_mode") == "property_conditioned":
        for key in ("table_sha256", "manifest_sha256"):
            value = cache_contract.get(key)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(
                    f"Property-conditioned manifest has invalid cache {key}"
                )
            try:
                int(value, 16)
            except ValueError as exc:
                raise ValueError(
                    f"Property-conditioned manifest has invalid cache {key}"
                ) from exc
        property_hash = ctmc.get("config", {}).get("property_model_checkpoint_sha256")
        if property_hash != artifacts["property_checkpoint"]["sha256"]:
            raise ValueError(
                "Condition cache property checkpoint differs from generation model"
            )
    normalization = ctmc.get("config", {}).get("condition_normalization")
    if not isinstance(normalization, dict):
        raise ValueError("CTMC manifest is missing condition normalization statistics")
    if (
        normalization.get("condition_names") != list(CONDITION_NAMES)
        or normalization.get("schema_version") != 1
    ):
        raise ValueError("Condition normalization schema is missing or incompatible")
    split_hash = ctmc.get("config", {}).get("training_split_sha256")
    if (
        not split_hash
        or normalization.get("fitting_split_sha256") != split_hash
        or normalization.get("fit_split") != "train"
    ):
        raise ValueError("Condition normalization was fitted on a different split")
    if len(str(split_hash)) != 64:
        raise ValueError("Condition normalization split hash is malformed")
    try:
        int(split_hash, 16)
    except ValueError as exc:
        raise ValueError("Condition normalization split hash is malformed") from exc
    statistics = normalization.get("statistics", {})
    if set(statistics) != set(CONDITION_NAMES):
        raise ValueError("Condition normalization statistics are incomplete")
    for name in CONDITION_NAMES:
        stat = statistics[name]
        vals = [stat.get(k) for k in ("mean", "std", "lower_clip", "upper_clip")]
        if (
            any(value is None or not math.isfinite(float(value)) for value in vals)
            or float(stat["std"]) <= 0
            or float(stat["lower_clip"]) >= float(stat["upper_clip"])
        ):
            raise ValueError(f"Invalid condition normalization statistics for {name}")
        if not isinstance(stat.get("count"), int) or stat["count"] < 0:
            raise ValueError(f"Invalid normalization count for {name}")
    if property_model.get("checkpoint_format_version") != 1:
        raise ValueError("Property checkpoint format must be version 1")
    if not property_model.get("backbone_dir") or not property_model.get(
        "tokenizer_dir"
    ):
        raise ValueError(
            "Manifest must specify local property backbone and tokenizer paths"
        )
    token_ids = property_model.get("residue_token_ids")
    if not isinstance(token_ids, dict) or set(token_ids) != set(CANONICAL_VOCAB):
        raise ValueError("Manifest must record every property-tokenizer residue ID")
    strain_ids = property_model.get("strain_ids", {})
    ranking_strains = manifest.get("ranking_strains")
    if not isinstance(strain_ids, dict) or not strain_ids:
        raise ValueError("Property metadata must define a nonempty strain_ids mapping")
    if not isinstance(ranking_strains, list) or not ranking_strains:
        raise ValueError("Manifest ranking_strains must be a nonempty list")
    if len(set(ranking_strains)) != len(ranking_strains):
        raise ValueError("Manifest ranking_strains contains duplicate IDs")
    if set(ranking_strains).difference(strain_ids):
        raise ValueError(
            f"Unknown ranking strains: {sorted(set(ranking_strains).difference(strain_ids))}"
        )
    groups = manifest.get("strain_groups", {})
    if not isinstance(groups, dict):
        raise ValueError("strain_groups must be an object")
    for group, members in groups.items():
        if not isinstance(members, list) or set(members).difference(ranking_strains):
            raise ValueError(
                f"Strain group {group!r} contains IDs outside ranking_strains"
            )
    if semantic_definition["ranking_strains"] != ranking_strains or semantic_definition[
        "strain_groups"
    ] != {
        key: list(groups.get(key, []))
        for key in ("gram_negative", "gram_positive", "mdr")
    }:
        raise ValueError("Condition semantic panel/groups differ from generation panel")
    weights = manifest.get("scoring_weights")
    allowed = {
        "mean_log2_mic",
        "worst_log2_mic",
        "gram_positive_mean_log2_mic",
        "gram_negative_mean_log2_mic",
        "mdr_mean_log2_mic",
        "broad_spectrum_probability",
        "mean_activity_probability",
        "mean_gram_negative_activity_probability",
        "mean_gram_positive_activity_probability",
        "mean_mdr_activity_probability",
        "amp_probability",
        "hemolysis_probability",
        "log2_hc50",
    }
    if not isinstance(weights, dict) or not weights or set(weights).difference(allowed):
        raise ValueError(
            "Manifest scoring_weights contains missing or unsupported fields"
        )
    if any(
        not math.isfinite(float(value)) or float(value) < 0
        for value in weights.values()
    ):
        raise ValueError("Scoring weights must be finite and nonnegative")
    if not any(float(value) > 0 for value in weights.values()):
        raise ValueError("At least one scoring weight must be positive")
    for key, group in (
        ("gram_positive_mean_log2_mic", "gram_positive"),
        ("gram_negative_mean_log2_mic", "gram_negative"),
        ("mdr_mean_log2_mic", "mdr"),
    ):
        if float(weights.get(key, 0)) and not groups.get(group):
            raise ValueError(f"Scoring weight {key!r} requires strain_groups.{group}")
    generation = manifest.get("generation", {})
    for key in (
        "library_count",
        "top_count",
        "oversampling_factor",
        "batch_size",
        "max_attempts",
        "steps",
        "temperature",
        "cfg_scale",
    ):
        if key not in generation:
            raise ValueError(f"Manifest generation configuration is missing {key}")
    if (
        int(generation["library_count"]) < 1
        or int(generation["top_count"]) < 1
        or int(generation["top_count"]) > int(generation["library_count"])
        or float(generation["oversampling_factor"]) < 1.0
        or int(generation["batch_size"]) < 1
        or int(generation["max_attempts"]) < 1
        or int(generation["steps"]) < 1
        or float(generation["temperature"]) <= 0
        or float(generation["cfg_scale"]) < 0
    ):
        raise ValueError(
            "Manifest contains invalid generation limits or sampling values"
        )
    profile = manifest.get("submission_profile", "production")
    if profile == "production" and (
        int(generation["library_count"]) != 50000 or int(generation["top_count"]) != 100
    ):
        raise ValueError(
            "Production manifest must request exactly 50,000 and 100 sequences"
        )
    if profile not in {"production", "test"}:
        raise ValueError(f"Unsupported submission profile: {profile}")
    conditions = manifest.get("conditions", {})
    if conditions.get("condition_names") != list(CONDITION_NAMES):
        raise ValueError(
            "Manifest condition schema/order does not match CONDITION_NAMES"
        )
    if len(conditions.get("values", [])) != len(CONDITION_NAMES) or len(
        conditions.get("observed", [])
    ) != len(CONDITION_NAMES):
        raise ValueError(
            "Manifest condition values/mask must match CTMC condition fields"
        )
    if not all(math.isfinite(float(value)) for value in conditions["values"]):
        raise ValueError("Manifest condition values must be finite")
    for index, name in enumerate(CONDITION_NAMES):
        if (
            name.endswith("_probability")
            and not 0 <= float(conditions["values"][index]) <= 1
        ):
            raise ValueError(f"Manifest probability {name} lies outside [0, 1]")
    outputs = manifest.get("outputs", {})
    if (
        outputs.get("library") != "generate/library.fasta"
        or outputs.get("top") != "generate/top.fasta"
    ):
        raise ValueError(
            "Manifest output paths must be generate/library.fasta and generate/top.fasta"
        )
    return manifest


def _set_offline_runtime() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["WANDB_MODE"] = "disabled"


def _make_config(manifest: dict, seed: int) -> AMPConfig:
    payload = dict(manifest["ctmc"]["config"])
    payload.update(
        {
            "seed": seed,
            "device": "auto",
            "n_sequences": int(manifest["generation"]["library_count"]),
            "top_k": int(manifest["generation"]["top_count"]),
            "cfg_scale": float(manifest["generation"]["cfg_scale"]),
            "ctmc_pretraining_mode": "none",
            "score_weights": {"property_model": 1.0},
            "wandb_enabled": False,
        }
    )
    return AMPConfig(**payload)


def _load_property_model(root: Path, manifest: dict, device: torch.device):
    import json

    artifacts = manifest["artifacts"]
    metadata_path = _root_path(root, artifacts["property_metadata"]["path"])
    stored_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for field in (
        "strain_ids",
        "unknown_strain_id",
        "strain_dim",
        "hidden_size",
        "ranking_strains",
        "strain_groups",
        "semantic_definition",
    ):
        if manifest["property_model"].get(field) != stored_metadata.get(field):
            raise ValueError(
                f"Property manifest field {field!r} does not match metadata"
            )
    model, tokenizer = ESMMultiTaskPredictor.from_local_artifacts(
        _root_path(root, artifacts["property_checkpoint"]["path"]),
        _root_path(root, manifest["property_model"]["backbone_dir"]),
        _root_path(root, manifest["property_model"]["tokenizer_dir"]),
        metadata_path,
        device,
    )
    expected_ids = manifest["property_model"]["residue_token_ids"]
    actual_ids = {
        residue: int(tokenizer.convert_tokens_to_ids(residue))
        for residue in CANONICAL_VOCAB
    }
    if actual_ids != {key: int(value) for key, value in expected_ids.items()}:
        raise ValueError(
            "Property tokenizer residue IDs do not match artifact manifest"
        )
    for residue, token_id in actual_ids.items():
        if tokenizer.convert_ids_to_tokens(token_id) != residue:
            raise ValueError(
                f"Property tokenizer cannot encode canonical residue {residue}"
            )
    stored_metadata["inference_max_length"] = int(
        manifest["property_model"].get("inference_max_length", 64)
    )
    stored_metadata["inference_batch_size"] = int(
        manifest["property_model"].get("inference_batch_size", 128)
    )
    return model, tokenizer, stored_metadata


def validate_checkpoint_conditioning_contract(contract: dict, manifest: dict) -> None:
    """Fail closed when checkpoint conditioning differs from generation semantics."""
    if contract.get("condition_names") != manifest["conditions"].get("condition_names"):
        raise ValueError(
            "Generation manifest condition schema differs from CTMC checkpoint"
        )
    if contract.get("semantic_definition") != manifest.get("condition_semantics"):
        raise ValueError("Generation semantic definition differs from CTMC checkpoint")
    ctmc_config = manifest["ctmc"]["config"]
    if contract.get("condition_normalization") != ctmc_config.get(
        "condition_normalization"
    ):
        raise ValueError("Generation normalization differs from CTMC checkpoint")
    if contract.get("training_split_sha256") != ctmc_config.get(
        "training_split_sha256"
    ):
        raise ValueError("Generation normalization split differs from CTMC checkpoint")
    cache = manifest.get("condition_cache", {})
    if contract.get("condition_table_sha256") != cache.get(
        "table_sha256"
    ) or contract.get("condition_manifest_sha256") != cache.get("manifest_sha256"):
        raise ValueError(
            "Generation condition cache hashes differ from CTMC checkpoint"
        )


def _rank_normalize(values: np.ndarray, maximize: bool) -> np.ndarray:
    oriented = values if maximize else -values
    order = np.argsort(oriented, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float32)
    if len(values) <= 1:
        return np.ones(len(values), dtype=np.float32)
    cursor = 0
    while cursor < len(values):
        end = cursor + 1
        while end < len(values) and oriented[order[end]] == oriented[order[cursor]]:
            end += 1
        rank = 0.5 * (cursor + end - 1) / (len(values) - 1)
        ranks[order[cursor:end]] = rank
        cursor = end
    return ranks


def panel_component_scores(
    mic_by_strain: dict[str, np.ndarray],
    amp: np.ndarray,
    hemolysis: np.ndarray,
    hc50: np.ndarray,
    ranking_strains: list[str],
    strain_groups: dict[str, list[str]],
    semantic_definition: dict,
) -> dict[str, np.ndarray]:
    """Shared property-evaluation and condition aggregation implementation."""
    missing = set(ranking_strains).difference(mic_by_strain)
    if missing:
        raise ValueError(f"Missing predictions for ranking strains: {sorted(missing)}")
    panel = np.stack([mic_by_strain[strain] for strain in ranking_strains], axis=1)
    expected_groups = {
        key: list(strain_groups.get(key, []))
        for key in ("gram_negative", "gram_positive", "mdr")
    }
    if (
        semantic_definition.get("ranking_strains") != ranking_strains
        or semantic_definition.get("strain_groups") != expected_groups
    ):
        raise ValueError(
            "Property scoring semantic definition does not match panel/groups"
        )
    result = {
        "mean_log2_mic": panel.mean(axis=1),
        "worst_log2_mic": panel.max(axis=1),
        "amp_probability": np.asarray(amp),
        "hemolysis_probability": np.asarray(hemolysis),
        "log2_hc50": np.asarray(hc50),
    }
    for field in ("amp_probability", "hemolysis_probability"):
        values = np.asarray(result[field], dtype=np.float64)
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError(
                f"Property probability {field} must be finite and in [0, 1]"
            )
    for group, field in (
        ("gram_positive", "gram_positive_mean_log2_mic"),
        ("gram_negative", "gram_negative_mean_log2_mic"),
        ("mdr", "mdr_mean_log2_mic"),
    ):
        members = strain_groups.get(group, [])
        if members:
            absent = set(members).difference(mic_by_strain)
            if absent:
                raise ValueError(f"Missing {group} predictions: {sorted(absent)}")
            result[field] = np.stack(
                [mic_by_strain[name] for name in members], axis=1
            ).mean(axis=1)
    semantics = [
        compute_panel_condition_semantics(
            {strain: float(mic_by_strain[strain][i]) for strain in ranking_strains},
            semantic_definition,
        )
        for i in range(panel.shape[0])
    ]
    for name in CONDITION_NAMES:
        values = [item["conditions"].get(name) for item in semantics]
        if all(value is not None for value in values):
            result[name] = np.asarray(values, dtype=np.float64)
    result["minimum_activity_probability"] = np.asarray(
        [item["audit"]["minimum_activity_probability"] for item in semantics]
    )
    result["worst_predicted_log2_mic"] = result["worst_log2_mic"]
    result["panel_coverage"] = np.asarray(
        [item["audit"]["panel_coverage"] for item in semantics]
    )
    for strain in ranking_strains:
        result[f"mic_log2__{strain}"] = np.asarray(mic_by_strain[strain])
        result[f"activity_probability__{strain}"] = np.asarray(
            [
                item["audit"]["per_strain_activity_probability"][strain]
                for item in semantics
            ]
        )
    return result


def combine_panel_scores(
    components: dict[str, np.ndarray], weights: dict[str, float]
) -> np.ndarray:
    """Combine component ranks with deterministic, direction-aware ranking."""
    maximize = {name for name in components if name.endswith("_probability")} | {
        "log2_hc50"
    }
    total = sum(float(value) for value in weights.values())
    if total <= 0 or set(weights).difference(components):
        raise ValueError("Scoring weights do not match available panel components")
    score = None
    for name, weight in weights.items():
        ranked = _rank_normalize(
            np.asarray(components[name], dtype=np.float64), name in maximize
        )
        contribution = ranked * float(weight)
        score = contribution if score is None else score + contribution
    assert score is not None
    return score / total


def _property_scores(
    sequences: list[str],
    model,
    tokenizer,
    model_metadata: dict,
    manifest: dict,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    strain_ids = model_metadata["strain_ids"]
    ranking_strains = manifest["ranking_strains"]
    semantics = model_metadata.get("semantic_definition")
    if not isinstance(semantics, dict) or semantics != manifest.get(
        "condition_semantics"
    ):
        raise ValueError("Property metadata and generation manifest semantics differ")
    missing = set(ranking_strains).difference(strain_ids)
    if missing:
        raise ValueError(
            f"Unknown ranking strains in property metadata: {sorted(missing)}"
        )
    max_length = int(model_metadata.get("inference_max_length", 64))
    batch_size = int(model_metadata.get("inference_batch_size", 128))
    mic_by_strain = {name: [] for name in ranking_strains}
    raw = {name: [] for name in ("amp", "hemolysis", "hc50")}
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start : start + batch_size]
            encoded = tokenizer(
                batch,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            panel_ids = torch.tensor(
                [int(strain_ids[name]) for name in ranking_strains],
                dtype=torch.long,
                device=device,
            )
            output = model.forward_panel(
                encoded["input_ids"], encoded["attention_mask"], panel_ids
            )
            panel_mic = output.mic_mean.cpu().numpy()
            for index, strain_name in enumerate(ranking_strains):
                mic_by_strain[strain_name].extend(panel_mic[:, index].tolist())
            raw["amp"].extend(output.amp_probability.cpu().tolist())
            raw["hemolysis"].extend(output.hemolysis_probability.cpu().tolist())
            raw["hc50"].extend(output.hc50_mean.cpu().tolist())
    components = panel_component_scores(
        {name: np.asarray(values) for name, values in mic_by_strain.items()},
        np.asarray(raw["amp"]),
        np.asarray(raw["hemolysis"]),
        np.asarray(raw["hc50"]),
        ranking_strains,
        manifest.get("strain_groups", {}),
        semantics,
    )
    scores = combine_panel_scores(components, manifest["scoring_weights"])
    if (
        scores.shape != (len(sequences),)
        or not np.isfinite(scores).all()
        or any(not np.isfinite(value).all() for value in components.values())
    ):
        raise ValueError("Property model returned invalid candidate scores")
    for name, values in components.items():
        if name.endswith("_probability") and ((values < 0).any() or (values > 1).any()):
            raise ValueError(f"Candidate score probability {name} lies outside [0, 1]")
    return scores, components


def _write_scores(
    path: Path,
    sequences: list[str],
    scores: np.ndarray,
    components: dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    fields = ["sequence", "score", *components]
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, sequence in enumerate(sequences):
            writer.writerow(
                {
                    "sequence": sequence,
                    "score": float(scores[index]),
                    **{key: float(value[index]) for key, value in components.items()},
                }
            )
    temporary.replace(path)


def generate_from_manifest(
    root: Path,
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    output_dir: Path | None = None,
    seed: int | None = None,
) -> tuple[Path, Path]:
    """Run the production path. All inputs and model files are local and hashed."""
    root = root.resolve()
    _set_offline_runtime()
    manifest = validate_manifest(root, manifest_path)
    seed = int(manifest["default_seed"] if seed is None else seed)
    random.seed(seed)
    np.random.seed(seed)
    set_global_determinism(seed)
    config = _make_config(manifest, seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = AMPCanvasEncoder(max_length=50, vocab=CANONICAL_VOCAB)
    ctmc_path = _root_path(root, manifest["artifacts"]["ctmc_checkpoint"]["path"])
    model = CTMCDenoiser.load(
        ctmc_path, config, encoder.vocab_size, encoder.pad_idx, device
    )
    checkpoint_payload = torch.load(ctmc_path, map_location="cpu", weights_only=True)
    contract = checkpoint_payload.get("conditioning_contract", {})
    validate_checkpoint_conditioning_contract(contract, manifest)
    if contract.get("training_mode") == "unconditional" and config.cfg_scale != 0:
        raise ValueError("Unconditional CTMC checkpoint refuses nonzero CFG scale")
    model.eval()
    property_model, tokenizer, property_metadata = _load_property_model(
        root, manifest, device
    )
    property_model.eval()
    repo = FastaRepository(root)
    training_sequences = repo.read_sequences(
        _root_path(root, manifest["artifacts"]["training_sequences"]["path"])
    )
    references = repo.read_sequences(
        _root_path(root, manifest["artifacts"]["reference_sequences"]["path"])
    )
    forbidden = set(training_sequences).union(references)
    training_manifest = manifest["artifacts"].get("training_data_manifest")
    if training_manifest is None:
        raise ValueError("Manifest must reference the training-data manifest")

    generation = manifest["generation"]
    library_count = int(generation["library_count"])
    top_count = int(generation["top_count"])
    if library_count < 1 or not 1 <= top_count <= library_count:
        raise ValueError("Invalid library/top counts in manifest")
    candidate_target = max(
        library_count,
        int(np.ceil(library_count * float(generation["oversampling_factor"]))),
    )
    batch_size = int(generation["batch_size"])
    max_attempts = int(generation["max_attempts"])
    if batch_size < 1 or max_attempts < 1:
        raise ValueError("Generation batch_size and max_attempts must be positive")
    lengths = np.bincount([len(seq) for seq in training_sequences], minlength=51)[
        8:51
    ].astype(np.float32)
    if lengths.sum() == 0:
        raise ValueError("Training sequences have no usable lengths in 8..50")
    length_prior = (lengths + 1e-6) / (lengths + 1e-6).sum()
    sampler = TauLeapingSampler(
        model,
        encoder,
        SinSquaredSchedule(),
        min_length=8,
        length_prior=length_prior.tolist(),
        dfm_stochasticity=float(generation.get("dfm_stochasticity", 0.0)),
    )
    cond = manifest["conditions"]
    condition_values = torch.tensor(cond["values"], dtype=torch.float32, device=device)
    condition_mask = torch.tensor(cond["observed"], dtype=torch.bool, device=device)
    condition_values = normalize_condition_values(
        condition_values.unsqueeze(0),
        condition_mask.unsqueeze(0),
        manifest["ctmc"]["config"]["condition_normalization"],
    ).squeeze(0)
    if condition_values.numel() != len(
        CONDITION_NAMES
    ) or condition_mask.numel() != len(CONDITION_NAMES):
        raise ValueError("Condition targets must match the documented CTMC fields")

    candidates: list[str] = []
    seen: set[str] = set()
    rejected = {
        "invalid_length": 0,
        "invalid_alphabet": 0,
        "reference_collision": 0,
        "duplicate": 0,
    }
    for attempt in range(max_attempts):
        n = min(batch_size, candidate_target - len(candidates))
        if n <= 0:
            break
        conditions = ConditionVector(
            condition_values.unsqueeze(0).expand(n, -1),
            condition_mask.unsqueeze(0).expand(n, -1),
            provenance=("manifest_target",) * n,
        )
        sampled = sampler.sample_batch(
            ReverseGenerationConfig(
                steps=int(generation["steps"]),
                temperature=float(generation["temperature"]),
                seed=seed + attempt,
                confidence_reveal_fraction=float(
                    generation.get("confidence_reveal_fraction", 0.0)
                ),
                conditions=conditions,
                cfg_scale=config.cfg_scale,
            ),
            n,
        )
        for sequence in sampled:
            if not 8 <= len(sequence) <= 50:
                rejected["invalid_length"] += 1
            elif any(residue not in CANONICAL_VOCAB for residue in sequence):
                rejected["invalid_alphabet"] += 1
            elif sequence in forbidden:
                rejected["reference_collision"] += 1
            elif sequence in seen:
                rejected["duplicate"] += 1
            else:
                seen.add(sequence)
                candidates.append(sequence)
        if len(candidates) >= candidate_target:
            break
    if len(candidates) < candidate_target:
        raise RuntimeError(
            f"Candidate budget exhausted: produced {len(candidates)}/{candidate_target} unique valid candidates after {max_attempts} attempts; rejections={rejected}"
        )

    scores, components = _property_scores(
        candidates, property_model, tokenizer, property_metadata, manifest, device
    )
    ranked_indices = sorted(
        range(len(candidates)), key=lambda i: (-float(scores[i]), candidates[i])
    )
    selected = ranked_indices[:library_count]
    library = [candidates[index] for index in selected]

    TOP_IDENTITY_LIMIT = 0.80
    ref_targets = tuple(set(references).union(training_sequences))

    eligible_top_indices: list[int] = []
    for index in ranked_indices:
        seq = candidates[index]
        if not ref_targets or max(official_identity(seq, ref) for ref in ref_targets) <= TOP_IDENTITY_LIMIT:
            eligible_top_indices.append(index)
        if len(eligible_top_indices) == top_count:
            break

    if len(eligible_top_indices) < top_count:
        raise RuntimeError(
            f"Only {len(eligible_top_indices)} top candidates satisfy identity <= {TOP_IDENTITY_LIMIT}"
        )

    top = [candidates[index] for index in eligible_top_indices]
    
    output_dir = output_dir or Path(manifest["outputs"]["library"]).parent
    output_path = _root_path(root, str(output_dir))
    library_path, top_path = output_path / "library.fasta", output_path / "top.fasta"
    repo.write_fasta(library_path, library, id_prefix="seq")
    repo.write_fasta(top_path, top, id_prefix="seq")
    _write_scores(
        output_path / "scores.csv",
        library,
        scores[selected],
        {key: values[selected] for key, values in components.items()},
    )
    report = validate_fasta_files(
        library_path,
        top_path,
        references,
        training_sequences,
        identity_function=official_identity,
        threshold=DEFAULT_THRESHOLD,
        expected_library_count=library_count,
        expected_top_count=top_count,
    )
    repo.write_json_atomic(
        output_path / "compliance_report.json",
        {
            **report,
            "official_identity_source_commit": "987afe5c9f73c82d39b0def7acdc56dec31a0132",
            "rejected_candidates": rejected,
            "candidate_count": len(candidates),
            "seed": seed,
            "manifest": str(manifest_path),
        },
    )
    return library_path, top_path
