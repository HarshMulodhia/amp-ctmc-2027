import json
import importlib.util
import socket
from pathlib import Path

import pytest
import torch
import numpy as np
import shutil

from amp_ctmc_2027.config import AMPConfig
from amp_ctmc_2027.core import CONDITION_NAMES, CTMCDenoiser
from amp_ctmc_2027.data.dataset import AMPCanvasEncoder
from amp_ctmc_2027.data.fasta_io import FastaRepository
from amp_ctmc_2027.models.esm_multitask import ESMMultiTaskPredictor
from amp_ctmc_2027.submission import (
    combine_panel_scores,
    generate_from_manifest,
    panel_component_scores,
    sha256_file,
    validate_checkpoint_conditioning_contract,
    validate_manifest,
)
from amp_ctmc_2027.data.conditions import (
    fit_condition_normalization,
    panel_semantic_definition,
)
from amp_ctmc_2027.compliance import validate_fasta_files
from amp_ctmc_2027.pipeline.property_training_pipeline import PropertyTrainingPipeline
from amp_ctmc_2027.pipeline.training_pipeline import TrainingPipeline

ROOT = Path(__file__).resolve().parents[1]


def test_worst_panel_strain_penalizes_weak_candidate():
    semantics = panel_semantic_definition(
        ["s1", "s2"], {"gram_negative": ["s2"]}, 4.0, 1.0
    )
    components = panel_component_scores(
        {"s1": np.array([1.0, 1.0]), "s2": np.array([1.0, 8.0])},
        np.array([0.5, 0.5]),
        np.array([0.1, 0.1]),
        np.array([5.0, 5.0]),
        ["s1", "s2"],
        semantics["strain_groups"],
        semantics,
    )
    scores = combine_panel_scores(components, {"worst_log2_mic": 1.0})
    assert components["worst_log2_mic"].tolist() == [1.0, 8.0]
    assert scores[0] > scores[1]
    assert (
        components["broad_spectrum_probability"][1]
        < components["broad_spectrum_probability"][0]
    )
    assert (
        components["mean_activity_probability"][1]
        > components["broad_spectrum_probability"][1]
    )
    assert components["mean_gram_negative_activity_probability"][1] == pytest.approx(
        1 / (1 + np.exp(4.0))
    )
    assert components["mic_log2__s2"].tolist() == [1.0, 8.0]


def _hash(path: Path, root: Path) -> dict:
    return {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}


def _write_fasta(path: Path, sequences: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f">seq{i}\n{sequence}\n" for i, sequence in enumerate(sequences, 1)),
        encoding="utf-8",
    )


def _make_tiny_artifacts(root: Path) -> Path:
    from transformers import EsmConfig, EsmModel, EsmTokenizer

    torch.manual_seed(1234)
    encoder = AMPCanvasEncoder()
    config = AMPConfig(
        d_model=16,
        n_heads=4,
        n_layers=1,
        d_ff=32,
        dropout=0.0,
        vocab=encoder.vocab,
        device="cpu",
        seed=29,
    )
    ctmc = CTMCDenoiser(config, encoder.vocab_size, encoder.pad_idx)
    ctmc_dir = root / "artifacts/ctmc"
    ctmc_dir.mkdir(parents=True)
    semantic_definition = panel_semantic_definition(
        ["strain_a", "strain_b"],
        {"gram_positive": ["strain_a"], "gram_negative": ["strain_b"], "mdr": []},
        4.0,
        1.0,
    )
    split_hash = "1" * 64
    normalization = fit_condition_normalization(
        [{"sequence": "ACDEFGHI", "values": {}, "observed": {}}],
        ["ACDEFGHI"],
        split_hash,
    )
    ctmc.save(
        ctmc_dir / "model.pt",
        encoder,
        contract={
            "training_mode": "unconditional",
            "condition_names": list(CONDITION_NAMES),
            "condition_normalization": normalization,
            "semantic_definition": semantic_definition,
            "training_split_sha256": split_hash,
        },
    )

    property_dir = root / "artifacts/property"
    backbone_dir = property_dir / "backbone"
    tokenizer_dir = property_dir / "tokenizer"
    backbone_dir.mkdir(parents=True)
    tokenizer_dir.mkdir(parents=True)
    vocab = ["<cls>", "<pad>", "<eos>", "<unk>", "<mask>", *encoder.vocab]
    vocab_path = root / "artifacts/property/vocab.txt"
    vocab_path.write_text("\n".join(vocab) + "\n", encoding="utf-8")
    tokenizer = EsmTokenizer(vocab_file=str(vocab_path))
    tokenizer.save_pretrained(tokenizer_dir)
    esm_config = EsmConfig(
        vocab_size=len(tokenizer),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        max_position_embeddings=64,
        mask_token_id=tokenizer.mask_token_id,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.cls_token_id,
        eos_token_id=tokenizer.eos_token_id,
        token_dropout=False,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    backbone = EsmModel(esm_config)
    backbone.save_pretrained(backbone_dir, safe_serialization=True)
    property_model = ESMMultiTaskPredictor(backbone, 16, 2, strain_dim=4)
    metadata = {
        "strain_ids": {"strain_a": 0, "strain_b": 1},
        "ranking_strains": ["strain_a", "strain_b"],
        "strain_groups": {
            "gram_positive": ["strain_a"],
            "gram_negative": ["strain_b"],
            "mdr": [],
        },
        "semantic_definition": semantic_definition,
        "unknown_strain_id": 2,
        "strain_dim": 4,
        "hidden_size": 16,
        "inference_max_length": 60,
        "inference_batch_size": 4,
    }
    (property_dir / "model_metadata.json").write_text(
        json.dumps(metadata) + "\n", encoding="utf-8"
    )
    property_checkpoint = {
        "checkpoint_version": 1,
        "architecture": {"hidden_size": 16, "strain_count": 2, "strain_dim": 4},
        "state_dict": property_model.state_dict(),
    }
    torch.save(property_checkpoint, property_dir / "model.pt")
    training_path = root / "data/training/training.fasta"
    reference_path = root / "data/antibacterial.fasta"
    _write_fasta(training_path, ["ACDEFGHI", "KLMNPQRS", "TVWYACDE"])
    _write_fasta(reference_path, ["YYYYYYYY"])
    training_manifest = root / "data/training/dataset_manifest.json"
    training_manifest.write_text('{"source":"tiny test fixture"}\n', encoding="utf-8")

    artifacts = {
        "ctmc_checkpoint": _hash(ctmc_dir / "model.pt", root),
        "property_checkpoint": _hash(property_dir / "model.pt", root),
        "property_metadata": _hash(property_dir / "model_metadata.json", root),
        "property_files": [
            _hash(path, root)
            for base in (backbone_dir, tokenizer_dir)
            for path in sorted(base.rglob("*"))
            if path.is_file()
        ],
        "training_sequences": _hash(training_path, root),
        "training_data_manifest": _hash(training_manifest, root),
        "reference_sequences": _hash(reference_path, root),
    }
    manifest = {
        "artifact_schema_version": 1,
        "submission_profile": "test",
        "official_identity_source": {
            "repository_template_commit": "987afe5c9f73c82d39b0def7acdc56dec31a0132",
            "threshold": 0.8,
            "reject_semantics": "strictly greater than",
        },
        "default_seed": 42,
        "outputs": {"library": "generate/library.fasta", "top": "generate/top.fasta"},
        "vocabulary": {
            "residues": encoder.vocab,
            "max_length": 50,
            "token_ids": encoder.token_to_idx,
        },
        "ctmc": {
            "checkpoint_format_version": 2,
            "config": {
                "vocab": encoder.vocab,
                "max_length": 50,
                "d_model": 16,
                "n_heads": 4,
                "n_layers": 1,
                "d_ff": 32,
                "condition_dim": 9,
                "condition_normalization": normalization,
                "condition_semantics": semantic_definition,
                "training_split_sha256": split_hash,
            },
        },
        "property_model": {
            "checkpoint_format_version": 1,
            "backbone_dir": "artifacts/property/backbone",
            "tokenizer_dir": "artifacts/property/tokenizer",
            "residue_token_ids": {
                residue: int(tokenizer.convert_tokens_to_ids(residue))
                for residue in encoder.vocab
            },
            **metadata,
        },
        "condition_semantics": semantic_definition,
        "condition_cache": {
            "table_sha256": None,
            "manifest_sha256": None,
            "semantic_definition": semantic_definition,
        },
        "scorers": [{"name": "tiny_property", "artifact": "property_checkpoint"}],
        "ranking_strains": ["strain_a", "strain_b"],
        "strain_groups": {
            "gram_positive": ["strain_a"],
            "gram_negative": ["strain_b"],
            "mdr": [],
        },
        "scoring_weights": {
            "mean_log2_mic": 0.25,
            "worst_log2_mic": 0.35,
            "amp_probability": 0.15,
            "hemolysis_probability": 0.1,
            "log2_hc50": 0.15,
        },
        "artifacts": artifacts,
        "generation": {
            "library_count": 6,
            "top_count": 2,
            "oversampling_factor": 1.5,
            "batch_size": 3,
            "max_attempts": 20,
            "steps": 10,
            "temperature": 1.0,
            "cfg_scale": 0.0,
        },
        "conditions": {
            "semantic_definition": semantic_definition,
            "condition_names": list(CONDITION_NAMES),
            "values": [1.0] * 9,
            "observed": [True] * 9,
        },
    }
    manifest_path = root / "artifacts/manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def test_end_to_end_generation_and_offline_rehearsal(tmp_path, monkeypatch):
    _make_tiny_artifacts(tmp_path)

    def no_network(*_args, **_kwargs):
        raise AssertionError("generation attempted network access")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.chdir("/")
    first = generate_from_manifest(tmp_path, Path("artifacts/manifest.json"))
    assert first == (
        tmp_path / "generate/library.fasta",
        tmp_path / "generate/top.fasta",
    )
    second = generate_from_manifest(
        tmp_path, Path("artifacts/manifest.json"), output_dir=Path("generate/run2")
    )
    assert [sha256_file(path) for path in first] == [
        sha256_file(path) for path in second
    ]
    different_seed = generate_from_manifest(
        tmp_path,
        Path("artifacts/manifest.json"),
        output_dir=Path("generate/other-seed"),
        seed=43,
    )
    assert sha256_file(first[0]) != sha256_file(different_seed[0])
    for library_path, top_path in (first, second, different_seed):
        library = FastaRepository(tmp_path).read_sequences(library_path)
        top = FastaRepository(tmp_path).read_sequences(top_path)
        assert len(library) == 6 and len(set(library)) == 6
        assert len(top) == 2 and len(set(top)) == 2
        assert set(top).issubset(library)
        assert all(
            8 <= len(seq) <= 50 and set(seq) <= set(AMPCanvasEncoder().vocab)
            for seq in library
        )

    spec = importlib.util.spec_from_file_location(
        "rehearse_submission", ROOT / "scripts/rehearse_submission.py"
    )
    rehearsal_module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(rehearsal_module)
    hashes = rehearsal_module.rehearse(tmp_path, Path("artifacts/manifest.json"))
    assert len(hashes[0]) == 64 and len(hashes[1]) == 64


def test_condition_semantics_mismatches_fail_closed(tmp_path):
    manifest_path = _make_tiny_artifacts(tmp_path)
    manifest = validate_manifest(tmp_path, manifest_path)
    checkpoint = torch.load(
        tmp_path / "artifacts/ctmc/model.pt", map_location="cpu", weights_only=True
    )
    contract = checkpoint["conditioning_contract"]
    altered_contract = {
        **contract,
        "semantic_definition": {
            **contract["semantic_definition"],
            "activity_temperature": 2.0,
        },
    }
    with pytest.raises(ValueError, match="semantic definition"):
        validate_checkpoint_conditioning_contract(altered_contract, manifest)
    manifest["condition_cache"] = {
        **manifest["condition_cache"],
        "semantic_definition": {
            **manifest["condition_semantics"],
            "activity_temperature": 2.0,
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Condition-cache manifest semantics"):
        validate_manifest(tmp_path, manifest_path)


def test_artifact_hash_missing_and_corrupt_checkpoint_failures(tmp_path):
    manifest_path = _make_tiny_artifacts(tmp_path)
    checkpoint = tmp_path / "artifacts/ctmc/model.pt"
    original = checkpoint.read_bytes()
    checkpoint.write_bytes(original + b"corrupt")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        generate_from_manifest(tmp_path)
    checkpoint.write_bytes(original)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint.unlink()
    with pytest.raises(FileNotFoundError, match="Missing required artifact"):
        generate_from_manifest(tmp_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "checkpoint_version": 2,
            "canvas_semantics": "residues_plus_eos",
            "vocabulary": AMPCanvasEncoder().vocab,
            "token_ids": AMPCanvasEncoder().token_to_idx,
            "architecture": {
                key: manifest["ctmc"]["config"][key]
                for key in (
                    "d_model",
                    "n_heads",
                    "n_layers",
                    "d_ff",
                    "max_length",
                    "condition_dim",
                )
            },
            "state_dict": {},
        },
        checkpoint,
    )
    manifest["artifacts"]["ctmc_checkpoint"]["sha256"] = sha256_file(checkpoint)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="Missing key|Unexpected key"):
        generate_from_manifest(tmp_path)


def test_tiny_training_exports_manifest_generation_validation_and_rehearsal(
    tmp_path, monkeypatch
):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from amp_ctmc_2027.official_identity import official_identity

    _make_tiny_artifacts(tmp_path)
    property_dir = tmp_path / "artifacts/property"
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(
        property_dir / "tokenizer", local_files_only=True
    )
    tokenizer.save_pretrained(property_dir / "backbone")
    sequences = ["ACDEFGHI", "KLMNPQRS", "TVWYACDE", "CDEFGHIK", "LMNPQRST", "VWYACDEF"]
    rows = []
    for index, sequence in enumerate(sequences):
        split = ("train", "train", "val", "val", "test", "test")[index]
        rows.append(
            {
                "sequence": sequence,
                "split": split,
                "cluster_id": f"c{index}",
                "strain": "strain_a" if index % 2 == 0 else "strain_b",
                "is_amp": float(index % 2),
                "mic_uM": float(2 ** (1 + index)),
                "mic_censor": "none",
                "mic_upper_uM": None,
                "is_hemolytic": float(index % 2),
                "hc50_uM": float(2 ** (2 + index)),
                "hc50_censor": "none",
                "hc50_upper_uM": None,
            }
        )
    processed = tmp_path / "data/processed/tiny.parquet"
    processed.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), processed)
    PropertyTrainingPipeline(
        processed,
        Path("artifacts/property"),
        str(property_dir / "backbone"),
        None,
        epochs=1,
        batch_size=2,
        max_length=64,
        seed=7,
        unfreeze_last_n_layers=0,
        repository_root=tmp_path,
        ranking_strains=["strain_a", "strain_b"],
        strain_groups={
            "gram_positive": ["strain_a"],
            "gram_negative": ["strain_b"],
            "mdr": [],
        },
        sequence_clustering_threshold=0.4,
    ).run()
    assert (tmp_path / "data/training/training.fasta").is_file()
    assert (tmp_path / "data/training/dataset_manifest.json").is_file()

    background = tmp_path / "data/generic/background.fasta"
    _write_fasta(background, ["YYYYYYYY", "WWWWWWWW"])
    config = AMPConfig(
        d_model=16,
        n_heads=4,
        n_layers=1,
        d_ff=32,
        dropout=0.0,
        n_epochs=1,
        batch_size=2,
        device="cpu",
        debug_overfit_samples=2,
        discriminator_ensemble_size=1,
        training_fasta_path=Path("data/training/training.fasta"),
        background_fasta_path=Path("data/generic/background.fasta"),
        checkpoint_dir=Path("artifacts/ctmc"),
    )
    TrainingPipeline(config, FastaRepository(tmp_path)).run()
    reference = tmp_path / "data/antibacterial.fasta"
    _write_fasta(reference, ["YYYYYYYY"])
    shutil.copy(
        ROOT / "artifacts/manifest.example.json",
        tmp_path / "artifacts/manifest.example.json",
    )
    spec = importlib.util.spec_from_file_location(
        "build_artifact_manifest", ROOT / "scripts/build_artifact_manifest.py"
    )
    builder = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(builder)
    builder.main(tmp_path)
    manifest_path = tmp_path / "artifacts/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["submission_profile"] = "test"
    manifest["generation"].update(
        {
            "library_count": 6,
            "top_count": 2,
            "oversampling_factor": 1.5,
            "batch_size": 3,
            "max_attempts": 20,
            "steps": 10,
            "cfg_scale": 0.0,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    _write_fasta(reference, ["YYYYYYYY"])
    output = generate_from_manifest(tmp_path)
    assert (tmp_path / "generate/scores.csv").is_file()
    report = validate_fasta_files(
        *output,
        ["YYYYYYYY"],
        sequences[:2],
        identity_function=official_identity,
        expected_library_count=6,
        expected_top_count=2,
    )
    assert report

    def no_network(*_args, **_kwargs):
        raise AssertionError("integration generation attempted network access")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    rehearsal_module_spec = importlib.util.spec_from_file_location(
        "rehearse_submission", ROOT / "scripts/rehearse_submission.py"
    )
    rehearsal = importlib.util.module_from_spec(rehearsal_module_spec)
    assert rehearsal_module_spec.loader is not None
    rehearsal_module_spec.loader.exec_module(rehearsal)
    hashes = rehearsal.rehearse(tmp_path, Path("artifacts/manifest.json"))
    assert hashes[0] == sha256_file(output[0])
