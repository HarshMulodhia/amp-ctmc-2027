import torch
import pytest
from pathlib import Path

from amp_ctmc_2027.config import AMPConfig
from amp_ctmc_2027.core import CONDITION_NAMES, ConditionVector, CTMCDenoiser
from amp_ctmc_2027.data.conditions import (
    compute_panel_condition_semantics,
    fit_condition_normalization,
    normalize_condition_values,
    panel_semantic_definition,
    measured_overrides,
    validate_rows,
    write_condition_table,
    load_condition_table,
)
from amp_ctmc_2027.pipeline.training_pipeline import TrainingPipeline
from amp_ctmc_2027.data.fasta_io import FastaRepository
from amp_ctmc_2027.data.dataset import AMPCanvasEncoder


def test_unimplemented_teacher_distillation_is_rejected_at_config_validation():
    with pytest.raises(ValueError, match="Teacher distillation is not implemented"):
        AMPConfig(ctmc_pretraining_mode="distill")


def test_conditional_unconditional_and_cfg_formula(tiny_model, encoder):
    tokens = torch.stack([encoder.encode("ACDEFGHI"), encoder.encode("KLMNPQRS")])
    time = torch.tensor([0.3, 0.7])
    values = torch.ones((2, len(CONDITION_NAMES)))
    observed = torch.ones_like(values, dtype=torch.bool)
    conditions = ConditionVector(values, observed)
    tiny_model.eval()
    with torch.inference_mode():
        conditional = tiny_model(tokens, time, conditions)
        unconditional = tiny_model(tokens, time)
        guided = tiny_model.guided_logits(tokens, time, conditions, 1.5)
    assert conditional.shape == (2, 51, encoder.vocab_size)
    torch.testing.assert_close(
        guided, unconditional + 1.5 * (conditional - unconditional)
    )


def test_unobserved_condition_values_do_not_affect_logits(tiny_model, encoder):
    tokens = encoder.encode("ACDEFGHI").unsqueeze(0)
    time = torch.tensor([0.4])
    values = torch.zeros((1, len(CONDITION_NAMES)))
    observed = torch.zeros_like(values, dtype=torch.bool)
    first = ConditionVector(values.clone(), observed)
    changed_values = values.clone()
    changed_values[0, 0] = 1000
    changed = ConditionVector(changed_values, observed)
    tiny_model.eval()
    with torch.inference_mode():
        left = tiny_model(tokens, time, first)
        right = tiny_model(tokens, time, changed)
    torch.testing.assert_close(left, right)
    active = ConditionVector(values.clone(), torch.ones_like(observed))
    with torch.inference_mode():
        conditioned = tiny_model(tokens, time, active)
    assert not torch.allclose(left, conditioned)


def test_cfg_endpoints_and_condition_gradient(tiny_model, encoder):
    tokens = encoder.encode("ACDEFGHI").unsqueeze(0)
    t = torch.tensor([0.5])
    conditions = ConditionVector(torch.ones(1, 9), torch.ones(1, 9, dtype=torch.bool))
    tiny_model.zero_grad(set_to_none=True)
    pred0 = tiny_model.guided_logits(tokens, t, conditions, 0)
    pred1 = tiny_model.guided_logits(tokens, t, conditions, 1)
    torch.testing.assert_close(pred0, tiny_model(tokens, t))
    torch.testing.assert_close(pred1, tiny_model(tokens, t, conditions))
    pred1.sum().backward()
    assert tiny_model.condition_embedding[0].weight.grad is not None
    assert tiny_model.condition_embedding[0].weight.grad.abs().sum() > 0


def test_fully_unconditional_path_has_no_condition_gradient(tiny_model, encoder):
    tokens = encoder.encode("ACDEFGHI").unsqueeze(0)
    tiny_model(tokens, torch.tensor([0.5])).sum().backward()
    assert tiny_model.condition_embedding[0].weight.grad is None


def test_training_dropout_removes_all_observed_conditions(tmp_path, encoder):
    config = AMPConfig(
        d_model=16,
        n_heads=4,
        n_layers=1,
        d_ff=32,
        dropout=0,
        condition_dropout=1.0,
        training_mode="property_conditioned",
    )
    pipeline = TrainingPipeline(config, FastaRepository(tmp_path))
    model = CTMCDenoiser(config, encoder.vocab_size, encoder.pad_idx)
    clean = encoder.encode("ACDEFGHI").unsqueeze(0)
    loss, stats = pipeline._loss_for_batch(
        model,
        clean,
        encoder,
        condition_values=torch.ones(1, 9),
        condition_observed=torch.ones(1, 9, dtype=torch.bool),
    )
    assert loss is not None
    assert stats["condition_drop_fraction"] == 1.0
    model.eval()
    tokens = clean
    t = torch.tensor([0.5])
    dropped = ConditionVector(torch.ones(1, 9), torch.zeros(1, 9, dtype=torch.bool))
    with torch.no_grad():
        conditional_api = model(tokens, t, dropped)
        unconditional_api = model(tokens, t)
    torch.testing.assert_close(conditional_api, unconditional_api)


def test_64_sequence_conditional_overfit_reduces_validation_loss():
    torch.manual_seed(9)
    encoder = AMPCanvasEncoder(max_length=10)
    config = AMPConfig(
        d_model=16,
        n_heads=4,
        n_layers=1,
        d_ff=32,
        dropout=0,
        max_length=10,
        min_length=8,
        batch_size=64,
        device="cpu",
    )
    model = CTMCDenoiser(config, encoder.vocab_size, encoder.pad_idx)
    sequences = ["ACDEFGHI", "KLMNPQRS", "TVWYACDE", "FGHIKLMN"] * 16
    clean = torch.stack([encoder.encode(s) for s in sequences])
    noisy = clean.clone()
    noisy[clean.ne(encoder.pad_idx)] = encoder.mask_idx
    target_mask = clean.ne(encoder.pad_idx)
    conditions = ConditionVector(torch.rand(64, 9), torch.ones(64, 9, dtype=torch.bool))
    time = torch.full((64,), 0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    def loss_value():
        with torch.no_grad():
            logits = model(noisy, time, conditions)
            return torch.nn.functional.cross_entropy(
                logits[target_mask], clean[target_mask]
            ).item()

    initial = loss_value()
    for _ in range(12):
        optimizer.zero_grad(set_to_none=True)
        logits = model(noisy, time, conditions)
        loss = torch.nn.functional.cross_entropy(
            logits[target_mask], clean[target_mask]
        )
        loss.backward()
        optimizer.step()
    assert loss_value() < initial
    changed = ConditionVector(conditions.values + 2.0, conditions.observed)
    with torch.no_grad():
        after = model(noisy, time, conditions)
        changed_after = model(noisy, time, changed)
    assert not torch.allclose(after, changed_after)


def test_condition_table_order_masks_override_and_integrity(tmp_path):
    values = {name: 0.5 for name in CONDITION_NAMES}
    observed = {name: True for name in CONDITION_NAMES}
    provenance = {name: "teacher" for name in CONDITION_NAMES}
    observed["hemolysis_probability"] = False
    provenance["hemolysis_probability"] = "missing"
    row = {
        "sequence": "ACDEFGHI",
        "condition_names": list(CONDITION_NAMES),
        "values": values,
        "observed": observed,
        "provenance": provenance,
    }
    checked = validate_rows([row], ["ACDEFGHI"])
    assert list(checked[0]["values"]) == list(CONDITION_NAMES)
    assert not checked[0]["observed"]["hemolysis_probability"]
    table = tmp_path / "conditions.jsonl"
    semantic = panel_semantic_definition(["s1"], {}, 4.0, 1.0)
    write_condition_table(
        table,
        [row],
        {
            "property_checkpoint_sha256": "a" * 64,
            "property_metadata_sha256": "b" * 64,
            "semantic_definition": semantic,
        },
    )
    loaded, _ = load_condition_table(
        table, table.with_suffix(".jsonl.manifest.json"), ["ACDEFGHI"]
    )
    assert loaded[0]["provenance"]["amp_probability"] == "teacher"
    with table.open("a", encoding="utf-8") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="SHA-256"):
        load_condition_table(table, table.with_suffix(".jsonl.manifest.json"))


def test_condition_table_rejects_implicit_order_and_conflicting_duplicates():
    row = {
        "sequence": "ACDEFGHI",
        "condition_names": list(reversed(CONDITION_NAMES)),
        "values": {},
        "observed": {},
        "provenance": {},
    }
    with pytest.raises(ValueError, match="schema/order"):
        validate_rows([row])


def test_property_conditioned_training_rejects_missing_cache(tmp_path):
    config = AMPConfig(training_mode="property_conditioned")
    pipeline = TrainingPipeline(config, FastaRepository(tmp_path))
    with pytest.raises(ValueError, match="requires a condition table"):
        pipeline._load_condition_rows(["ACDEFGHI"])


def test_measured_condition_overrides_teacher_prediction():
    teacher = {
        "values": {name: 0.2 for name in CONDITION_NAMES},
        "observed": {name: True for name in CONDITION_NAMES},
        "provenance": {name: "teacher" for name in CONDITION_NAMES},
    }
    measured = {
        "condition_names": list(CONDITION_NAMES),
        "values": {"amp_probability": 0.9},
        "observed": {"amp_probability": True},
    }
    merged = measured_overrides(teacher, measured)
    assert merged["values"]["amp_probability"] == 0.9
    assert merged["provenance"]["amp_probability"] == "measured"
    assert merged["values"]["hemolysis_probability"] == 0.2
    assert merged["provenance"]["hemolysis_probability"] == "teacher"


def test_panel_semantics_are_probabilities_grouped_and_conservative():
    semantics = panel_semantic_definition(
        ["strong", "weak", "other"],
        {
            "gram_negative": ["weak"],
            "gram_positive": ["strong", "other"],
            "mdr": ["weak", "other"],
        },
        4.0,
        1.0,
    )
    result = compute_panel_condition_semantics(
        {"strong": 0.0, "weak": 8.0, "other": 1.0}, semantics
    )
    values = result["conditions"]
    assert 0 <= values["broad_spectrum_probability"] <= 1
    assert values["mean_gram_negative_activity_probability"] == pytest.approx(
        1 / (1 + torch.exp(torch.tensor(4.0)).item())
    )
    assert (
        values["mean_gram_positive_activity_probability"]
        > values["mean_gram_negative_activity_probability"]
    )
    assert values["mean_activity_probability"] > values["broad_spectrum_probability"]
    assert values["broad_spectrum_probability"] == pytest.approx(
        result["audit"]["minimum_activity_probability"]
    )
    assert result["audit"]["panel_coverage"] == 1.0
    assert result["audit"]["worst_predicted_log2_mic"] == 8.0


def test_normalization_fits_train_only_roundtrips_and_preserves_missing_mask():
    rows = []
    for sequence, value, mask in (
        ("ACDEFGHI", 0.2, True),
        ("KLMNPQRS", 0.6, True),
        ("TVWYACDE", 100.0, True),
    ):
        rows.append(
            {
                "sequence": sequence,
                "values": {name: value for name in CONDITION_NAMES},
                "observed": {name: mask for name in CONDITION_NAMES},
            }
        )
    norm = fit_condition_normalization(rows, ["ACDEFGHI", "KLMNPQRS"], "f" * 64)
    assert norm["statistics"]["amp_probability"]["mean"] == pytest.approx(0.4)
    assert norm["statistics"]["amp_probability"]["count"] == 2
    values = torch.tensor([[0.2] * 9, [0.0] * 9])
    observed = torch.tensor([[True] * 9, [False] * 9])
    transformed = normalize_condition_values(values, observed, norm)
    stat = norm["statistics"]["amp_probability"]
    recovered = transformed[0, 0] * stat["std"] + stat["mean"]
    assert recovered.item() == pytest.approx(0.2)
    assert transformed[1].eq(0).all()
    assert observed[1].eq(False).all()


def test_measured_builder_keeps_censoring_and_masks_insufficient_panel(
    tmp_path, monkeypatch
):
    import csv
    import hashlib
    import importlib.util
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    sequence = "ACDEFGHI"
    semantic = panel_semantic_definition(
        ["strain_a", "strain_b"],
        {"gram_negative": ["strain_a", "strain_b"], "gram_positive": [], "mdr": []},
        4.0,
        1.0,
    )
    metadata = tmp_path / "property.json"
    metadata.write_text(json.dumps({"semantic_definition": semantic}), encoding="utf-8")
    split = tmp_path / "split.csv"
    with split.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sequence", "split_cluster_id", "split"])
        writer.writeheader()
        writer.writerow(
            {"sequence": sequence, "split_cluster_id": "c1", "split": "train"}
        )
    source = tmp_path / "observations.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "sequence": sequence,
                    "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                    "split": "train",
                    "split_cluster_id": "c1",
                    "strain": "strain_a",
                    "mic_uM": 64.0,
                    "mic_upper_uM": None,
                    "mic_censor": "right",
                    "hc50_uM": 128.0,
                    "hc50_upper_uM": None,
                    "hc50_censor": "right",
                    "is_amp": 1.0,
                    "is_hemolytic": None,
                }
            ]
        ),
        source,
    )
    output = tmp_path / "measured.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_measured_conditions",
            "--observations",
            str(source),
            "--split-manifest",
            str(split),
            "--property-metadata",
            str(metadata),
            "--output",
            str(output),
            "--min-panel-coverage",
            "0.75",
        ],
    )
    module_path = (
        Path(__file__).resolve().parents[1] / "scripts/build_measured_conditions.py"
    )
    spec = importlib.util.spec_from_file_location(
        "build_measured_conditions", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    exact = module._mic_probability_interval(
        {"mic_uM": 8.0, "mic_censor": "none"}, semantic
    )
    left = module._mic_probability_interval(
        {"mic_uM": 8.0, "mic_censor": "left"}, semantic
    )
    right = module._mic_probability_interval(
        {"mic_uM": 8.0, "mic_censor": "right"}, semantic
    )
    interval = module._mic_probability_interval(
        {"mic_uM": 4.0, "mic_upper_uM": 16.0, "mic_censor": "interval"}, semantic
    )
    assert exact[3] == "none" and exact[1] == exact[2]
    assert left[3] == "left" and left[1] < left[2]
    assert right[3] == "right" and right[1] < right[2]
    assert interval[3] == "interval" and interval[1] < interval[2]
    module.main()
    row = json.loads(output.read_text().splitlines()[0])
    assert not row["observed"]["mean_activity_probability"]
    assert not row["observed"]["broad_spectrum_probability"]
    assert row["audit"]["censoring"]["strain_a"]["right"] == 1
    assert row["audit"]["fields"]["hc50"]["censoring_counts"]["right"] == 1
    assert row["provenance"]["predicted_log2_hc50"] == "missing"
