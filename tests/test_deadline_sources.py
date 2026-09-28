import json
from pathlib import Path

import pytest

from amp_ctmc_2027.data.deadline import (
    DEADLINE_STRAIN_COLUMNS,
    DEADLINE_STRAIN_GROUPS,
    deterministic_sequence_split,
    grampa_log10_mic_to_um,
    hc50_log10_to_um,
    hemopi2_hemolysis_label,
)

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "amp_diffusion_train_eval": "424e1ae8b40bb0e6367251be930caf704673d27b0e583082a119f77d89e48a97",
    "grampa": "9e33122afb3bdd169e8bed620b1040bf1601d60cf25b6c8c8d9870bf37522b6f",
    "cleaned_hc50": "adee9ec2d1cacbfb26db35984ac1e2914d25927d425a76af016258745fd5cf1c",
    "hemopi2_cross_val": "7bdaf3ede499d1eda2712585d2e52d7700f3f138776d1a0a46e2ca88e8152da0",
    "hemopi2_independent": "500013c2244219762ff3ff4a03401c7419790c83d0ef0c3aeebfdbea426b3eb5",
    "amplify_amp_positive": "a04e28f8d29d1bb4f445a6162e210e0999289c31bf93f2f13c8d2268c8dd9cdc",
    "amplify_non_amp": "02161c5d18ff8c5c8c712527fb84a9c51606a46b11ad987cd91c60306acd25a3",
}


def test_deadline_source_catalog_contains_the_exact_pinned_hashes():
    catalog = json.loads((ROOT / "config/deadline_sources.json").read_text())
    sources = {item["id"]: item for item in catalog["sources"]}
    assert len(sources) == 8
    for source_id, digest in EXPECTED.items():
        assert sources[source_id]["sha256"] == digest
    assert sources["organizer_reference"]["local_only"] is True


def test_public_log10_concentration_unit_conversions():
    assert grampa_log10_mic_to_um(1.0) == pytest.approx(10.0)
    assert hc50_log10_to_um(2.0) == pytest.approx(100.0)


def test_hemopi2_label_direction_is_preserved_for_concentration_extremes():
    high_concentration_nonhemolytic = {"uM": 1000.0, "label": 0}
    low_concentration_hemolytic = {"uM": 1.0, "label": 1}
    assert hemopi2_hemolysis_label(high_concentration_nonhemolytic["label"]) == 0
    assert hemopi2_hemolysis_label(low_concentration_hemolytic["label"]) == 1


def test_strain_aliases_and_group_membership_are_exact():
    assert len(DEADLINE_STRAIN_COLUMNS) == 11
    assert DEADLINE_STRAIN_COLUMNS["E_coli_AIC221"] == "E. coli AIG221"
    assert DEADLINE_STRAIN_COLUMNS["E_coli_AIC222"] == "E. coli AIG222"
    assert DEADLINE_STRAIN_GROUPS["gram_negative"] == [
        "A_baumannii_ATCC19606",
        "E_coli_ATCC11775",
        "E_coli_AIC221",
        "E_coli_AIC222",
        "K_pneumoniae_ATCC13883",
        "P_aeruginosa_PAO1",
        "P_aeruginosa_PA14",
    ]
    assert DEADLINE_STRAIN_GROUPS["gram_positive"] == [
        "S_aureus_ATCC12600",
        "S_aureus_ATCC_BAA1556",
        "E_faecalis_ATCC700802",
        "E_faecium_ATCC700221",
    ]
    assert DEADLINE_STRAIN_GROUPS["mdr"] == [
        "E_coli_AIC222",
        "S_aureus_ATCC_BAA1556",
        "E_faecalis_ATCC700802",
        "E_faecium_ATCC700221",
    ]


def test_exact_sequence_hash_split_is_stable_and_global():
    seq = "ACDEFGHIK"
    assert deterministic_sequence_split(seq) == deterministic_sequence_split(seq)
    assert deterministic_sequence_split(seq) in {"train", "val", "test"}
    assert deterministic_sequence_split(seq, 43) == deterministic_sequence_split(
        seq, 43
    )


def test_prepared_fasta_sets_do_not_mix_amp_and_non_amp():
    train = ROOT / "data/training/training.fasta"
    background = ROOT / "data/training/background.fasta"
    if not train.is_file() or not background.is_file():
        pytest.skip("run deadline bootstrap to exercise prepared real FASTA sets")
    from amp_ctmc_2027.data.fasta_io import FastaRepository

    positive = set(FastaRepository(ROOT).read_sequences(train))
    negative = set(FastaRepository(ROOT).read_sequences(background))
    assert positive
    assert negative
    assert positive.isdisjoint(negative)


def test_prepared_observations_retain_micromolar_hc50_and_provenance():
    import pyarrow.parquet as pq

    table = ROOT / "data/processed/deadline_observations.parquet"
    if not table.is_file():
        pytest.skip("run deadline bootstrap to exercise prepared observation rows")
    rows = pq.read_table(table).to_pylist()
    hc50 = next(row for row in rows if row["source"] == "cleaned_hc50")
    assert hc50["hc50_uM"] == pytest.approx(10.0 ** hc50["hc50_original_log10_uM"])
    assert hc50["label_provenance"] == "measured_public_database"
    pseudo = next(row for row in rows if row["source"] == "amp_diffusion_train_eval")
    assert pseudo["label_provenance"] == "pseudo_apex_or_source_model"
    assert pseudo["mic_uM"] == pytest.approx(pseudo["mic_value"])
    hemopi = next(row for row in rows if row["source"] == "hemopi2_cross_val")
    assert hemopi["is_hemolytic"] in (0.0, 1.0)
    assert hemopi["hc50_uM"] > 0


def test_all_prepared_observations_keep_one_global_sequence_split():
    import pyarrow.parquet as pq

    table = ROOT / "data/processed/deadline_observations.parquet"
    if not table.is_file():
        pytest.skip("run deadline bootstrap to exercise prepared observation rows")
    assignments = {}
    for row in pq.read_table(table, columns=["sequence", "split"]).to_pylist():
        old = assignments.setdefault(row["sequence"], row["split"])
        assert old == row["split"]


def test_pinned_snapshot_loads_offline_when_bootstrapped(monkeypatch):
    snapshot = ROOT / "data/pretrained/esm2_t12_35M_UR50D"
    if not snapshot.is_dir():
        pytest.skip("run deadline model bootstrap to exercise the local snapshot")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer, EsmModel

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    EsmModel.from_pretrained(str(snapshot), local_files_only=True)
    ids = [
        tokenizer.encode(residue, add_special_tokens=False)
        for residue in "ACDEFGHIKLMNPQRSTVWY"
    ]
    assert all(len(tokens) == 1 for tokens in ids)
    assert len({tokens[0] for tokens in ids}) == 20
