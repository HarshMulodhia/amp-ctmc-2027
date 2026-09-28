from amp_ctmc_2027.data.preprocess import clean_observations


def test_cleaning_preserves_observations_and_reports_bad_rows():
    rows = [
        {"sequence": " acdefghi ", "source": "x", "is_amp": 1},
        {"sequence": "ACDEFGHI", "source": "y", "is_amp": 0},
        {"sequence": "ACDEFGH*", "source": "x"},
        {"sequence": "ACDEFGHI", "source": "x", "is_linear": "False"},
    ]
    result = clean_observations(rows)
    assert len(result.observations) == 2
    assert result.duplicate_count == 1
    assert result.conflicts[0]["kind"] == "contradictory_amp_labels"
    assert {row["reason"] for row in result.rejected} == {
        "noncanonical_residue",
        "modified_or_non_linear",
    }


def test_reference_sequences_are_removed_from_data():
    result = clean_observations([{"sequence": "ACDEFGHI", "source": "x"}], {"ACDEFGHI"})
    assert not result.observations
    assert result.rejected[0]["reason"] == "compliance_reference_overlap"
