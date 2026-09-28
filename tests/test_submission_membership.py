from amp_ctmc_2027.compliance import validate_submission


def test_ranked_sequences_must_be_in_library():
    report = validate_submission(
        ["ACDEFGHI", "KLMNPQRS"],
        ["KLMNPQRS"],
        [],
        identity_function=lambda _a, _b: 0.0,
        threshold=0.8,
        expected_library_count=2,
        expected_top_count=1,
    )
    assert report["top_is_library_subset"]
