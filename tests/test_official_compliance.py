import pytest

from amp_ctmc_2027.compliance import load_official_identity, validate_submission


def identity(candidate, reference):
    return 0.8 if candidate == "ACDEFGHI" and reference == "KLMNPQRS" else 0.2


def test_missing_official_identity_validator_fails_closed():
    with pytest.raises(RuntimeError, match="not configured"):
        load_official_identity(None)


def validate(library, top, refs=(), threshold=0.8):
    return validate_submission(
        library,
        top,
        list(refs),
        identity_function=identity,
        threshold=threshold,
        expected_library_count=None,
        expected_top_count=None,
    )


def test_threshold_equality_is_allowed_and_top_is_library_subset():
    report = validate(["ACDEFGHI", "LMNPQRST"], ["ACDEFGHI"], ["KLMNPQRS"])
    assert report["pass"]
    assert report["top_is_library_subset"]
    with pytest.raises(ValueError, match="exceeds"):
        validate(["ACDEFGHI"], ["ACDEFGHI"], ["KLMNPQRS"], threshold=0.79)


@pytest.mark.parametrize(
    "library,top,refs,error",
    [
        (["ACDEFGHI", "ACDEFGHI"], ["ACDEFGHI"], [], "duplicate"),
        (["ACDEFGH*"], ["ACDEFGH*"], [], "Invalid"),
        (["ACDEFGH"], ["ACDEFGH"], [], "Invalid"),
        (["ACDEFGHI"], ["KLMNPQRS"], [], "occur in the library"),
        (["ACDEFGHI"], ["ACDEFGHI"], ["ACDEFGHI"], "exact organizer reference"),
    ],
)
def test_compliance_rejects_invalid_submission(library, top, refs, error):
    with pytest.raises(ValueError, match=error):
        validate(library, top, refs)
