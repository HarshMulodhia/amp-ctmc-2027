from amp_ctmc_2027.official_identity import (
    DEFAULT_THRESHOLD,
    SOURCE_COMMIT,
    SOURCE_PATH,
    exceeds_official_threshold,
    official_identity,
)


def test_official_identity_is_vendored_from_template_history():
    assert SOURCE_COMMIT == "987afe5c9f73c82d39b0def7acdc56dec31a0132"
    assert SOURCE_PATH == "scripts/verify_submission.py:_veritfy_max_simularity"
    assert official_identity("ACDEFGHI", "ACDEFGHI") == 1.0


def test_official_identity_threshold_uses_strict_greater_than():
    assert DEFAULT_THRESHOLD == 0.8
    assert not exceeds_official_threshold("AAAAA", "AAAAB")
    assert exceeds_official_threshold("ACDEFGHI", "ACDEFGHI")
