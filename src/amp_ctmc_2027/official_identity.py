"""Identity rule copied from the challenge template's local git history.

Source: 987afe5:scripts/verify_submission.py, function
``_veritfy_max_simularity``. That verifier calls ``Levenshtein.ratio`` and
rejects only ratios strictly greater than 0.8.
"""

from Levenshtein import ratio

SOURCE_COMMIT = "987afe5c9f73c82d39b0def7acdc56dec31a0132"
SOURCE_PATH = "scripts/verify_submission.py:_veritfy_max_simularity"
DEFAULT_THRESHOLD = 0.8


def official_identity(candidate: str, reference: str) -> float:
    """Return the exact ratio used by the challenge template verifier."""
    return float(ratio(candidate, reference))


def exceeds_official_threshold(candidate: str, reference: str) -> bool:
    """Apply the template's strict greater-than threshold semantics."""
    return official_identity(candidate, reference) > DEFAULT_THRESHOLD
