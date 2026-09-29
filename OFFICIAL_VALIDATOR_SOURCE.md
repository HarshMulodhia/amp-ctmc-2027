# Official identity validator provenance

The available official challenge template is the repository's initial commit `987afe5c9f73c82d39b0def7acdc56dec31a0132`. Its `scripts/verify_submission.py` defines `_veritfy_max_simularity` and calls `Levenshtein.ratio(seq, ref) > threshold`, with default threshold `0.8`.

The implementation is vendored as `amp_ctmc_2027.official_identity.official_identity`; `SOURCE_COMMIT` and `SOURCE_PATH` identify its origin. The strict greater-than behavior means a value of exactly 0.8 is allowed. `tests/test_official_identity_integration.py` asserts the source metadata and threshold semantics.

The generation validator uses this function on every top sequence against the organizer reference FASTA. `Levenshtein.normalized_similarity` is not used for organizer compliance.
