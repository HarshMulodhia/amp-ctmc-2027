# Submission status and procedure

The intended entry point is `uv run generate` with the fixed default seed in `configs/generate.json`. It writes the 50,000-sequence library and ranked 100-sequence list, plus score and run audit files. Generation uses local artifacts only. The top 100 are explicitly inserted into the library.

The code checks canonical alphabet, length, uniqueness, exact membership, and the configured organizer novelty rule through an import adapter. Set `official_identity_function` to the pinned official module and function; the current repository does not contain that validator or its reference commit. The application fails closed when it is missing. RapidFuzz normalized edit similarity is still used as an internal design filter and is not represented as the organizer's official identity score.

Before submission, provide the full source/model disclosure, verified checksums, public inference weights, license and third-party notices, official validator code/revision, and a passing `compliance_report.json`. No manual candidate edits or experimental results are encoded by this repository. At present, absent model weights and source/validator pins mean this checkout is not a ready-to-submit artifact.
