# Method

## Generation path

The production entry point reconstructs a version-2 masked continuous-time Markov chain (CTMC) denoiser using the architecture and residue-plus-EOS token IDs recorded in `artifacts/manifest.json`. A manifest target `ConditionVector` is passed at each reverse sampling step, and classifier-free guidance uses the configured scale. Each bounded sampling batch has an explicit seed-derived PyTorch generator.

Candidates are checked for canonical amino acids, lengths 8–50, duplicates, and exact overlap with training and organizer reference FASTAs before scoring. The local trained ESM multitask property model predicts AMP probability, hemolysis probability, MIC, and HC50. Their deterministic rank-normalized values form the selection score; score ties are ordered lexicographically by sequence. The selected library begins with the ranked top 100, guaranteeing membership.

The candidate target is the configured oversampling multiple of 50,000. A hard batch-attempt ceiling applies. If the target cannot be met, generation reports rejection counts and fails; it never fills with duplicates, known peptides, random sequences, or untrained weights.

## Determinism and validation

Python, NumPy, PyTorch CPU and CUDA seeds are set from the manifest seed. Model loading is strict, uses `map_location`, validates checkpoint format, dimensions, token order, and finite parameters, and sets models to evaluation mode. Model/tokenizer loading is local-only and Hugging Face offline mode is enabled.

The authoritative validator checks FASTA records/headers, exact counts, canonical alphabet, lengths, uniqueness, training/reference overlap, and top-list membership. The novelty threshold uses the exact `Levenshtein.ratio` call and strict `> 0.8` rule from the local challenge template at commit `987afe5c9f73c82d39b0def7acdc56dec31a0132`; see [OFFICIAL_VALIDATOR_SOURCE.md](OFFICIAL_VALIDATOR_SOURCE.md).

## Readiness status

This repository contains no trained CTMC or property checkpoint. It also has no completed training-data manifest. No biological performance, generation speed, or memory measurements are claimed.
