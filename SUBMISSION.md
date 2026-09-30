# AMP-CTMC-150M: Property-Conditioned Continuous-Time Markov Generation of Antimicrobial Peptides

## Participation

**Requested tier:** Full (co-authorship eligibility)  
**Repository:** https://github.com/HarshMulodhia/amp-ctmc-2027  
**Frozen revision:** Git tag `full-submission-2026-09-30`
**Reproducibility seed:** `42`

## Abstract

This submission combines a property-conditioned continuous-time Markov chain (CTMC) peptide generator with a fine-tuned ESM-2 150M multitask activity and safety predictor. The generator models canonical linear peptide sequences on an explicit residue-plus-EOS canvas and is conditioned on nine antimicrobial activity and hemolysis-related targets. Public measured labels take precedence over cached model predictions, while pseudo-labels fill missing conditioning fields. Generated candidates are filtered for structural validity, uniqueness, and organizer-reference novelty, then ranked using worst-strain predicted MIC, conservative broad-spectrum activity, mean activity, AMP probability, predicted hemolysis, and predicted HC50. The frozen seed-42 pipeline deterministically produces 50,000 unique peptides and a ranked top-100 subset without manual peptide selection. The repository contains trained weights, pinned data/model provenance, deterministic inference code, validation scripts, and frozen output hashes.

## Method

### Property model

The property predictor starts from `facebook/esm2_t30_150M_UR50D`, pinned to revision `a695f6045e2e32885fa60af20c13cb35398ce30c`. The local immutable snapshot was fine-tuned for two epochs with batch size 16, maximum tokenized length 64, mixed precision, and the final two transformer layers unfrozen. Sequence representations feed multitask heads for AMP probability, strain-conditioned MIC, hemolysis probability, and HC50.

MIC is predicted across an 11-strain public-data proxy panel. Predicted log2 MIC is converted into activity probability using a threshold of 16 micromolar (log2 MIC 4) and temperature 1.0. Activity probabilities are aggregated over the full, Gram-negative, Gram-positive, and MDR panels. Broad-spectrum probability is the minimum per-strain activity probability, deliberately penalizing a candidate that is predicted to be weak against any modeled strain. This 11-strain proxy is not claimed to replace the official 20-strain wet-lab panel.

### CTMC generator

The generator is a discrete property-conditioned CTMC over 20 canonical amino acids plus EOS, mask, and padding states. Its denoiser uses:

- 10 transformer layers
- Hidden dimension 512
- 16 attention heads
- Feed-forward dimension 2,048
- Dropout 0.10
- Maximum peptide length 50
- A 51-position residue-plus-EOS canvas
- Batch size 64
- Learning rate 0.0003 with cosine decay
- 20 training epochs
- Classifier-free condition dropout 0.15
- Seed 42

The nine conditions are AMP probability, broad-spectrum probability, mean activity probability, Gram-negative activity probability, Gram-positive activity probability, MDR activity probability, hemolysis probability, mean predicted log2 MIC, and predicted log2 HC50. Measured values override model-derived values field by field. Missing conditions are filled from a frozen offline ESM-2 prediction cache rather than recomputing the language model during CTMC training.

### Generation

Generation uses 64 reverse CTMC steps and classifier-free guidance scale 1.5. The target vector is:

| Condition | Target |
|---|---:|
| AMP probability | 0.95 |
| Broad-spectrum probability | 0.75 |
| Mean activity probability | 0.85 |
| Gram-negative activity probability | 0.85 |
| Gram-positive activity probability | 0.85 |
| MDR activity probability | 0.80 |
| Hemolysis probability | 0.05 |
| Predicted log2 MIC summary | 4.0 |
| Predicted log2 HC50 | 7.0 |

Candidates are decoded, structurally validated, deduplicated, compared with the organizer reference, and scored by the frozen property model. Deterministic additional batches are sampled until exactly 50,000 valid unique sequences are available.

### Ranking

The top 100 are selected using a deterministic weighted score:

| Component | Weight | Preferred direction |
|---|---:|---|
| Worst predicted log2 MIC | 0.22 | Lower |
| Broad-spectrum probability | 0.20 | Higher |
| Mean activity probability | 0.14 | Higher |
| AMP probability | 0.14 | Higher |
| Hemolysis probability | 0.12 | Lower |
| Predicted log2 HC50 | 0.08 | Higher |
| Gram-negative activity probability | 0.04 | Higher |
| Gram-positive activity probability | 0.03 | Higher |
| MDR activity probability | 0.03 | Higher |

Worst-strain MIC and minimum per-strain activity prevent strong predictions for a subset of strains from hiding one predicted weak strain. Deterministic sequence ordering resolves exact score ties.

## Training data

Only public data were used. No proprietary, private, or unpublished peptide data were used.

| Source | Use | Frozen source |
|---|---|---|
| AMP-Diffusion `train_eval.csv` | AMP-positive corpus and 11-strain pseudo-MIC labels | `programmablebio/amp-diffusion`, commit `a59de9e95ef13d684d940025665c5783598367b9` |
| GRAMPA | Public measured MIC observations and AMP-positive sequences | `zswitten/Antimicrobial-Peptides`, commit `a6819dcd671291af8013627a2d2fbc32858346bc` |
| Cleaned hemolysis data | HC50 observations | Same pinned Antimicrobial-Peptides commit |
| HemoPI2 | Hemolysis classification and concentration observations | `raghavagps/hemopi2`, commit `2b67a5c85422b25ae847100ebaa81ad586950928` |
| AMPlify | AMP/non-AMP classification supervision | Zenodo record 7320306 |
| Organizer `antibacterial.fasta` | Compliance screening only; never used as supervised training data | Organizer-provided reference |

The prepared data contain 243,823 normalized observation rows, 19,541 unique canonical linear AMP-positive CTMC sequences, and 2,974 unique non-AMP background sequences. They include 27,116 public measured MIC rows, 207,460 AMP-Diffusion pseudo-MIC rows, 3,247 measured HC50 rows, and 1,908 measured hemolysis rows.

AMP-Diffusion MIC outputs are explicitly treated as pseudo-labels, not experimental measurements. GRAMPA and hemolysis/HC50 database rows retain public measured provenance. Measured conditions override pseudo-labels; pseudo-labels fill only missing fields.

Sequences containing noncanonical residues, invalid lengths, terminal modifications, or uncertain modification status were excluded and retained in audit files. All observations for one exact sequence were assigned to the same deterministic split. The deadline split is exact-sequence disjoint, not homology-disjoint, and this is an acknowledged limitation.

## Filtering and human intervention

The pipeline enforces:

- Only `ACDEFGHIKLMNPQRSTVWY`
- Length 8–50, inclusive
- Linear peptides with unmodified termini
- No duplicate sequences
- No full-library exact matches to the organizer antibacterial reference
- The organizer identity ceiling for every top-100 peptide
- Every top-100 sequence must belong to the 50,000-sequence library

No candidate was manually edited, substituted, or hand-selected. All filtering, ranking, and tie-breaking were performed by the frozen deterministic pipeline.

## Reproducibility

```bash
git clone https://github.com/HarshMulodhia/amp-ctmc-2027.git
cd amp-ctmc-2027
git fetch --tags
git checkout full-submission-2026-09-30
git lfs install
git lfs pull
uv sync
uv run generate
uv run python scripts/validate_submission.py
uv run python scripts/rehearse_submission.py
```

### Frozen hashes

```text
4770d5e26baa35269e24069dff934943b46a4cbc06961f66f955d8e966c54183  generate/library.fasta
2e8e01ab83c4c9a8eda2e85b1d3b68c8a37b2f904c04299b6e30f7f493c9ec84  generate/top.fasta
9906fc093e9985438de30f52851be40ed65e9f4bc1367e789681bdf10721b895  generate/scores.csv
0910981748c82b39dd159ba111d87a41ac7db971f7491e7df1fe5147ca5d3051  artifacts/manifest.json
12fd4d0f14469957303a66709df071fc9df96c6dcae825900286dd6242673eeb  artifacts/conditions/deadline_conditions.jsonl
```

A clean clone with Git LFS, dependency synchronization, generation, and validation reproduced these hashes. The deterministic rehearsal generated the library and top 100 twice and obtained identical outputs.

## Limitations

- Conditioning covers an 11-strain public proxy panel, not the complete official 20-strain wet-lab panel.
- Pseudo-MIC values can transfer biases and calibration errors from their source predictor.
- Exact-sequence splitting prevents exact duplicates across splits but does not guarantee homology-disjoint evaluation.
- Internal scores are model predictions, not experimental evidence of potency or safety.
- Public AMP and hemolysis databases combine heterogeneous assays and protocols.
- Sequence diversity does not guarantee mechanistic or structural diversity.

## Eligibility statement

This entry requests Full participation and co-authorship eligibility. The public template-compatible repository contains trained weights, deterministic inference code, documentation, a permissive license for original project code, public training-data disclosure, third-party notices, and reproducible output hashes. No proprietary or non-public training data were used.

## Contributor statement

**Harsh Mulodhia, Drona Manan Bajaj:** Conceptualization; methodology; software; data curation; model training; validation; formal analysis; reproducibility; documentation; submission preparation.

**Drona Manan Bajaj contact:** dronabajajofficial@gmail.com; GitHub: [@dbajaj123](https://github.com/dbajaj123).

Add only genuine contributors and their actual roles. Do not list competition organizers as contributors merely because they have repository access.
