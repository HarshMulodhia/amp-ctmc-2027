# AMP Challenge CTMC Repository — Codex Implementation Specification

## Purpose

This document is an implementation contract for upgrading the existing `amp_ctmc_2027` repository. The coding agent must modify the repository in place; preserve working CTMC functionality; add reproducible data preparation, pretrained protein-model fine-tuning, conditional CTMC training, efficient multi-stage ranking, and official challenge compliance; and add tests and documentation.

The target submission consists of 50,000 unique linear peptide sequences and a ranked top-100 list. Every sequence must contain only the 20 standard proteinogenic amino acids and be 8–50 residues long. The top-100 candidates must also satisfy the organizer’s novelty/identity rule against the competition reference database. Full-track participation additionally requires disclosed data, public weights and inference code, a permissive license, detailed usage documentation, and deterministic generation that the organizers can reproduce by running `uv sync` and the repository entry point.[^1][^2]

## Existing implementation audit

The supplied code already contains useful foundations: a fixed peptide encoder, transformer CTMC denoiser, masked corruption training, length prediction, guided tau-leaping, handcrafted-feature/XGBoost discrimination, multi-objective scoring, FASTA I/O, validation, W&B logging, and deterministic seed setup. The current generation pipeline loads `model.pt`, configuration/tokenizer/statistics/discriminator artifacts; generates a candidate pool; scores it; applies novelty and diversity filters; and writes `library.fasta` and `top.fasta`.[^3]

The implementation is not yet ready for final competition training. The supplied `train.py` hard-codes `batch_size=16` and `n_epochs=1`; splitting is random at the sequence level rather than homology-clustered; the denoiser starts from random weights; the discriminator only sees handcrafted composition features; MIC, bacterial-panel, and HC50 labels are not represented; an external scorer failure silently returns neutral values; expensive scorers are applied to the entire candidate pool; the identity calculation uses normalized Levenshtein similarity rather than a verified copy of the official metric; and `top.fasta` is not explicitly required to be a subset of `library.fasta`.[^3]

Several current behaviors require correction rather than tuning:

- `_sample_t_curriculum` describes low `t` as “low-mask,” but the implemented schedule uses `kappa(t)=sin²(pi t/2)`, so low `t` keeps fewer tokens and is the high-mask regime.
- Validation calls `_loss_for_batch` with default epoch values, which produces a different, strongly skewed time distribution from late-stage uniform training.
- `RealismScorer` evaluates unmasked sequences at clean time and averages maximum softmax confidence. This is not a valid pseudo-likelihood estimate and can reward overconfidence.
- `VendoredScorerClient` silently substitutes neutral scores after failure. A weighted unavailable scorer must fail closed, or it can change the ranking without an obvious error.
- The full library is selected independently from the novelty-filtered top-list path. The final top 100 therefore may not be included in the submitted 50,000 unless explicitly enforced.
- `GreedyMMRSelector` exists but is not used in the final selection path.
- Exact known AMP/training sequences are not rejected by the base validator; only the antibacterial FASTA is placed in its forbidden set.

## Required architecture

Implement a three-stage system rather than placing a 150M-parameter model inside every CTMC reverse step.

1. **Pretrained property model:** fine-tune ESM-2 150M on AMP classification, strain/species-aware MIC prediction, and hemolysis/HC50 prediction.
2. **Conditional CTMC student:** preserve a lightweight CTMC denoiser for affordable generation. Initialize its amino-acid representation from ESM-2 and optionally distill clean-sequence representations/logits from the fine-tuned ESM model. Train it on cleaned AMP sequences with property conditions generated from measured labels where available and teacher predictions otherwise.
3. **Cascaded ranking:** score every candidate with cheap validity, physicochemical, lightweight CTMC, and lightweight surrogate scores; apply ESM/property ensembles only to a reduced shortlist; apply the slowest pseudo-likelihood and optional external models only to the final few thousand.

This design uses the pretrained model meaningfully while keeping generation of 50,000–120,000 candidates computationally feasible. ESM-2 150M is a masked-language protein model with 30 layers and approximately 150M parameters, distributed under the MIT license and intended for task fine-tuning.[^4][^5][^6]

## Dataset implementation

### Source hierarchy

Create a versioned data pipeline with these logical sources:

| Role | Primary source | Use |
|---|---|---|
| Unconditional/AMP generation | Officially supplied challenge training data, if present; otherwise the cleaned AMP-positive portions of HydrAMP/GRAMPA-compatible public data | CTMC sequence modeling |
| AMP negatives | HydrAMP’s non-redundant UniProt/background subset or another disclosed, length-matched public background set | AMP classifier and lightweight discriminator |
| MIC supervision | GRAMPA plus the HydrAMP-processed MIC subset | Species/strain-aware potency prediction |
| Hemolysis supervision | HemoPI2 | Hemolysis classification and HC50 regression |
| Compliance reference | The organizer-provided known-AMP/reference FASTA | Exact novelty and top-100 identity checks only; do not train on hidden/private organizer data |

HydrAMP’s published training collection combines curated AMP sequences, GRAMPA MIC records, and non-redundant UniProt sequences; its sequences are canonical and at most 25 residues in the published setup. The public project provides retraining scripts and downloadable data. GRAMPA contains 6,760 unique sequences and 51,345 MIC measurements with organism/strain associations, but measurements come from heterogeneous sources and duplicate peptide–organism combinations exist. HemoPI2 contains 1,926 unique peptides with experimentally determined hemolytic concentrations and makes its data/code public.[^7][^8][^9][^10][^11]

Do not blindly concatenate APD, DBAASP, DRAMP, HydrAMP, and GRAMPA exports. GRAMPA already aggregates several AMP databases, so naive unioning creates extensive leakage and duplicate weighting. Record source provenance for every row and aggregate only after canonicalization.

### New files

Add:

```text
configs/
  data_sources.yaml
  train_property.yaml
  train_ctmc.yaml
  generate.yaml
scripts/
  download_data.py
  prepare_data.py
  train_property_model.py
  cache_teacher_predictions.py
  train_ctmc.py
  generate_submission.py
  validate_submission.py
src/amp_ctmc_2027/data/
  schema.py
  sources.py
  preprocess.py
  clustering.py
  splits.py
  manifests.py
```

Do not place downloaded raw databases in Git unless their licenses permit redistribution. `data_sources.yaml` must contain source name, exact URL or accession, version/date, expected filename, SHA-256 checksum, license/terms URL, and whether redistribution is permitted. `download_data.py` must be idempotent, verify checksums, and never silently replace an existing mismatched file.

### Canonical schema

Represent processed observations in Parquet, with at least:

```text
sequence: string
sequence_sha256: string
source: string
source_record_id: string|null
is_amp: float|null
organism: string|null
strain: string|null
panel_target: string|null
mic_value: float|null
mic_unit_original: string|null
mic_uM: float|null
mic_censor: {none,left,right,interval}|null
hc50_uM: float|null
hc50_censor: {none,left,right,interval}|null
terminal_modification: string|null
other_modification: string|null
is_linear: bool|null
split_cluster_id: string
split: {train,val,test}
label_provenance: {measured,aggregated,pseudo,missing}
```

Never convert mass concentration to molar concentration without a documented molecular-weight calculation. Preserve inequalities such as `>64` and `>128` using censor fields; do not replace them with ordinary exact values. Preserve organism and strain rather than averaging all MIC measurements into one universal target. Challenge assays define activity at MIC ≤16 µM, use an upper MIC test limit of 64 µM, and measure HC50 up to 128 µM.[^1]

### Cleaning rules

Apply the following deterministic order:

1. Uppercase and strip whitespace.
2. Reject empty strings and characters outside `ACDEFGHIKLMNPQRSTVWY`.
3. Retain lengths 8–50 inclusive.
4. Reject records explicitly marked cyclic, branched, stapled, lipidated, glycosylated, PEGylated, dendrimeric, terminally modified, or containing noncanonical residues.
5. Require free termini when modification metadata are available. If metadata are absent, mark them unknown; do not claim the record is experimentally unmodified.
6. Exact-deduplicate by sequence while retaining a source/observation table with all labels.
7. Aggregate repeated numeric labels with a documented robust rule and retain count, dispersion, and censoring information.
8. Create a conflict report for sequences carrying contradictory AMP labels or incompatible measurements.
9. Remove all official compliance-reference sequences from validation/test contamination analysis, and keep the compliance reference separate from training inputs unless it is independently present in a disclosed public training source.
10. Write a machine-readable `dataset_manifest.json` containing source versions, hashes, filter counts, deduplication counts, split counts, and code commit.

### Leakage-resistant splitting

Replace random splitting with sequence-homology group splitting. Use MMseqs2 or CD-HIT clustering, pin the tool/version and command, and assign complete clusters to train/validation/test. Default to a conservative identity threshold selected in configuration; report results at multiple thresholds rather than claiming one universal cutoff. Assert that exact duplicates and cluster members cannot cross splits.

For property-model evaluation, also create source-held-out or database-held-out diagnostics where feasible. Report classification AUROC/AUPRC and calibration, MIC MAE/RMSE or censored likelihood metrics, and HC50 metrics separately. Do not use random-row splitting of repeated measurements because the same peptide can otherwise occur on both sides.

## Pretrained property model

### Model choice

Use `facebook/esm2_t30_150M_UR50D` as the main backbone. Support `facebook/esm2_t12_35M_UR50D` as the low-compute fallback and optionally 650M only as an ablation. Derive token IDs and hidden size from the loaded tokenizer/config; never hard-code ESM vocabulary IDs or dimensions.[^5][^6]

Add:

```text
src/amp_ctmc_2027/models/
  esm_multitask.py
  property_heads.py
  property_ensemble.py
  distillation.py
src/amp_ctmc_2027/pipeline/
  property_training_pipeline.py
```

Implement `ESMMultiTaskPredictor` with:

- `AutoTokenizer` and `EsmModel`/`AutoModel` loaded from a pinned revision.
- Residue-only masked mean pooling; exclude BOS/EOS/PAD tokens.
- AMP classification head.
- MIC head conditioned on a learned organism/strain embedding. Unknown strains must map to an explicit unknown token.
- Hemolysis classification head and HC50 regression head.
- Optional panel head producing probabilities for the official 20-strain panel only when label mapping is defensible.
- Label masks so missing tasks contribute zero loss.
- Output dataclass containing logits, probabilities, regression means, uncertainty parameters, pooled embeddings, and per-task masks.

Use a masked multi-task loss:

```text
L = lambda_amp * BCE_amp
  + lambda_mic * L_censored_mic
  + lambda_hemo * BCE_hemo
  + lambda_hc50 * L_censored_hc50
  + lambda_cal * calibration_regularizer
```

Implement censor-aware Gaussian NLL or an equivalent survival-style likelihood for right/left-censored MIC and HC50 observations. If this is too large for the first patch, implement uncensored regression first but explicitly exclude censored rows and leave a tested interface for censor-aware loss; never treat `>64` as exactly 64.

Training must support staged unfreezing:

- Warm up heads with the backbone frozen.
- Unfreeze the last configurable ESM blocks.
- Optionally fine-tune all blocks with a much smaller backbone learning rate.
- Support gradient accumulation, AMP/bfloat16, gradient clipping, checkpoint resume, early stopping, and per-head metrics.
- Save tokenizer, model config, exact Hugging Face revision, state dict, data manifest hash, split manifest, training config, and metrics.

Train 3–5 cluster-split folds or independent seeds if compute permits. Expose ensemble mean and standard deviation. Calibration must be fit on validation predictions only.

### Optional external models

APEX is attractive because it predicts pathogen-specific MICs and covers clinically relevant strains, but it must be optional. Include it only if its code, weights, dependencies, and license can be redistributed and executed reproducibly. Document exact supported strains and map names explicitly; never map a species-level prediction to an exact strain without marking it as a fallback. A published review describes APEX as a strain-specific MIC predictor covering 34 pathogens and sequences up to 50 residues.[^12]

Do not add an external model merely as an undocumented weighted score. Every model must have a model card, source/revision, license, input restrictions, output semantics, and a unit test. Optional external oracles must be disabled by default if required artifacts are absent.

## CTMC integration

### Preserve the student generator

Keep `CTMCDenoiser` lightweight. Do not run full ESM-2 once per reverse-time step. Add configurable pretrained transfer:

```text
ctmc_pretraining_mode: none | embedding_init | distill | embedding_init_and_distill
teacher_checkpoint_dir: path|null
teacher_distill_weight: float
teacher_logit_kl_weight: float
condition_dim: int
condition_dropout: float
cfg_scale: float
```

Implement amino-acid embedding initialization by reading the 20 ESM residue embeddings, projecting them to `d_model` when dimensions differ, and copying only semantically corresponding amino-acid rows. Initialize CTMC `<EOS>`, `<MASK>`, and `<PAD>` separately. Save the mapping and source checkpoint revision in checkpoint metadata.

For distillation, cache teacher outputs for clean training peptides in a versioned memory-mapped/Parquet artifact. Train a projection from teacher pooled/residue representations to the CTMC hidden space. The distillation term must be masked to real residues, detached from the teacher, and configurable. Teacher inference must not run repeatedly inside every CTMC epoch.

### Conditional generation

Add a `ConditionVector` containing soft targets and an observation mask. Minimum conditions:

```text
amp_probability
broad_spectrum_probability
mean_activity_probability
mean_gram_negative_activity_probability
mean_gram_positive_activity_probability
mean_mdr_activity_probability
hemolysis_probability
predicted_log2_mic_summary
predicted_log2_hc50
condition_is_measured mask for every field
```

Measured labels take precedence. Missing conditions may be filled by calibrated teacher predictions and must be marked `pseudo`. Store pseudo-label model hash and uncertainty. Never mix pseudo-labels with measured labels without provenance.

Inject time and condition embeddings by FiLM/AdaLN or additive conditioning in each transformer block. Train with classifier-free condition dropout. At inference compute conditional and unconditional logits and combine them as:

```text
logits_guided = logits_unconditional
              + cfg_scale * (logits_conditional - logits_unconditional)
```

Retain an unconditional path for ablations and recovery if conditional training degrades diversity.

### Time/corruption corrections

Define and document one orientation everywhere:

- `t=0`: fully masked/noise endpoint.
- `t=1`: clean data endpoint.
- `kappa(t)` is the probability that a clean token remains revealed.

Rename ambiguous methods or add explicit docstrings. Replace the current curriculum with a configuration-driven sampler and use the same fixed evaluation distribution every epoch. Recommended validation: deterministic stratified `t` bins covering the full interval. Log token accuracy and cross-entropy per time bin.

The corruption function must never modify PAD. Decide explicitly whether EOS is modeled. Prefer a canvas that always represents sequence length unambiguously: either reserve `max_length+1` positions for up to 50 residues plus EOS, or sample length first and use exactly 50 residue slots with PAD beyond the sampled length. Do not preserve the current edge case in which length-50 peptides have no EOS while shorter peptides do. Add a checkpoint version and migration guard because changing canvas semantics invalidates old checkpoints.

### CTMC losses and diagnostics

Retain masked-token cross-entropy and the length loss, and add optional distillation and condition-consistency losses. Log:

- Total and component losses.
- Masked-token accuracy by `t` bin.
- EOS/length accuracy.
- Amino-acid marginal KL between generated and training data.
- Unique fraction and valid fraction from periodic samples.
- Nearest-reference similarity distribution.
- Teacher property predictions on periodic generated samples.
- Gradient norm and learning rate.

Add a 64-example overfit test. It must achieve a clearly documented near-perfect reconstruction regime before full training is launched.

## Scoring and selection

### Replace scoring semantics

Refactor `objectives.py` so each score declares:

```text
name
direction: maximize|minimize
required_artifacts
cost_tier: cheap|medium|expensive
batch_size
raw_units
failure_policy
```

Return a `ScoreTable` with every raw score, calibrated score, rank-normalized score, uncertainty, model version, and failure state. Save it as Parquet/CSV for auditability.

Replace `RealismScorer` with masked pseudo-log-likelihood or remove it. If pseudo-log-likelihood is used, mask one residue or blocks of residues and compute the observed-token log probability. Apply it only to a shortlist because exact residue-wise ESM PLL is expensive.

Split `ActivityHemolysisScorer` into separate activity, MIC, hemolysis, HC50, and selectivity components. A product such as activity × (1 − hemolysis risk) can remain as a derived score but must not hide the raw predictions.

### Cascaded execution

Implement configurable stages:

1. **All candidates:** hard validity, exact deduplication, length, cheap physicochemical features, lightweight discriminator/CTMC score.
2. **Medium shortlist:** CTMC surrogate/property heads and ESM multitask ensemble in batches.
3. **Final shortlist:** masked pseudo-likelihood, optional APEX/external ensemble, uncertainty-aware panel metrics.
4. **Selection:** quality/uncertainty constraints followed by diversity-aware MMR or clustering.

Never send 120,000 sequences in one JSON subprocess request. Chunk requests, validate returned lengths and finite values, and use a disk cache keyed by `(model_hash, sequence_hash, scorer_config_hash)`.

If a component has nonzero weight and fails, abort. Neutral fallback is allowed only when the component weight is zero and the run manifest records that it was skipped.

### Ranking targets

For each candidate estimate:

- `p_overall_success = mean_j P(MIC_j <= 16 uM)` over the official 20 strains.
- Gram-positive, Gram-negative, and MDR subset equivalents.
- Predicted MIC50/MIC90 summaries, with uncertainty.
- Hemolysis risk and predicted HC50.
- Selectivity proxy based on HC50 and MIC50.
- Novelty and diversity using the organizer-verified identity implementation.

Missing strain predictions must not be silently imputed as active. Use an explicit availability mask and penalize uncertainty/coverage gaps. Optimize score weights on validation or retrospective held-out data, not on the candidate pool’s own rank distribution.

## Compliance integration

The challenge website states that automated screening checks alphabet, length, duplicates, metadata, data disclosure, and repository licensing. The full library is evaluated for diversity, novelty, and physicochemical distributions, while top-100 candidates exceeding the identity threshold against the organizer reference are invalid/replaced.[^1]

Implement the following:

1. Vendor or import the official validator and official sequence-identity function from the challenge template at a pinned commit.
2. Do not call RapidFuzz normalized edit similarity “sequence identity” unless the official validator uses exactly that definition.
3. Maintain the organizer reference FASTA as an immutable input with SHA-256 in the run manifest.
4. Run compliance before expensive scoring, after every filtering stage, and on final files.
5. Enforce `set(top_100).issubset(set(library_50000))`.
6. Enforce exactly 50,000 unique library sequences and exactly 100 unique ranked candidates.
7. Enforce canonical alphabet, lengths 8–50, no empty sequences, and deterministic FASTA identifiers/order.
8. Enforce the official top-100 identity rule with the exact organizer metric and threshold semantics.
9. Produce a `compliance_report.json` with counts, maximum identity, nearest-reference IDs, duplicate counts, invalid-character counts, length histogram, hashes, and pass/fail.
10. Add a test that intentionally inserts a duplicate, noncanonical residue, length violation, known sequence, and threshold violation and verifies rejection.

Use the repository’s current stricter novelty cutoff only as a tunable design filter. It must not substitute for the official compliance check.

## File-by-file changes

### `src/amp_ctmc_2027/config.py`

- Split configuration into nested `DataConfig`, `PropertyModelConfig`, `CTMCConfig`, `ScoringConfig`, `GenerationConfig`, and `ComplianceConfig`, while retaining backward loading for old JSON when practical.
- Add all paths/revisions/hashes, conditioning, distillation, cluster split, mixed precision, staged scoring, failure policies, and official validator settings.
- Validate score weights, enabled components, model-artifact existence, stage sizes, and shortlist monotonicity.
- Remove the claim that `max_length=50` implies a canvas length of exactly 50; model canvas semantics separately.
- Save a fully resolved config in every run.

### `src/amp_ctmc_2027/data/dataset.py`

- Extend samples from a tensor to a dataclass/dict containing tokens, attention mask, sequence length, condition values, condition-observed mask, label provenance, and optional teacher-cache key.
- Fix EOS/length-50 semantics and version tokenizer metadata.
- Add a collator with documented shapes: tokens `[B,L]`, attention mask `[B,L]`, conditions `[B,C]`, condition mask `[B,C]`.
- Keep a minimal sequence-only dataset for backward-compatible smoke tests.

### `src/amp_ctmc_2027/core.py`

- Preserve public CTMC classes where possible.
- Add checkpoint versioning, ESM embedding initialization, condition encoder, classifier-free guidance, and distillation hooks.
- Use explicit `torch.Generator` objects in sampling; do not rely on mutable global RNG.
- Restrict logits so PAD/MASK/EOS cannot appear as amino acids in invalid positions.
- Make length/EOS handling explicit and tested.
- Return sampler diagnostics including reveal counts, invalid-token attempts, and termination state.

### `src/amp_ctmc_2027/pipeline/training_pipeline.py`

- Load prepared split manifests rather than randomly splitting raw FASTA.
- Remove hard-coded assumptions about FASTA-only examples.
- Fix time sampling and deterministic validation.
- Support resume, AMP, gradient accumulation, separate learning rates, teacher cache, periodic generation, and best-checkpoint selection by a composite validation metric.
- Save manifests and hashes beside weights.
- Train the XGBoost baseline separately; do not conflate it with CTMC training.

### `src/amp_ctmc_2027/train.py`

- Replace hard-coded one-epoch configuration with `--config`, `--resume`, and selected overrides.
- Default to a checked-in production config, not debug settings.
- Add a `--smoke-test` mode that is clearly non-production.

### `src/amp_ctmc_2027/discriminator.py`

- Retain the handcrafted XGBoost model as a cheap baseline only.
- Train with cluster-aware folds and balanced/length-matched negatives.
- Add held-out metrics and calibration.
- Include feature/model version in serialization.
- Avoid nondeterministic settings where exact reproduction is required.

### `src/amp_ctmc_2027/external_scorer.py`

- Replace silent neutral fallback with configurable `raise|skip_if_zero_weight` behavior.
- Add chunking, schema validation, finite/range checks, persistent cache, model hash, retry diagnostics, and timeout per chunk.
- Require local weights at inference and avoid network downloads.

### `src/amp_ctmc_2027/objectives.py`

- Add typed score results, direction, cost tier, uncertainty, and provenance.
- Remove/replace the current clean-confidence realism score.
- Add ESM property, MIC panel, hemolysis, HC50, selectivity, synthesis-risk, and official novelty components.
- Use MMR or cluster-constrained selection for top 100.
- Add uncertainty penalty and score coverage checks.
- Retain raw scores in an audit table.

### `src/amp_ctmc_2027/pipeline/generation_pipeline.py`

- Use staged candidate scoring and chunked persistence so interrupted runs can resume.
- Apply official compliance metric/reference.
- Make top 100 a ranked subset of the final 50,000 library.
- Build the library with diversity-aware selection rather than simply taking the first 50,000 global-score rows.
- Write `library.fasta`, `top.fasta`, `scores.parquet`, `run_manifest.json`, and `compliance_report.json` atomically.
- Check hashes on every required artifact.
- Verify determinism by optionally running a small duplicate generation pass.

### `src/amp_ctmc_2027/generate.py`

- Match the official template entry point exactly.
- Keep the default seed fixed.
- Provide `--config`, `--checkpoint`, `--output-dir`, and `--device` without changing no-argument behavior expected by organizers.
- Never train or download from the network during generation.

### `src/amp_ctmc_2027/infra.py`

- Extend `ConstraintValidator` with the official identity validator through composition, not an approximate duplicate implementation.
- Add artifact hashing, run manifests, environment capture, atomic Parquet/JSON writes, and structured logging.
- Ensure W&B is disabled or offline by default for organizer inference.

### `pyproject.toml` and lockfile

- Pin Python and direct dependencies, including compatible PyTorch/Transformers/PEFT versions if used.
- Define official console scripts.
- Keep `uv.lock` committed.
- Put training-only tools in dependency groups where possible.
- Ensure `uv sync` is sufficient on a clean machine and inference does not import undeclared packages.

### Documentation and repository assets

Add/update:

```text
README.md
LICENSE
MODEL_CARD.md
DATA_CARD.md
TRAINING.md
SUBMISSION.md
THIRD_PARTY_NOTICES.md
weights/ or documented release/LFS pointers with checksums
```

README must show the exact organizer reproduction command first, expected runtime/hardware, output paths, and hash verification. DATA_CARD must list every source, version, license/terms, filters, counts, overlap policy, and split method. MODEL_CARD must explain pretrained initialization, tasks, limitations, intended use, and calibration. SUBMISSION must disclose manual intervention (preferably none), computational filters, external models, and ranking procedure.

## Tests

Add unit, integration, and slow tests:

```text
tests/test_preprocess.py
tests/test_unit_conversion.py
tests/test_cluster_split.py
tests/test_encoder.py
tests/test_esm_multitask.py
tests/test_censored_losses.py
tests/test_ctmc_conditioning.py
tests/test_time_schedule.py
tests/test_sampler_determinism.py
tests/test_score_failures.py
tests/test_official_compliance.py
tests/test_generation_smoke.py
tests/test_submission_membership.py
```

Minimum acceptance checks:

- Cleaning is deterministic and counts match a saved fixture manifest.
- No exact sequence or cluster crosses train/validation/test.
- Every model forward pass has asserted tensor shapes and finite outputs.
- Missing labels contribute no gradient to their task losses.
- Censored-loss toy cases move in the expected direction.
- Conditional and unconditional CTMC passes both work.
- Same seed and artifacts produce byte-identical small FASTA outputs on the supported reference environment.
- Different seeds produce different outputs.
- External scorer failures cannot silently affect nonzero-weight rankings.
- Final top 100 is a ranked subset of the 50,000 library.
- The official compliance validator passes final outputs.
- Generation works with network disabled and `WANDB_MODE=disabled`.

## Implementation order

### Milestone 0 — Baseline lock

Before refactoring, run existing tests, add a small deterministic generation fixture, and record current checkpoint/config formats. Add checkpoint version `1` to current artifacts. Do not delete the old path until migration tests exist.

### Milestone 1 — Compliance correctness

Integrate the official validator/identity function, enforce top-in-library membership, remove silent scorer fallback, add final reports, and fix the CLI/reproducibility path. This milestone must land before model improvements.

### Milestone 2 — Data pipeline

Implement manifests, cleaning, unit/censor handling, exact deduplication, cluster splitting, prepared Parquet tables, and data-card generation. Produce filter/split reports. Stop if licenses or modification metadata are unresolved.

### Milestone 3 — Property baseline

Train the existing XGBoost baseline with leakage-resistant splits. Then implement ESM-2 35M smoke training and ESM-2 150M production training. Compare against the handcrafted baseline on fixed cluster splits; do not claim improvement without held-out metrics.

### Milestone 4 — CTMC transfer

Implement ESM amino-acid embedding initialization, teacher caches, conditions, classifier-free guidance, corrected time validation, and checkpoint v2. Pass the 64-example overfit test and unconditional-regression tests.

### Milestone 5 — Cascaded ranking

Add property ensembles, staged scoring, disk cache, raw score tables, uncertainty-aware ranking, MMR selection, and final library diversity controls. Benchmark memory and wall time.

### Milestone 6 — Reproducible release

Pin dependencies and model revisions, include weights/checksums, run generation twice in a clean environment with network disabled, compare hashes, run all compliance tests, and finish model/data/submission documentation.

## Required ablations

Keep experiments focused:

| Ablation | Question |
|---|---|
| Random CTMC vs ESM embedding initialization | Does pretrained residue geometry improve denoising/sample quality? |
| Initialization vs initialization + distillation | Does teacher transfer add value beyond embeddings? |
| Unconditional vs property-conditioned CTMC | Does conditioning improve predicted activity without collapsing diversity? |
| 35M vs 150M teacher | Is the larger teacher worth its compute? |
| Handcrafted/XGBoost vs ESM property model | Does ESM improve cluster-held-out prediction and calibration? |
| Single scorer vs ensemble/uncertainty penalty | Does uncertainty-aware ranking improve held-out target metrics? |
| Global score selection vs MMR/cluster selection | What quality–diversity trade-off is achieved? |

For each experiment, report fixed data/splits/seeds, compute, checkpoint, validity, uniqueness, novelty distribution, pairwise diversity, property predictions, and calibration. Do not use the competition’s final reference screening outcome to tune training.

## Non-goals and safety rails

- Do not replace the CTMC with a generic autoregressive generator.
- Do not make ESM-2 a mandatory per-step reverse-process call.
- Do not train on the organizer’s hidden/private evaluation data.
- Do not treat database absence as a negative AMP label.
- Do not treat missing strain MICs as inactivity.
- Do not call approximate edit similarity “sequence identity.”
- Do not silently download model weights during official inference.
- Do not silently continue when a weighted scorer is unavailable.
- Do not commit data or weights whose redistribution terms are unclear.
- Do not fabricate dataset sizes after filtering; generate them from manifests.
- Do not hard-code reported performance. Metrics must come from saved evaluation artifacts.

## Definition of done

The repository is complete only when a clean checkout can run `uv sync`, load all local/checksummed artifacts, execute the official no-argument generation entry point without network access, and deterministically produce the required 50,000-sequence library and ranked top-100 subset. Both files must pass the official validator; the top list must satisfy the official identity rule; all model/data provenance and licenses must be documented; and tests must cover data leakage, condition masking, CTMC schedule semantics, scorer failures, deterministic sampling, output membership, and final compliance.[^2][^1]

---

## References

1. [AMP Challenge | Generative AI for Antimicrobial Peptide Design](https://szczurek-lab.github.io/amp-challenge-website/) - The first head-to-head benchmarking competition for generative AI in antimicrobial peptide design.

2. [szczurek-lab/amp-challenge-2027](https://github.com/szczurek-lab/amp-challenge-2027) - Antimicrobial resistance is one of the most pressing global health challenges. This competition invi...

3. [paste.txt](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/88949578/c87952df-0285-4cde-b04a-e16c6dec44b7/paste.txt?AWSAccessKeyId=ASIA2F3EMEYEZX7ORYD7&Signature=jUZhLQILAtYARjc8vSdKBEgGrRc%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEGkaCXVzLWVhc3QtMSJHMEUCIQDHmDOH5PM7rgvD%2FAPQH%2B0VwWW0eF1%2FMfRDL8ohwk6TsQIgLSoOTqIO4vohAEqEk8QTcSb3fXB%2BXlUiFbGwmDoVQiYq8wQIMRABGgw2OTk3NTMzMDk3MDUiDO1pR3nHkoauMjozMCrQBJVUdYmWvkQho2ETGbV48lqz1TXq5VTrfCf4QpiXcMDTZTqK1L1vagHxPp3CZ71vLT6%2Fgt3ed6bohqAgiV%2FWX95NHGYv1Yvvglsmqv1p1xdbkS3eCbjUDVngAL5WD7ff1W0Kco54e7QHBhiyRgsi0Owha9GJwT2rcm8WlLjUr14AiIjj1F5X4yHUBn05UGBN6Lk8HNPpu6zm412EfG78evH1X%2FGva%2FHxnF%2B9ApfJ740RoAAWfX9yPRjgQ2PT4Vpa5WCejbmwh3O8ppKhSrFYn6iuSPfKSbkqvMg4UrL2a0CEu0tziLgegpbLJWr4QFndNbjn1qwH91ceE4hYXMnb1mgukpHN4t2lmRxj%2FOkkq7GaMZ%2F5AgTtoUmnjcmhE330GTqc2qSubSFI60VGie5cO0GOQuiFuHZ8T%2Bnp3%2B6mD6xbEuVeUDOGB8NStL9U9Ri02PkWeJUsBqLTsU6rjtEvf%2BOcajXJbBVPRzZmkmnvbJq5CaVTUOQTURVxwcEeyBHgGj3bNbiaKxF7Ydp7oGepvxn7RM8IZLmLYWo8klhO%2FTHXPQaytGO4joaaztn3W2FO84bM2rHMhAHWZM7cG40Lmv%2FwfA9AiWEoLKQfBjaLwR1KBgAlWVlv3mxDGNS4EoNDySaiP%2BDkUcAd4lj9%2BvxyO3pQnL%2BKWnoAPqLdUtqeHdz5OxVZok%2FVXYeLCOtVvbfKwcYK7C1NXHWVVyHuWQz6hV2s3enVHa313BxK7TqHSp556alyNtf89Zlz3yHwe9jrQF2qy5BEaLAaOuKOvSyL8HEwjcfo1QY6mAExGiP1vMazetdzF2yD%2BIyZWZ2a1%2B1aXn71EhkmkpPySIIZVjMGO9WEYm7HEvAfAe90xdteU2hon8qi1Di6XHy8fnU%2FOSwTjONfcmndaRUtGmTP6MzUDAOVWRt7PPYbfW%2FDps0cUUppPize3AIQv7aP6QXoQ6J7Gr9vqNJGIGPfPafWPvf2EnFzgr3ypdrfruHUI1SBsd%2BRmQ%3D%3D&Expires=1790587232) - rootphysical-aistackinferencehmapamp-ctmc-2027 cat srcampctmc2027discriminator.py Discriminator for ...

4. [nvidia/esm2_t30_150M_UR50D](https://huggingface.co/nvidia/esm2_t30_150M_UR50D) - ESM-2 is a state-of-the-art protein model trained on a masked language modelling objective. ESM-2 is...

5. [README.md · facebook/esm2_t30_150M_UR50D at ...](https://huggingface.co/facebook/esm2_t30_150M_UR50D/blame/791c7a473b4c5396d2e184311e56b47e426a5d5a/README.md) - ESM-2 is a state-of-the-art protein model trained on a masked language modelling objective. It is su...

6. [facebook/esm2_t30_150M_UR50D](https://huggingface.co/facebook/esm2_t30_150M_UR50D) - ESM-2 is a state-of-the-art protein model trained on a masked language modelling objective. It is su...

7. [Deep learning regression model for antimicrobial peptide ...](https://www.biorxiv.org/content/10.1101/692681v1.full-text) - GRAMPA contains 6760 unique sequences, and 51345 total MIC measurements. GRAMPA publically available...

8. [AMPCliff: Quantitative definition and benchmarking of activity cliffs in ...](https://pmc.ncbi.nlm.nih.gov/articles/PMC12869253/) - An overview of the quantitative definition and benchmarking of AMPCliff. a-c are the model architect...

9. [HydrAMP: a deep generative model for antimicrobial peptide discovery](https://github.com/szczurek-lab/hydramp) - HydrAMP: a deep generative model for antimicrobial peptide discovery - szczurek-lab/hydramp

10. [Discovering highly potent antimicrobial peptides with deep ... - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10017685/) - Antimicrobial peptides emerge as compounds that can alleviate the global health hazard of antimicrob...

11. [Prediction of hemolytic peptides and their ... - PMC - NIH](https://pmc.ncbi.nlm.nih.gov/articles/PMC11794569/) - Peptide-based drugs often fail in clinical trials due to their toxicity or hemolytic activity agains...

12. [Reviewing the role of artificial intelligence in accelerating the ...](https://academic.oup.com/jambio/article/137/3/lxag036/8445386) - Abstract. Antimicrobial resistance (AMR) is one of the most critical public health threats of the 21...
