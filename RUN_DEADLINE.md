# Deadline training run

## Hardware and scope

The defaults target one NVIDIA RTX 4090 (24 GB VRAM), a recent CUDA runtime,
and enough host RAM/disk for the ESM-2 150M snapshot and source tables. Property
fine-tuning uses the pinned local 150M snapshot (`configs/deadline_property_150m.json`),
token length 64, mixed precision when CUDA is available, and two unfrozen
transformer blocks for two epochs. CTMC uses batch 64 and 20 epochs. These are
production settings; the `smoke` stage is separate and does not overwrite
production artifacts.

## Pinned sources and terms

The download catalog is `config/deadline_sources.json`. A public URL does not
itself grant redistribution or training rights; users must review each source's
terms and cite the original work. The repository downloads only to the local
data directory.

| Source | Pinned URL / revision | SHA-256 | Intended use and terms warning |
|---|---|---|---|
| AMP-Diffusion training/evaluation | [train_eval.csv](https://raw.githubusercontent.com/programmablebio/amp-diffusion/a59de9e95ef13d684d940025665c5783598367b9/note/eval/train_eval.csv), `a59de9e95ef13d684d940025665c5783598367b9` | `424e1ae8b40bb0e6367251be930caf704673d27b0e583082a119f77d89e48a97` | AMP corpus and pseudo-MIC labels. Dataset-specific redistribution terms are not encoded by this data file; review [upstream repository](https://github.com/programmablebio/amp-diffusion). |
| GRAMPA | [grampa.csv](https://raw.githubusercontent.com/zswitten/Antimicrobial-Peptides/a6819dcd671291af8013627a2d2fbc32858346bc/data/grampa.csv), `a6819dcd671291af8013627a2d2fbc32858346bc` | `9e33122afb3bdd169e8bed620b1040bf1601d60cf25b6c8c8d9870bf37522b6f` | Measured public MIC source; review [upstream terms](https://github.com/zswitten/Antimicrobial-Peptides). `value` is log10(MIC µM), converted once as `10**value`. |
| Cleaned HC50 regression | [CSV](https://raw.githubusercontent.com/zswitten/Antimicrobial-Peptides/a6819dcd671291af8013627a2d2fbc32858346bc/Hemolysis/Cleaned_hemolytic_data.csv), same commit | `adee9ec2d1cacbfb26db35984ac1e2914d25927d425a76af016258745fd5cf1c` | Review the same upstream terms. `log10_HC50` is converted once to µM. |
| HemoPI2 cross-validation | [CSV](https://raw.githubusercontent.com/raghavagps/hemopi2/2b67a5c85422b25ae847100ebaa81ad586950928/Dataset/cross_val_dataset.csv), `2b67a5c85422b25ae847100ebaa81ad586950928` | `7bdaf3ede499d1eda2712585d2e52d7700f3f138776d1a0a46e2ca88e8152da0` | Hemolysis/HC50 training data; review [HemoPI2 terms](https://webs.iiitd.edu.in/raghava/hemopi2/) and cite its paper. Labels are preserved as documented. |
| HemoPI2 independent | [CSV](https://raw.githubusercontent.com/raghavagps/hemopi2/2b67a5c85422b25ae847100ebaa81ad586950928/Dataset/independent_dataset.csv), same commit | `500013c2244219762ff3ff4a03401c7419790c83d0ef0c3aeebfdbea426b3eb5` | Same terms and label handling as cross-validation data. |
| AMPlify AMP positives | [FASTA](https://zenodo.org/api/records/7320306/files/AMPlify_AMP_train_common.fa/content), Zenodo record `7320306` | `a04e28f8d29d1bb4f445a6162e210e0999289c31bf93f2f13c8d2268c8dd9cdc` | AMP classifier positives and positive CTMC corpus; verify record-level license/terms at [Zenodo](https://zenodo.org/records/7320306). |
| AMPlify non-AMP negatives | [FASTA](https://zenodo.org/api/records/7320306/files/AMPlify_non_AMP_train_balanced.fa/content), same record | `02161c5d18ff8c5c8c712527fb84a9c51606a46b11ad987cd91c60306acd25a3` | Classifier negatives and CTMC background only; same Zenodo terms. Never included in positive FASTA. |
| Organizer reference | Existing `data/antibacterial.fasta` | recorded in download manifest | Compliance checks only; not model supervision or background. Do not replace it. |

The fixed ESM-2 snapshot is `facebook/esm2_t30_150M_UR50D` revision
`a695f6045e2e32885fa60af20c13cb35398ce30c` from [Hugging Face](https://huggingface.co/facebook/esm2_t30_150M_UR50D/tree/a695f6045e2e32885fa60af20c13cb35398ce30c).
Review the model card and license before redistribution. Its immutable starting
copy is stored at `data/pretrained/esm2_t30_150M_UR50D/` (Git LFS); fine-tuned
inference files are exported to
`artifacts/property/{backbone,tokenizer,model.pt,model_metadata.json}`.

## Bootstrap

One command downloads and prepares all training data and the local model:

```bash
./scripts/run_deadline_pipeline.sh bootstrap
```

Equivalent explicit commands, useful for reruns:

```bash
uv run python scripts/bootstrap_deadline_data.py
uv run python scripts/prepare_deadline_training_data.py
# Optional: verify or refresh the bundled 150M snapshot
uv run python scripts/download_deadline_model.py --offline-check
```

Data is placed under `data/raw/deadline/`, `data/processed/`, `data/training/`,
and `data/audit/`. The bootstrap verifies every published SHA-256 before it
renames a `.part` file. It refuses a hash mismatch and never substitutes a
source. The organizer reference remains in its existing location.

The CTMC conditioning panel is an **11-strain proxy panel**, not the complete
official 20-strain wet-lab panel. The source labels `AIG221` and `AIG222` are
explicitly audited aliases for official IDs `AIC221` and `AIC222`. AMP-Diffusion
MIC values are source-model pseudo-labels, not measured experiments; GRAMPA
MIC rows retain measured provenance and take precedence during condition
aggregation while both raw observations remain in the table.

Splits are deterministic exact-sequence hash splits (seed 42, nominal 80/10/10).
They prevent exact sequence leakage but are **not homology-disjoint**; the
pipeline does not require MMseqs2 or CD-HIT.

## Production command sequence

```bash
./scripts/run_deadline_pipeline.sh property
./scripts/run_deadline_pipeline.sh cache
./scripts/run_deadline_pipeline.sh ctmc-conditioned
./scripts/run_deadline_pipeline.sh manifest
./scripts/run_deadline_pipeline.sh generate
./scripts/run_deadline_pipeline.sh validate
```

The `property` stage fine-tunes from the local pinned ESM-2 150M snapshot using
`configs/deadline_property_150m.json`. `cache` builds train-split measured
conditions and fills missing fields with offline property predictions.
`ctmc-conditioned` trains/resumes the existing CTMC architecture. The manifest
stage hashes local artifacts. The generation contract requests exactly 50,000
unique valid peptides, ranks and writes exactly 100 top peptides, and emits
`generate/scores.csv`. Validation runs both repository compliance checks and
deterministic generation/hash rehearsal. No property model is run inside CTMC
epochs.

The unconditional fallback is trained separately:

```bash
./scripts/run_deadline_pipeline.sh ctmc-unconditional
```

To deploy it, select its checkpoint into the fixed `artifacts/ctmc/model.pt`
production path, ensure `artifacts/ctmc/config.json` records
`training_mode: unconditional`, set manifest CFG to zero, then rebuild and
validate the manifest. Never combine conditional fields with an unconditional
checkpoint. Emergency fallback: use the unconditional CTMC with CFG scale zero
and keep the fine-tuned property model for candidate ranking.

Training resumes from `artifacts/ctmc/training_state.pt` or
`artifacts/ctmc_unconditional/training_state.pt` when present. Preserve the
checkpoint directory when resuming; do not remove a good checkpoint to restart.

## Runtime adjustments

If the ESM property batch of 64 runs out of CUDA memory, rerun with a copied
property config using `batch_size: 32`; the training entry point reports this
specific adjustment and does not silently lower the batch. For lower-memory
CTMC GPUs, lower CTMC `batch_size` in a copied deadline config while keeping
the training split and architecture fields fixed. The 50,000-candidate scorer
uses bounded vectorized batches; if host memory is constrained, lower
`property_scoring_batch_size` in `configs/deadline_generation.json` and copy it
into the manifest build configuration.

The smoke stage is isolated under `artifacts/deadline_smoke/` and is intended
to use no more than 256 sequences and one tiny epoch per model. It must not
modify production checkpoints or manifest.
