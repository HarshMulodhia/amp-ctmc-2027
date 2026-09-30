# AMP Challenge 2027 submission

## Local inference artifacts

Place the trained files at these repository paths:

- `artifacts/ctmc/model.pt` and `artifacts/ctmc/config.json`: version-2 CTMC checkpoint and its training configuration.
- `artifacts/property/model.pt`, `artifacts/property/model_metadata.json`, `artifacts/property/backbone/`, and `artifacts/property/tokenizer/`: trained multitask property model, local ESM backbone, tokenizer, and strain metadata.
- `data/training/training.fasta` and `data/training/dataset_manifest.json`: disclosed training sequences and their provenance manifest.
- `data/antibacterial.fasta`: organizer reference sequences.

Generate `artifacts/manifest.json` after placing all files:

```bash
uv run python scripts/build_artifact_manifest.py
```

The manifest pins every artifact by SHA-256, records the CTMC vocabulary/token IDs and architecture, identifies the training-data manifest, records the official identity source commit, and fixes the seed and output paths. Generation fails before model loading if a required artifact is missing or its hash differs.

## Generate

From the repository root, with no required arguments:

```bash
uv run generate
```

The deterministic outputs are `generate/library.fasta` (50,000 sequences) and `generate/top.fasta` (100 sequences). Generation loads only local checkpoint/tokenizer files and sets Hugging Face offline mode. It does not train or fetch datasets or model weights.

## Validate and rehearse

```bash
uv run python scripts/validate_submission.py
uv run python scripts/rehearse_submission.py
```

The rehearsal runs the normal `generate` entry point twice in fresh output directories, validates both FASTA pairs, and compares SHA-256 hashes.

## Hardware estimate

CUDA is used when available; otherwise inference uses CPU. An RTX 4090-class GPU with 24 GiB VRAM and 32 GiB host memory are planning estimates only; peak memory and generation time have **not been measured**. CPU generation time has **not been measured**.

## Disclosures

Training-data sources and provenance belong in [DATA.md](DATA.md) and the referenced `data/training/dataset_manifest.json`. External model details and limitations belong in [MODEL_CARD.md](MODEL_CARD.md). The implemented method and official-validator provenance are documented in [METHOD.md](METHOD.md) and [OFFICIAL_VALIDATOR_SOURCE.md](OFFICIAL_VALIDATOR_SOURCE.md).

Trained CTMC and ESM-2 150M property checkpoints are included under `artifacts/` and `data/pretrained/esm2_t30_150M_UR50D/` via Git LFS. After a clean clone:

```bash
git lfs install
git lfs pull
uv sync
uv run generate
uv run python scripts/validate_submission.py
```

`artifacts/manifest.example.json` still contains placeholder hashes and is not a runnable model manifest; use `artifacts/manifest.json`.

## Training exports

Prepare the property-training Parquet table with train/validation/test cluster assignments, then set the exact challenge panel IDs and group membership in `configs/train_property.json`. Empty panel lists are deliberate placeholders and property export rejects them. The property run exports its checkpoint and local model files to `artifacts/property/`, and writes the training FASTA and provenance manifest under `data/training/`.

```bash
uv run python scripts/train_property_model.py --data data/processed/observations.parquet --output artifacts/property
uv run python scripts/build_measured_conditions.py --observations data/processed/observations.parquet --split-manifest data/training/split_manifest.csv --property-metadata artifacts/property/model_metadata.json --output artifacts/conditions/measured_conditions.jsonl
uv run python scripts/train_ctmc.py --config configs/train_ctmc.json
uv run python scripts/build_artifact_manifest.py
uv run generate
```

The default CTMC command uses `training_mode: unconditional`. Property conditioning uses a separate local prediction cache; property inference is never part of a CTMC epoch:

```bash
uv run python scripts/cache_property_conditions.py --checkpoint artifacts/property --fasta data/training/training.fasta --output artifacts/conditions/ctmc_conditions.jsonl --batch-size 32
uv run python scripts/train_ctmc.py --config configs/train_ctmc_property.json
```

The measured-condition builder uses train-split rows only, aggregates replicates by median, retains assay/censoring/uncertainty audits, and masks conditions that fail panel or group coverage thresholds. Add `--measured artifacts/conditions/measured_conditions.jsonl` to the cache command to give those measured fields precedence; teacher predictions fill only missing fields. Each wide row has `sequence`, explicit `condition_names` in `CONDITION_NAMES` order, plus `values`, `observed`, `provenance`, and `uncertainty` maps. Provenance values are `measured`, `teacher`, `derived`, or `missing`. Activity threshold, temperature, strain groups, and conservative broad aggregation come from property-model metadata. CTMC normalization is fitted only on observed train-split cache entries, stored in the checkpoint, and reused for validation and generation. The cache is create-once and its adjacent `.manifest.json` records schema, semantics, model hashes, and table SHA-256. Property-conditioned training verifies complete sequence coverage and at least one observation. The CTMC trainer exports to the established `artifacts/ctmc/` path. The manifest builder requires the local organizer reference at `data/antibacterial.fasta`; it never downloads model or data files.
