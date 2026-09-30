#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
stage="${1:-}"
case "$stage" in
  bootstrap)
    uv run python scripts/bootstrap_deadline_data.py
    uv run python scripts/prepare_deadline_training_data.py
    uv run python -c 'import json; p="data/audit/deadline_source_counts.json"; print(json.dumps(json.load(open(p)), indent=2)); print("Outputs: data/raw/deadline, data/pretrained/esm2_t30_150M_UR50D (bundled), data/processed/deadline_observations.parquet, data/processed/deadline_split_manifest.csv, data/training/training.fasta, data/training/background.fasta")'
    ;;
  smoke)
    uv run python scripts/deadline_smoke.py
    ;;
  property)
    uv run python scripts/train_property_model.py --config configs/deadline_property_150m.json --data data/processed/deadline_observations.parquet --output artifacts/property
    ;;
  cache)
    mkdir -p artifacts/conditions
    uv run python scripts/build_measured_conditions.py --observations data/processed/deadline_observations.parquet --split-manifest data/processed/deadline_split_manifest.csv --property-metadata artifacts/property/model_metadata.json --fasta data/training/training.fasta --output artifacts/conditions/deadline_measured.jsonl
    uv run python scripts/cache_property_conditions.py --checkpoint artifacts/property --fasta data/training/training.fasta --measured artifacts/conditions/deadline_measured.jsonl --output artifacts/conditions/deadline_conditions.jsonl --batch-size 64
    ;;
  ctmc-conditioned)
    if [[ -f artifacts/ctmc/training_state.pt ]]; then
      uv run python scripts/train_ctmc.py --config configs/deadline_ctmc_conditioned.json --resume artifacts/ctmc/training_state.pt
    else
      uv run python scripts/train_ctmc.py --config configs/deadline_ctmc_conditioned.json
    fi
    ;;
  ctmc-unconditional)
    if [[ -f artifacts/ctmc_unconditional/training_state.pt ]]; then
      uv run python scripts/train_ctmc.py --config configs/deadline_ctmc_unconditional.json --resume artifacts/ctmc_unconditional/training_state.pt
    else
      uv run python scripts/train_ctmc.py --config configs/deadline_ctmc_unconditional.json
    fi
    ;;
  manifest)
    uv run python scripts/build_artifact_manifest.py
    ;;
  generate)
    uv run python scripts/generate_submission.py --manifest artifacts/manifest.json --seed 42
    ;;
  validate)
    uv run python scripts/validate_submission.py --manifest artifacts/manifest.json --library generate/library.fasta --top generate/top.fasta
    uv run python scripts/verify_submission.py --manifest artifacts/manifest.json
    ;;
  *)
    echo "Usage: $0 {bootstrap|smoke|property|cache|ctmc-conditioned|ctmc-unconditional|manifest|generate|validate}" >&2
    exit 2
    ;;
esac
