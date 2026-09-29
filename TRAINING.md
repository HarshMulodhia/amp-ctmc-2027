# Training

Install the data and model extras with `uv sync --extra data --extra training`. Prepare a source export only after recording its license, version, and checksum:

```bash
uv run python scripts/prepare_data.py disclosed_export.csv --output data/processed/observations.parquet
uv run train --config configs/train_ctmc.json
```

The raw FASTA CTMC trainer still consumes the configured training/background FASTA files. `--smoke-test` runs a small non-production overfit subset. Resume from an optimizer checkpoint with `uv run train --config configs/train_ctmc.json --resume checkpoint/training_state.pt`.

For property training, provide a Parquet table that already contains cluster-level `split` assignments, then run `uv run python scripts/train_property_model.py --data data/processed/property_training.parquet`. MMseqs2 is required to create homology clusters; pin its binary version and preserve the exact command in the data manifest. The CSV preparation script does not create those assignments automatically.

Training requires local disclosed data. Generation must use local checkpoints and must not download models. Report cluster-held-out metrics separately by AMP, MIC strain, hemolysis, and HC50 task. The included pipeline does not yet implement complete calibration, ensemble training, or teacher-cache distillation.
