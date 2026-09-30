To train models, build artifacts, and generate valid submission libraries according to the competition protocol, follow these workflows based on your current project state.

### **Prerequisites & Expected Local Artifact Layout**

Before generating artifacts/manifest.json, the following local files must be present:

| Target Artifact Path | Description |
| :---- | :---- |
| artifacts/ctmc/model.pt | Trained CTMC generator weights |
| artifacts/ctmc/config.json | CTMC architecture & training configuration |
| artifacts/property/model.pt | Multitask property model checkpoint |
| artifacts/property/model\_metadata.json | Property model metadata & strain definitions |
| artifacts/property/backbone/ | Local ESM model backbone directory |
| artifacts/property/tokenizer/ | Local ESM tokenizer directory |
| data/training/training.fasta | Disclosed sequence dataset |
| data/training/dataset\_manifest.json | Dataset provenance manifest |
| data/antibacterial.fasta | Local organizer reference sequences |

### **Scenario A: Full Training & Export Pipeline**

If you are starting from processed observations and need to train models from scratch:

> 1. **Train Multitask Property Model**  
>    Bash  
>    uv run python scripts/train\_property\_model.py \\  
>      \--data data/processed/observations.parquet \\  
>      \--output artifacts/property

> 2. **Build Measured Conditions Table**  
>    Bash  
>    uv run python scripts/build\_measured\_conditions.py \\  
>      \--observations data/processed/observations.parquet \\  
>      \--split-manifest data/training/split\_manifest.csv \\  
>      \--property-metadata artifacts/property/model\_metadata.json \\  
>      \--output artifacts/conditions/measured\_conditions.jsonl

> 3. **Cache Property Conditions**  
>    Bash  
>    uv run python scripts/cache\_property\_conditions.py \\  
>      \--checkpoint artifacts/property \\  
>      \--fasta data/training/training.fasta \\  
>      \--measured artifacts/conditions/measured\_conditions.jsonl \\  
>      \--output artifacts/conditions/ctmc\_conditions.jsonl \\  
>      \--batch-size 32

> 4. **Train Conditional CTMC Generator**  
>    Bash  
>    uv run python scripts/train\_ctmc.py \--config configs/train\_ctmc\_property.json

> 5. **Build Validated Artifact Manifest**  
>    Bash  
>    uv run python scripts/build\_artifact\_manifest.py

> 6. **Run Generation & Verify Output**  
>    Bash  
>    uv run generate

### **Scenario B: Existing Weights Workflow**

If trained model files are already placed in artifacts/ctmc/ and artifacts/property/:

> 1. **Generate SHA-256 Pinned Manifest**  
>    Bash  
>    uv run python scripts/build\_artifact\_manifest.py

> 2. **Generate FASTA Libraries**  
>    Bash  
>    uv run generate

>    *Outputs:* generate/library.fasta (50,000 sequences) and generate/top.fasta (100 sequences).

### **Validation & Rehearsal Verification**

To verify full pipeline compliance, output integrity, and deterministic reproducibility:

```bash  
# Validate generated FASTA pair structure and compliance  
uv run python scripts/validate_submission.py

# Rehearse end-to-end execution across fresh directories and verify SHA-256 match  
uv run python scripts/rehearse_submission.py  
```