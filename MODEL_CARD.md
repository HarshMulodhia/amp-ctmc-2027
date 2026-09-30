# Model card

## Intended use

This repository implements offline AMP sequence generation for the AMP Challenge 2027 submission format. The generator combines a conditional version-2 CTMC denoiser with a separately trained ESM multitask property predictor. Generation is a computational design workflow, not evidence of antimicrobial activity or safety.

## Required model artifacts

This Full-tier checkout includes the trained production artifacts:

- `artifacts/ctmc/model.pt` and `artifacts/ctmc/config.json`
- `artifacts/property/model.pt`, `artifacts/property/model_metadata.json`
- locally exported `artifacts/property/backbone/` and `artifacts/property/tokenizer/`
- pinned pretrained snapshot `data/pretrained/esm2_t30_150M_UR50D/` (`facebook/esm2_t30_150M_UR50D` revision `a695f6045e2e32885fa60af20c13cb35398ce30c`)
- `artifacts/manifest.json` with SHA-256 pins for reconstruction

Large weight files are stored with Git LFS. After clone, run `git lfs install` and `git lfs pull`.

The property predictor has AMP classification, strain-conditioned MIC, hemolysis classification, and HC50 regression heads. Inference loads its ESM backbone and tokenizer from local files only. No Hub fallback or randomly initialized model fallback is permitted.

## Limitations

MIC and HC50 depend on strain, assay, units, and censoring. Generic AMP/hemolysis predictions do not establish broad-spectrum efficacy or safety. Conditioning uses an 11-strain public proxy panel rather than the full official 20-strain wet-lab panel. Report model identifiers, exact revisions, training-data provenance, split protocol, and held-out metrics when discussing biological claims.
