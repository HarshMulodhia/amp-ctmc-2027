# Model card

## Intended use

This repository implements offline AMP sequence generation for the AMP Challenge 2027 submission format. The generator combines a conditional version-2 CTMC denoiser with a separately trained ESM multitask property predictor. Generation is a computational design workflow, not evidence of antimicrobial activity or safety.

## Required model artifacts

No trained model weights are checked in. A runnable release requires `artifacts/ctmc/model.pt`, `artifacts/property/model.pt`, the locally exported `artifacts/property/backbone/` and `artifacts/property/tokenizer/`, and `artifacts/property/model_metadata.json`. `artifacts/manifest.json` records their hashes and reconstruction metadata. `artifacts/manifest.example.json` has placeholder hashes and cannot be used for inference.

The property predictor has AMP classification, strain-conditioned MIC, hemolysis classification, and HC50 regression heads. Inference loads its ESM backbone and tokenizer from local files only. No Hub fallback or randomly initialized model fallback is permitted.

## Limitations

No trained checkpoint, validation metrics, calibration measurements, wet-lab results, or generation hardware measurements are available in this checkout. MIC and HC50 depend on strain, assay, units, and censoring. Generic AMP/hemolysis predictions do not establish broad-spectrum efficacy or safety. Report model identifiers, exact revisions, training-data provenance, split protocol, and held-out metrics when publishing actual trained artifacts.
