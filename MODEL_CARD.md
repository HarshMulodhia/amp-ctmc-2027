# Model card

## Components

The sequence generator is a lightweight time-conditioned CTMC denoiser with a version-2 canvas: up to 50 amino acids followed by a reserved EOS position and PAD positions. `ConditionVector` and the denoiser expose nine soft target values with observed masks. The property model module wraps an ESM-2 encoder with AMP classification, strain-conditioned MIC, hemolysis, and HC50 heads. These components are separate so ESM is not executed at every reverse CTMC step.

The ESM model default is `facebook/esm2_t30_150M_UR50D`; the 35M model is the documented lower-compute option. A checkpoint revision, tokenizer, data manifest, calibration state, and license notice must accompany any released fine-tuned weights. No pretrained property weights are included here. The CTMC embedding-copy helper maps canonical amino-acid IDs explicitly, leaving special-token rows independently initialized.

## Training and calibration

The property training entry point expects a Parquet table with cluster-disjoint split labels. It uses measured labels only where present and task masks otherwise. Exact and left/right censored Gaussian terms are supported; interval censoring is currently excluded from regression loss and must not be mistaken for an exact measurement. Validation uses held-out clusters. Current code does not fit calibration maps, train an ensemble, or produce challenge-panel probabilities.

The CTMC trainer retains unconditional training as the supported baseline. Conditional embeddings exist, but condition construction, classifier-free condition dropout, teacher pseudo-label provenance, ESM initialization wiring, and cached distillation loss are not yet connected to the training loop. No claims of conditional-generation benefit are made.

## Limitations and intended use

Predictions are computational hypotheses, not biological evidence. MIC depends on organism, strain, assay conditions, and censoring; generic AMP classification does not establish broad-spectrum activity. Hemolysis predictions do not establish safety. Cluster-split evaluation and experimental validation are required before selecting candidates for synthesis.
