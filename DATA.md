# Data disclosure

## Required local inputs

Generation requires `data/training/training.fasta` and `data/training/dataset_manifest.json`. The FASTA supplies the length prior and exact-collision exclusion set. The manifest records sources, licenses, filtering, split provenance, and SHA-256 values. Generation also requires `data/antibacterial.fasta`, which is reserved for exact-overlap and official top-list identity checks; it is not a supervised training source.

## Source status

`configs/data_sources.yaml` lists intended source datasets and source roles. Source releases, licenses, redistribution permission, and checksums remain unverified in this checkout. Do not represent the catalog as a completed data disclosure. Before training or release, complete the manifest from the actual acquired files and include counts, cleaning/rejection summaries, overlap analysis, and cluster split details.

## Dataset rules

Keep AMP sequence data, organism/strain MIC measurements, and hemolysis/HC50 observations as separate records with provenance. Preserve units and censor bounds. Do not turn `data/antibacterial.fasta` into ordinary supervised training data. Document every external dataset and any non-public material in `data/training/dataset_manifest.json`.

No training dataset is included here.
