# Data card

## Intended data flow

The data preparation code consumes disclosed CSV exports and creates a row-preserving Parquet observation table. It canonicalizes sequences, records SHA-256 hashes, excludes invalid/modified/reference-overlap observations, keeps assay units and censor indicators, and reports contradictory AMP labels. Repeated observations are retained instead of collapsed across strains or assay limits.

The intended source roles are the AMP-Diffusion starter corpus for generation, HydrAMP positives/background and MIC rows, GRAMPA for organism/strain MIC labels, and HemoPI2 for hemolysis/HC50. The organizer reference FASTA is reserved for compliance checks and must not be used as supervised training data. Generic UniProt peptides are only a potential matched negative or continued-pretraining source.

## Provenance status

`configs/data_sources.yaml` records source URLs, roles, expected filenames, version fields, and terms links. The precise release version and SHA-256 fields are deliberately unfilled because no source artifacts or verified checksums are included in this checkout. `scripts/download_data.py` refuses to download an entry until its SHA-256 is pinned. Do not treat this catalog as a completed data disclosure.

## Cleaning and labels

Sequences are uppercased and whitespace-stripped; only the 20 canonical residues and lengths 8–50 are retained. Explicit modification and non-linear metadata cause rejection. Missing modification metadata remain unknown. Mass concentration is never converted to molar concentration here. MIC and HC50 censor fields are retained; observation aggregation is not performed by the preparation script.

Homology grouping is available through the MMseqs2 adapter. The tool version and command must be recorded by the run that produces cluster IDs; complete clusters must remain in one split. The current CSV preparation entry point does not invoke clustering or assign split labels automatically. Property training therefore requires an externally prepared Parquet table with cluster-aware `train`, `val`, and `test` assignments.

## Counts and limitations

No source data are checked into this repository, so counts, overlap rates, license conclusions, filter statistics, and split sizes are unavailable. They must be included in a completed `dataset_manifest.json` before training claims are made. See `configs/data_sources.yaml` for outstanding pins.
