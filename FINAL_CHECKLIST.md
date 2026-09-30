# Full-Tier Final Submission Checklist

## Critical eligibility actions

- [ ] Make `https://github.com/HarshMulodhia/amp-ctmc-2027` **public** before submission. Organizer read access alone does not establish a public Full-tier repository.
- [ ] Keep `@RasmusML` and `@szymczakpau` as collaborators/readers until verification is complete.
- [ ] Verify the repository root has a GitHub-detected permissive OSI-approved license for original code.
- [ ] Keep third-party datasets and model weights under their original terms; confirm `THIRD_PARTY_NOTICES.md` distinguishes them from the root code license.
- [ ] Confirm an unauthenticated user can clone the repository and download every Git LFS object.
- [ ] Confirm frozen commit `99490e2` exists on the public default branch or a permanent public tag.
- [ ] Create a permanent release/tag, for example `full-submission-2026-09-29`, pointing to `99490e2`.
- [ ] Do not force-push or garbage-collect frozen LFS artifacts until organizer verification ends.

## Repository test

Run from a clean directory with no existing Hugging Face cache assumptions:

```bash
git clone https://github.com/HarshMulodhia/amp-ctmc-2027.git amp-final-check
cd amp-final-check
git checkout 99490e2
git lfs install
git lfs pull
uv sync
uv run generate
uv run python scripts/validate_submission.py
uv run python scripts/rehearse_submission.py
sha256sum generate/library.fasta generate/top.fasta generate/scores.csv artifacts/manifest.json
```

Expected hashes:

```text
8b072a9b5579793ac3d7a229345b28c71d5b5a7a1b81f6ff37197bd12fd63833  generate/library.fasta
2e8e01ab83c4c9a8eda2e85b1d3b68c8a37b2f904c04299b6e30f7f493c9ec84  generate/top.fasta
0bbde936a8d5e65ecd6d247ee7843742a894b1ff864fc8c9b3791288cf58222f  generate/scores.csv
0910981748c82b39dd159ba111d87a41ac7db971f7491e7df1fe5147ca5d3051  artifacts/manifest.json
```

- [ ] `library.fasta` contains exactly 50,000 unique sequences.
- [ ] `top.fasta` contains exactly 100 unique sequences.
- [ ] Every top sequence occurs in the library.
- [ ] All sequences use canonical amino acids and lengths 8–50.
- [ ] Full library has no prohibited exact reference match.
- [ ] Every top-100 peptide passes the organizer identity rule.
- [ ] Generation works without internet access after clone/LFS/dependency setup.
- [ ] No absolute paths, credentials, API keys, or private URLs are present.

## Kaggle submission

- [ ] Select **Full** participation/co-authorship eligibility.
- [ ] Paste the title, abstract, method, data, intervention, limitations, reproducibility, and eligibility statement from `SUBMISSION.md`.
- [ ] Add repository URL: `https://github.com/HarshMulodhia/amp-ctmc-2027`.
- [ ] Add frozen commit: `99490e2`.
- [ ] Add generation command: `uv run generate`.
- [ ] State seed: `42`.
- [ ] Upload the required library and top-100 files using the organizer-specified filenames and format.
- [ ] Verify the uploaded files—not merely local files—have the expected SHA-256 values if Kaggle exposes them for download.
- [ ] State explicitly that only public data were used.
- [ ] State explicitly that AMP-Diffusion panel MICs are pseudo-labels.
- [ ] State explicitly that the conditioning panel covers 11 proxy strains rather than all 20 official strains.
- [ ] State explicitly that the split is exact-sequence disjoint, not homology-disjoint.
- [ ] State explicitly that no manual candidate editing or selection was performed.

## Authorship

- [ ] List only people who genuinely contributed to the work.
- [ ] Do not list `@RasmusML` or `@szymczakpau` as authors solely because they have read access; they are organizer reviewers unless they made a qualifying contribution.
- [ ] Use a CRediT-style statement for each genuine contributor.
- [ ] Ensure all listed contributors agree to the submission and public release.

## Preserve evidence

```bash
git bundle create amp-ctmc-150m-99490e2.bundle --all
sha256sum amp-ctmc-150m-99490e2.bundle > amp-ctmc-150m-99490e2.bundle.sha256

tar -czf amp-ctmc-150m-submission.tar.gz \
  generate/library.fasta generate/top.fasta generate/scores.csv \
  artifacts/manifest.json SUBMISSION.md METHOD.md DATA.md \
  DATA_CARD.md MODEL_CARD.md TRAINING.md RUN_DEADLINE.md \
  THIRD_PARTY_NOTICES.md LICENSE
sha256sum amp-ctmc-150m-submission.tar.gz > amp-ctmc-150m-submission.tar.gz.sha256
```

- [ ] Store the archive, bundle, hashes, final logs, Kaggle confirmation, and screenshots in two locations.
- [ ] Do not commit backup archives to Git.
- [ ] Keep the repository and LFS objects available through organizer verification and paper preparation.
