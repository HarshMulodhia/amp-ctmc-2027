# AMP Challenge: Dataset and Pretrained-Model Strategy for a CTMC Generator

## Executive recommendation

Use a **two-stage data strategy**, not one indiscriminate merged corpus:

1. **CTMC generative target:** train on a rigorously cleaned, canonical, challenge-compliant AMP corpus formed from the official AMP-Diffusion set plus HydrAMP positive/MIC sets, deduplicated and similarity-clustered. The AMP-Diffusion starter kit discloses 19,670 unique sequences from DRAMP 3.0, APD3, and DBAASP; the HydrAMP release adds 20,313 AMP positives, 4,546 measured *E. coli* MIC records, classifier positives/negatives, and a large UniProt peptide corpus.[^1][^2][^3]
2. **Conditioning and ranking:** fine-tune a compact protein language model on strain/panel-specific MIC and hemolysis tasks, then use its frozen hidden representation and predicted properties to condition or guide the CTMC. Binary AMP labels alone are not sufficient: the 2026 BATTLE-AMP benchmark found that MIC-trained models outperform binary classifiers, that the strongest model depends on the pathogen, and that current models struggle with composition-preserving perturbations and activity cliffs.[^4]

The best practical backbone is **ESM-2 150M**, with parameter-efficient fine-tuning or last-layer unfreezing. It is materially easier to train than 650M–3B alternatives while still being a strong sequence encoder; ESM-2 is available in checkpoints from 8M to 15B, and published analysis found the 150M model more parameter-efficient than the older ESM-1b 650M. If compute is very tight, start with ESM-2 35M; if a 24–48 GB GPU is reliably available, benchmark ESM-2 650M with LoRA against the 150M model rather than assuming that bigger is better.[^5][^6]

Do **not** train the CTMC on generic UniProt peptides alone, and do **not** treat the challenge `antibacterial.fasta` reference list as supervised training data. Generic UniProt is useful only for continued masked-language pretraining or as carefully matched negatives. The official baseline repositories explicitly distinguish the challenge reference set—used for overlap and novelty checks—from their genuine training datasets.[^2][^1]

## What the rules allow

The current official challenge page welcomes language models, diffusion models, VAEs, GFlowNets, reinforcement learning, Bayesian optimization, evolutionary methods, and hybrids. It asks teams to disclose training data, external databases, manual intervention, and computational filters; full-reproducibility entries must publish weights and inference code, use a permissive OSI-approved repository license, and release any proprietary or non-public training data under a permissive license. Therefore, **public external datasets and pretrained models are allowed in practice**, provided their provenance, terms, weights, and use are disclosed and the submitted repository satisfies the reproducibility and licensing requirements.[^7]

Every submitted peptide must contain only the 20 canonical amino acids, be linear and 8–50 residues long, have free unmodified termini, and be unique. Each top-100 peptide must have no more than 80% sequence identity to the competition reference database. The model should generate well beyond 50,000 candidates, then apply deterministic compliance, novelty, diversity, potency, and safety filters before selecting the required 50,000 library and ranked top 100.[^7]

A licensing caveat remains: “publicly accessible” does not automatically mean “permissively redistributable.” The challenge itself tells teams to verify each database’s terms, so raw upstream records should be accompanied by provenance and either redistribution permission or a deterministic downloader/processing script. The official starter kits are the lowest-friction source because they already disclose the records used by the baselines, but their upstream database terms should still be documented.[^7]

## Recommended datasets

### Primary training stack

| Priority | Dataset | Intended role | Why it belongs | Main caution |
|---|---|---|---|---|
| 1 | AMP-Diffusion training corpus | CTMC AMP distribution | Official starter-kit corpus; 19,670 unique canonical sequences, length 1–50, compiled from DRAMP 3.0, APD3, and DBAASP[^1][^8] | Apply challenge length 8–50; remove reference-set overlap and cross-source duplicates |
| 1 | HydrAMP `unlabelled_positive` and AMP-classifier positives | Enlarge positive AMP corpus | HydrAMP publicly releases its data and training scripts; its paper describes AMP, low-MIC, and UniProt components[^9][^10] | HydrAMP was designed for peptides under 25 aa; deduplicate heavily against AMP-Diffusion |
| 1 | GRAMPA MIC data | Property supervision and conditional CTMC labels | 6,760 unique sequences and 51,345 MIC measurements with species/strain and source metadata[^11] | Assay conditions and reporting differ; aggregate replicates robustly and retain censoring |
| 1 | HydrAMP `mic_data.csv` | *E. coli* MIC supervision | 4,546 measured records used in a released challenge baseline[^2] | Likely overlaps GRAMPA; identify by normalized sequence plus organism/strain |
| 1 | HemoPI2 or HAPPENN | Hemolysis/HC50 supervision | HemoPI2 provides 1,926 experimentally validated hemolytic peptides with concentrations; HAPPENN provides 3,738 experimentally validated sequences[^12][^13] | Do not mix incompatible thresholds without retaining quantitative assay metadata |
| 2 | HydrAMP UniProt short peptides | Continued MLM/domain adaptation and matched negatives | About 225k short biological sequences in the released HydrAMP data; useful for learning peptide syntax and constructing non-AMP controls[^2] | Not an AMP target distribution; assumed negatives can contain undiscovered AMPs |
| 2 | BATTLE-AMP datasets and holdout definitions | External evaluation and oracle selection | Reproducible framework spanning classification, MIC, species, strain, syntax perturbation, and activity-cliff tests[^14][^4] | Keep true holdouts out of training; benchmark contamination would invalidate model selection |
| 3 | HemoPI-1 random negatives | Sanity check only | Widely used baseline dataset[^15][^16] | Random Swiss-Prot negatives create an easy shortcut and are inferior to experimentally weak/non-hemolytic negatives |

### Cleaning protocol

Create a single canonical record table with at least: `sequence`, `source_db`, `source_id`, `activity_type`, `organism`, `strain`, `mic_value`, `mic_unit`, `mic_relation`, `hemolysis_value`, `hemolysis_unit`, `assay_conditions`, `modifications`, and `citation`. Convert MIC to micromolar only when molecular weight and terminal chemistry are known; preserve inequality-censored measurements such as `>64` rather than converting them into exact numbers.

Apply these filters before model training:

- Uppercase and retain only `ACDEFGHIKLMNPQRSTVWY`.
- Keep 8–50 residues for the final CTMC corpus; retain an out-of-range audit file rather than silently deleting records.
- Exclude modified, cyclic, stapled, lipidated, glycosylated, dendrimeric, or non-free-terminus sequences from the competition-facing corpus.
- Exact-deduplicate sequences across all databases while preserving all labels and provenance.
- Detect contradictory labels; for MIC, aggregate repeated measurements within the same organism/strain using a robust statistic in log2 space and retain dispersion.
- Cluster by sequence identity before splitting. Use cluster-level train/validation/test splits, with a stringent 40–60% identity threshold for honest generalization testing and an additional 80% audit matching the competition novelty boundary.
- Remove any competition reference sequence from training if strict novelty is the objective; at minimum, tag all overlaps so Phase-1 novelty cannot be mistaken for learned generalization.
- Preserve separate pathogen heads or labels. BATTLE-AMP shows that predictor choice is pathogen-dependent, so collapsing all MICs into one unconditioned scalar discards crucial biology.[^4]

## Pretrained model shortlist

### Representation backbones

| Rank | Model | Best use | Strengths | Weaknesses |
|---|---|---|---|---|
| 1 | **ESM-2 150M** | Main fine-tuned encoder and CTMC backbone initializer | Strong parameter efficiency; canonical ESM stack; manageable fine-tuning; per-residue representations[^5][^6] | Pretrained on proteins rather than specifically short peptides |
| 2 | **ESM-2 650M** | High-compute ablation or LoRA model | 33 layers, 1,280-dimensional hidden states; strong general protein representations[^17][^5] | Higher memory and slower candidate scoring; full fine-tuning is unnecessary |
| 3 | **ESM-2 35M** | Fast baseline, ablations, 50k-scale generation | 480-dimensional embeddings and low deployment cost[^5] | Lower representation capacity than 150M/650M |
| 4 | **ProtT5-XL-UniRef50** | Frozen-embedding ensemble | Pretrained self-supervised on roughly 45 million UniRef50 sequences and intended for feature extraction/fine-tuning[^18] | Approximately 3B parameters; expensive and unlikely to justify its cost for an 8–50 aa task |
| 5 | **ESM-C 300M** | Modern alternative benchmark | Open weights under the Cambrian Open License; fine-tuning and embedding use are explicitly supported[^19][^20] | Custom license must be checked against the submission’s redistribution plan; less challenge-specific precedent |

PeptideBERT is valuable primarily as a **hemolysis/solubility auxiliary model**, not the main generative backbone. It fine-tunes ProtBERT for hemolysis, solubility, and non-fouling prediction, and publishes code, models, and data. Because its backbone is older and its tasks are auxiliary, use its labels/checkpoint as a baseline or ensemble member, while keeping the CTMC representation on ESM-2 for a simpler stack.[^21][^22][^23]

### Activity and safety oracles

| Model | Recommended role | Evidence and scope | Integration advice |
|---|---|---|---|
| **MBC-Attention** | Strong *E. coli*/broad and Gram-negative ranker | Predicts *E. coli* MIC; its original work reports PCC 0.775 and RMSE 0.533 log micromolar, with production models released[^24][^25] | Use as one oracle, not as ground truth; it was among the strongest broad/Gram-negative models in the 2026 BATTLE-AMP framework |
| **SenseXAMP** | ESM-based classifier and *E. coli*/*S. aureus* MIC ensemble member | Combines fine-tuned ESM-1b with physicochemical descriptors and releases code, data, and checkpoints[^26][^27] | Useful teacher/ensemble; porting its head to ESM-2 is preferable to inheriting ESM-1b indefinitely |
| **AMPredictor** | MIC and structure-aware ensemble member | Uses ESM-1b embeddings, predicted contact maps, graph convolution, and chemical fingerprints; code and data are public[^28][^29][^30] | Expensive at 50k+ scale; cache embeddings and use only for a reduced shortlist |
| **APEX ensemble** | Strain-specific ESKAPE and panel ranking | Predicts MIC across pathogen strains; published work reports superiority to classical baselines for most pathogens, and the official AMP-Diffusion starter kit bundles an eight-model APEX scorer[^1][^31][^32][^33] | Particularly relevant to the challenge’s MDR and broad-spectrum panels; verify upstream model/data licensing separately |
| **HydrAMP AMP/MIC heads** | Reproducible baseline and teacher | Released checkpoint jointly scores AMP probability and low-MIC probability; complete training data and retraining notebooks are available[^10][^2][^3] | Limited to 25 residues; use for a short-peptide ensemble, not as the sole selector |
| **HemoPI2 + PeptideBERT hemolysis** | Safety and HC50 ranking | HemoPI2 supplies quantitative hemolysis data and downloadable software; PeptideBERT publishes a hemolysis checkpoint[^12][^34][^22] | Prefer quantitative HC50 regression with censoring, backed by classifier agreement; calibrate on held-out clusters |

No single pretrained predictor is reliably “best” for all five categories. BATTLE-AMP’s central result is exactly that MIC-trained models beat generic AMP classifiers and that the best model changes with the target pathogen. Build a calibrated ensemble with category-specific weights and report disagreement; a candidate should not reach the top 100 solely because one oracle gives an extreme prediction.[^4]

## CTMC integration design

### Recommended architecture

Let a peptide be represented as a padded sequence of length 50 over a vocabulary of 22 tokens: 20 amino acids, `MASK`, and `PAD/EOS`. Define a time-inhomogeneous CTMC corruption path from clean sequence `z` to a masked/noisy sequence `x_t`. A neural denoiser predicts a categorical distribution over the clean residue at every non-padding position:

$$
p_\theta(z_i \mid x_t,t,c),
$$

where `c` contains desired properties such as Gram class, pathogen/panel, potency bin, and low-hemolysis target. The associated CTMC rate matrix is constructed from these posterior probabilities, and the model is trained with token cross-entropy/conditional generator-matching loss.

Initialize the denoiser from **ESM-2 150M**, add a continuous-time embedding, and inject the condition through feature-wise modulation or cross-attention. Use a length token or explicitly model `EOS`; masking padded positions without a length model will reproduce the training length histogram poorly and can bias generation toward short peptides.

### Three training stages

1. **Domain adaptation:** optional masked-language-model continuation on the cleaned mixture of AMP sequences plus short UniProt peptides. Sample datasets so that the 225k UniProt component does not swamp the AMP corpus.
2. **Property fine-tuning:** train category-specific MIC heads and a hemolysis/HC50 head on cluster-disjoint splits. Start with the encoder frozen, then unfreeze the last 4–8 layers or use LoRA. Use censored regression or interval loss for `>`/`<` measurements rather than pretending assay limits are exact.
3. **Conditional CTMC:** train on the cleaned AMP corpus using conditions derived from measured labels where available. For unlabeled AMP sequences, use classifier-free conditioning or mark labels as unknown; do not indiscriminately attach teacher pseudo-labels as facts.

A more conservative first milestone is an **unconditional ESM-2-initialized CTMC plus post-generation ensemble ranking**. Conditional generation should be added only after this baseline passes validity, diversity, novelty, and recovery tests. This isolates whether gains come from a better CTMC or from the ranking oracle.

### Guidance and ranking

Generate at least 5–10 times the required library size, apply hard compliance filters, remove duplicates and known-reference overlaps, then score survivors. Use classifier-free guidance during CTMC sampling only after confirming that it does not collapse diversity. A practical category score is a calibrated combination of:

- predicted log2 MIC over the relevant challenge strains;
- predicted probability of MIC at or below 16 micromolar;
- worst-case or upper-quantile MIC across the category panel, not only the mean;
- predicted HC50 and safety window;
- synthesis-risk penalties, aggregation/hydrophobic-run penalties, and charge/hydrophobicity constraints;
- novelty to the official reference set and diversity relative to already selected candidates;
- ensemble uncertainty/disagreement penalty.

For the **Optimal Selectivity** category, directly optimize a conservative safety-window estimate rather than AMP probability. For **broad-spectrum** and **MDR**, use a minimax or lower-tail objective so one excellent strain prediction cannot hide failures on the rest of the panel.

## Validation plan

Use four non-overlapping evaluation layers:

- **Generation quality:** valid fraction, unique fraction, length distribution, amino-acid and motif statistics, novelty, nearest-neighbor identity, and internal diversity.
- **Recovery tests:** corrupt held-out real AMPs at several times and evaluate token recovery and sequence likelihood.
- **Biological prediction:** MIC rank correlation/error, active-at-16-micromolar AUROC/AUPRC, HC50 error, and calibration on cluster-disjoint holdouts.
- **Robustness:** composition-preserving shuffles, single-residue activity cliffs, label/source holdouts, and performance stratified by length and similarity. BATTLE-AMP specifically warns that many predictors fail on composition-preserving perturbations and unresolved activity cliffs.[^4]

The decisive ablations should be small and interpretable:

| Question | Controlled comparison |
|---|---|
| Does protein pretraining help? | Randomly initialized CTMC vs ESM-2 35M vs ESM-2 150M |
| Does peptide domain adaptation help? | ESM-2 base vs continued MLM on the same cleaned corpus |
| Does conditioning outperform filtering? | Unconditional CTMC + oracle ranking vs conditional CTMC + identical ranking |
| Does a single oracle overfit? | Best single predictor vs calibrated multi-oracle ensemble |
| Does the data merge help? | AMP-Diffusion-only vs merged, exact-deduplicated vs merged, cluster-deduplicated |
| Does guidance collapse diversity? | Guidance scales with matched generation budget and identical compliance filters |

## Concrete build order

1. Vendor or download the two official starter-kit datasets and create a provenance manifest.
2. Build the canonical challenge-compliant sequence table and an immutable competition-reference exclusion/audit set.
3. Cluster sequences before splitting; produce exact-overlap and near-overlap reports across every source.
4. Train a simple amino-acid-frequency or small Transformer CTMC as an end-to-end compliance baseline.
5. Replace the denoiser with ESM-2 35M, then 150M; verify gains at fixed sampler steps and compute.
6. Fine-tune MIC and hemolysis heads, comparing against untouched MBC-Attention, SenseXAMP, APEX, AMPredictor, HydrAMP, and HemoPI2 baselines.
7. Add property conditioning and classifier-free guidance only after the unconditional model is stable.
8. Generate a large candidate pool, run deterministic hard filters, ensemble ranking, uncertainty filtering, and greedy diversity selection.
9. Run the official validator twice with the fixed seed and verify byte-identical outputs.
10. Publish exact data hashes, preprocessing scripts, split assignments, model weights, environment lockfile, generation config, random seed, and ranking rationale.

## Final choice

If one configuration must be selected now, use:

- **Generator data:** exact- and cluster-deduplicated union of the AMP-Diffusion 19,670-sequence corpus and HydrAMP AMP-positive/MIC sequences, restricted to canonical linear 8–50 aa peptides.
- **Property data:** GRAMPA plus HydrAMP MIC records with organism/strain retained; HemoPI2 plus HAPPENN for quantitative or thresholded hemolysis.
- **Backbone:** ESM-2 150M with LoRA or final-layer fine-tuning.
- **Generator:** time-conditioned masked-token CTMC over 20 amino acids plus MASK and EOS/PAD.
- **Selection ensemble:** APEX for strain/panel MIC, MBC-Attention for *E. coli*/Gram-negative strength, SenseXAMP and AMPredictor for representation-diverse agreement, HydrAMP as a short-peptide baseline, and HemoPI2/PeptideBERT for safety.
- **Risk controls:** cluster-disjoint evaluation, reference-set exclusion, uncertainty-aware ensemble scoring, hard novelty/diversity constraints, and no unverified teacher pseudo-labels.

This stack best matches the challenge’s actual wet-lab endpoints while keeping the course contribution clear: a pretrained protein representation adapted into a conditional discrete CTMC, with controlled experiments separating representation transfer, generative modeling, property guidance, and post-generation selection.

---

## References

1. [szczurek-lab/ampdiffusion-starter-kit: AMP-Diffusion ...](https://github.com/szczurek-lab/ampdiffusion-starter-kit) - A self-contained, reproducible generator that produces the AMP-Diffusion baseline library for the AM...

2. [HydrAMP Starter Kit: AMP Challenge 2027 Baseline](https://github.com/szczurek-lab/hydramp-starter-kit) - A self-contained, reproducible generator that produces the HydrAMP baseline library for the AMP Chal...

3. [Data for HydrAMP - a deep generative model for antimicrobial peptide discovery](https://zenodo.org/records/7420278) - data- training data for peptides < 25 AA (16.8 MB) models - checkpoints of HydrAMP, PepCVAE, and Bas...

4. [BATTLE-AMP: Benchmarking Antimicrobial Peptide ...](https://www.biorxiv.org/content/10.64898/2026.06.19.733349v1) - by P Szymczak · 2026 — BATTLE-AMP is released as an open Snakemake framework szczurek-lab/battleamp-...

5. [facebookresearch/esm: Evolutionary Scale Modeling ...](https://github.com/facebookresearch/ESM) - This repository contains code and pre-trained weights for Transformer protein language models … our ...

6. [Language models of protein sequences at the scale of evolution enable accurate structure prediction](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1.full.pdf)

7. [AMP Challenge](https://www.kaggle.com/competitions/amp-challenge/overview) - AMP Challenge addresses this through a common benchmark design, a shared experimental pipeline, and ...

8. [Generative latent diffusion language modeling yields anti- ...](https://www.biorxiv.org/content/10.1101/2025.01.31.636003v1.full.pdf) - by MDT Torres · 2025 · Cited by 44 — This study highlights the potential of AMP-Diffusion as a robus...

9. [a deep generative model for antimicrobial peptide discovery](https://www.biorxiv.org/content/10.1101/2022.01.27.478054v1.full.pdf) - The training data for HydrAMP consists of a curated data set of peptide sequences, pre-trained prior...

10. [Discovering highly potent antimicrobial peptides with deep generative model HydrAMP](https://www.biorxiv.org/content/10.1101/2022.01.27.478054v2.full.pdf)

11. [Deep learning regression model for antimicrobial peptide design](https://www.biorxiv.org/content/10.1101/692681v1.full.pdf)

12. [Prediction of hemolytic peptides and their ... - PMC - NIH](https://pmc.ncbi.nlm.nih.gov/articles/PMC11794569/) - Peptide-based drugs often fail in clinical trials due to their toxicity or hemolytic activity agains...

13. [HAPPENN is a novel tool for hemolytic activity prediction ...](https://pmc.ncbi.nlm.nih.gov/articles/PMC7331684/) - The growing prevalence of resistance to antibiotics motivates the search for new antibacterial agent...

14. [szczurek-lab/battleamp-snakemake](https://github.com/szczurek-lab/battleamp-snakemake) - A Snakemake pipeline for benchmarking antimicrobial peptide (AMP) prediction models against a curate...

15. [HemoPI: Hemolytic Peptide Identification Server](https://webs.iiitd.edu.in/raghava/hemopi/help.php) - designing and prediction of hemolytic peptides

16. [Machine learning-guided discovery and design of non-hemolytic peptides](https://www.nature.com/articles/s41598-020-73644-6) - Reducing hurdles to clinical trials without compromising the therapeutic promises of peptide candida

17. [ESM 2](https://docs.nvidia.com/bionemo-framework/latest/models/ESM-2/)

18. [virtual-human-chc/prot_t5_xl_uniref50 - Hugging Face](https://huggingface.co/virtual-human-chc/prot_t5_xl_uniref50) - We’re on a journey to advance and democratize artificial intelligence through open source and open s...

19. [esm/LICENSE.md at main · evolutionaryscale/esm](https://github.com/evolutionaryscale/esm/blob/main/LICENSE.md) - Contribute to evolutionaryscale/esm development by creating an account on GitHub.

20. [Cambrian Open License Agreement](https://www.evolutionaryscale.ai/policies/cambrian-open-license-agreement) - “ESM-3 Model Weights” means the trained model weights for EvolutionaryScale's ESM-3 Open Model made ...

21. [PeptideBERT: A Language Model based on Transformers for Peptide Property Prediction](https://ar5iv.labs.arxiv.org/html/2309.03099) - Recent advances in Language Models have enabled the protein modeling community with a powerful tool ...

22. [PeptideBERT: A Language Model Based on Transformers for ... - PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10683064/) - Recent advances in language models have enabled the protein modeling community with a powerful tool ...

23. [GitHub - ChakradharG/PeptideBERT: Transformer Based Language Model for Peptide Property Prediction](https://github.com/ChakradharG/PeptideBERT) - Transformer Based Language Model for Peptide Property Prediction - ChakradharG/PeptideBERT

24. [A deep learning method for predicting the minimum ...](https://journals.asm.org/doi/10.1128/msystems.00345-23) - by J Yan · 2023 · Cited by 52 — MBC-Attention, a regressive model designed to predict the MIC value ...

25. [A deep learning method for predicting the minimum ...](https://pubmed.ncbi.nlm.nih.gov/37431995/) - by J Yan · 2023 · Cited by 52 — A deep learning method for predicting the minimum inhibitory concent...

26. [cross-modal framework for general identification of AMPs | Briefings ...](https://academic.oup.com/bib/article/24/6/bbad336/7287428?guestAccessKey=) - Abstract. Antimicrobial peptides (AMPs) are promising candidates for the development of new antibiot...

27. [GitHub - William-Zhanng/SenseXAMP](https://github.com/William-Zhanng/SenseXAMP) - Contribute to William-Zhanng/SenseXAMP development by creating an account on GitHub.

28. [ruihan-dong/AMPredictor](https://github.com/ruihan-dong/AMPredictor) - The first step is to transform a peptide sequence into ESM embedding and obtain its contact map. The...

29. [Exploring the repository of de novo designed bifunctional antimicrobial peptides through deep learning](https://elifesciences.org/reviewed-preprints/97330v1)

30. [Dong, Liu, Liu et al. eLife 2024;13:RP97330. DOI: https://doi.org/10.7554/eLife.97330](https://elifesciences.org/articles/97330.pdf)

31. [Computational exploration of global venoms for antimicrobial discovery with Venomics artificial intelligence](https://www.nature.com/articles/s41467-025-60051-6) - Researchers used artificial intelligence to mine global venom proteomes and discovered novel peptide...

32. [Deep learning reveals antibiotics in the archaeal proteome](https://www.nature.com/articles/s41564-025-02061-0) - Use of artificial intelligence to mine proteomes of archaea led to the discovery of archaeasins, ant...

33. [Deep-learning-enabled antibiotic discovery through ...](https://www.nature.com/articles/s41551-024-01201-x) - by F Wan · 2024 · Cited by 219 — The radius reflects the R2 value for each of the models. APEX varia...

34. [HemoPI2: Prediction of hemolytic potential of peptides](https://webs.iiitd.edu.in/raghava/hemopi2/) - HemoPI2 is an advanced computational tool designed for predicting the hemolytic activity of peptides...
