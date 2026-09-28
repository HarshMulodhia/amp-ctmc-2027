from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from amp_ctmc_2027.config import AMPConfig, set_global_determinism
from amp_ctmc_2027.compliance import load_official_identity, validate_submission
from amp_ctmc_2027.core import CTMCDenoiser, GuidedTauLeapingSampler, ReverseGenerationConfig, SinSquaredSchedule
from amp_ctmc_2027.data.dataset import AMPCanvasEncoder
from amp_ctmc_2027.data.fasta_io import FastaRepository
from amp_ctmc_2027.discriminator import DiscriminatorEnsemble, FeatureStats, PeptideFeatureExtractor
from amp_ctmc_2027.external_scorer import VendoredScorerClient
from amp_ctmc_2027.infra import log_device, resolve_device, WandbTracker, ConstraintValidator
from amp_ctmc_2027.objectives import (
    ActivityHemolysisScorer,
    ConformityScorer,
    DiscriminatorScorer,
    ESM2PseudoPerplexityScorer,
    GreedyMMRSelector,
    MaskedPseudoLikelihoodScorer,
    MultiObjectiveScorer,
    NoveltyScorer,
    QualityScorer,
    ScoringContext,
)

logger = logging.getLogger(__name__)


class GenerationPipeline:
    """Pipeline that loads checkpoint artifacts and writes final FASTA library."""

    def __init__(self, config: AMPConfig, fasta_repo: FastaRepository) -> None:
        self.config = config
        self.repo = fasta_repo

    def _load_training_stats(self, path: Path) -> FeatureStats:
        payload = json.loads(path.read_text(encoding="utf-8"))
        stats = payload["feature_stats"]
        return FeatureStats(
            means={k: float(v) for k, v in stats["means"].items()},
            stds={k: float(v) for k, v in stats["stds"].items()},
            q05={k: float(v) for k, v in stats["q05"].items()},
            q95={k: float(v) for k, v in stats["q95"].items()},
        )

    @staticmethod
    def _merge_runtime_overrides(ckpt: AMPConfig, runtime: AMPConfig) -> AMPConfig:
        return ckpt.model_copy(
            update={
                "n_sequences": runtime.n_sequences,
                "top_k": runtime.top_k,
                "seed": runtime.seed,
                "device": runtime.device,
                "generation_batch_size": runtime.generation_batch_size,
                "checkpoint_dir": runtime.checkpoint_dir,
                "training_fasta_path": runtime.training_fasta_path,
                "antibacterial_fasta_path": runtime.antibacterial_fasta_path,
                "generate_dir": runtime.generate_dir,
                "novelty_similarity_ceiling": runtime.novelty_similarity_ceiling,
                "diversity_similarity_ceiling": runtime.diversity_similarity_ceiling,
                "top_identity_ceiling": runtime.top_identity_ceiling,
                "external_scorer_project_dir": runtime.external_scorer_project_dir,
                "external_scorer_timeout_sec": runtime.external_scorer_timeout_sec,
            }
        )

    @staticmethod
    def _length_prior(training_sequences: list[str], min_length: int, max_length: int) -> list[float]:
        train_lengths = np.array([len(seq) for seq in training_sequences], dtype=np.int64)
        counts = np.bincount(train_lengths, minlength=max_length + 1)
        prior = counts[min_length : max_length + 1].astype(np.float64)
        prior = np.maximum(prior, 1e-6)
        prior = prior / prior.sum()
        return prior.tolist()

    @staticmethod
    def _max_similarity_to_references(candidates: list[str], references: list[str]) -> np.ndarray:
        if not candidates:
            return np.array([], dtype=np.float32)
        if not references:
            return np.zeros(len(candidates), dtype=np.float32)
        out = np.zeros(len(candidates), dtype=np.float32)
        chunk_size = 256
        for start in range(0, len(candidates), chunk_size):
            chunk = candidates[start : start + chunk_size]
            sims = process.cdist(
                chunk,
                references,
                scorer=Levenshtein.normalized_similarity,
                dtype=np.float32,
            )
            out[start : start + len(chunk)] = np.max(sims, axis=1)
        return out

    @staticmethod
    def _diversity_dedup(candidates: list[str], scores: np.ndarray, similarity_ceiling: float) -> list[str]:
        if not candidates:
            return []
        ordered_idx = sorted(range(len(candidates)), key=lambda i: (-float(scores[i]), candidates[i]))
        selected: list[str] = []
        for idx in ordered_idx:
            seq = candidates[idx]
            if not selected:
                selected.append(seq)
                continue
            sims = process.cdist(
                [seq],
                selected,
                scorer=Levenshtein.normalized_similarity,
                dtype=np.float32,
            )[0]
            if float(np.max(sims)) <= similarity_ceiling:
                selected.append(seq)
        return selected

    def run(self) -> None:
        """Generate challenge-ready broad-spectrum outputs."""
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
        checkpoint_dir = self.repo.resolve(self.config.checkpoint_dir)
        model_path = checkpoint_dir / "model.pt"
        config_path = checkpoint_dir / "config.json"
        tokenizer_path = checkpoint_dir / "tokenizer.json"
        stats_path = checkpoint_dir / "training_stats.json"
        discriminator_path = checkpoint_dir / "discriminator.json"
        required = [model_path, config_path, tokenizer_path, stats_path, discriminator_path]
        missing = [path for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing checkpoint artifact(s): {missing}")

        ckpt_config = AMPConfig.from_json_file(config_path)
        config = self._merge_runtime_overrides(ckpt_config, self.config)
        set_global_determinism(config.seed)

        tokenizer_meta = json.loads(tokenizer_path.read_text(encoding="utf-8"))
        encoder = AMPCanvasEncoder.from_metadata(tokenizer_meta)

        device = resolve_device(config)
        log_device(device)
        tracker = WandbTracker(config, job_type="generate")
        tracker.log({"runtime/device": device.type})
        model = CTMCDenoiser.load(
            model_path,
            config=config,
            vocab_size=encoder.vocab_size,
            pad_idx=encoder.pad_idx,
            device=device,
        )
        model.eval()

        feature_extractor = PeptideFeatureExtractor(alphabet=config.vocab)
        feature_stats = self._load_training_stats(stats_path)
        discriminator = DiscriminatorEnsemble.load(discriminator_path, extractor=feature_extractor)
        external_scorer = VendoredScorerClient(
            project_dir=self.repo.resolve(config.external_scorer_project_dir),
            timeout_sec=config.external_scorer_timeout_sec,
        )

        training_sequences = self.repo.read_sequences(config.training_fasta_path)
        antibacterial_sequences = self.repo.read_sequences(config.antibacterial_fasta_path)
        forbidden = set(antibacterial_sequences).union(training_sequences)
        train_lengths = np.array([len(seq) for seq in training_sequences], dtype=np.int64)
        length_lo, length_hi = np.quantile(train_lengths, [0.05, 0.95])
        length_prior = self._length_prior(
            training_sequences,
            config.min_length,
            config.max_length,
        )
        logger.info(
            "Length prior band=%.0f-%.0f train_mean=%.1f prior_entries=%d",
            length_lo,
            length_hi,
            float(train_lengths.mean()),
            len(length_prior),
        )

        validator = ConstraintValidator(
            alphabet=config.vocab,
            min_len=config.min_length,
            max_len=config.max_length,
            forbidden=forbidden,
        )
        official_identity = None
        if config.require_official_compliance:
            official_identity = load_official_identity(config.official_identity_function)
        sampler = GuidedTauLeapingSampler(
            model=model,
            encoder=encoder,
            schedule=SinSquaredSchedule(),
            min_length=config.min_length,
            discriminator=discriminator,
            length_prior=length_prior,
            guidance_every=8,
            guidance_keep_frac=0.5,
            dfm_stochasticity=config.dfm_stochasticity,
        )

        seen: set[str] = set()
        candidates: list[str] = []
        if config.n_sequences >= 50_000:
            target_candidate_pool = max(config.candidate_pool_size, config.n_sequences * 2)
        else:
            target_candidate_pool = max(config.n_sequences * 3, config.top_k * 20)
        for attempt in range(config.max_generation_attempts):
            for idx, (temp, steps, share) in enumerate(
                zip(
                    config.generation_temperatures,
                    config.generation_step_counts,
                    config.generation_shares,
                )
            ):
                batch_n = max(1, int(round(config.generation_batch_size * share)))
                run_config = ReverseGenerationConfig(
                    steps=int(steps),
                    temperature=float(temp),
                    seed=config.seed + attempt * 10_000 + idx,
                    confidence_reveal_fraction=config.confidence_reveal_fraction,
                    cfg_scale=config.cfg_scale,
                )
                sampled = sampler.sample_batch(run_config, batch_n)
                sampled = [seq for seq in sampled if length_lo <= len(seq) <= length_hi]
                valid = validator.filter_valid(sampled, seen=seen)
                candidates.extend(valid)

            if len(candidates) >= target_candidate_pool:
                break
            tracker.log(
                {
                    "generation/attempt": attempt + 1,
                    "generation/candidate_count": len(candidates),
                },
                step=attempt + 1,
            )

        if len(candidates) < config.n_sequences:
            raise RuntimeError(
                f"Unable to build enough valid unique candidates: got {len(candidates)}, need {config.n_sequences}"
            )

        # Stage 1: cheap all-candidate screen.
        cheap_components = [
            DiscriminatorScorer(),
            ConformityScorer(),
            NoveltyScorer(),
            QualityScorer(),
        ]
        context = ScoringContext(
            training_sequences=training_sequences,
            antibacterial_sequences=antibacterial_sequences,
            feature_extractor=feature_extractor,
            feature_stats=feature_stats,
            discriminator=discriminator,
            model=model,
            encoder=encoder,
            external_scorer=external_scorer,
        )
        scores = MultiObjectiveScorer(cheap_components, config.score_weights, config.score_combine_mode).score(candidates, context)
        # Medium property ensemble and expensive token pseudo-likelihood are only run on shortlists.
        medium_idx = np.argsort(-scores, kind="mergesort")[: min(len(candidates), config.medium_shortlist_size)]
        medium_candidates = [candidates[i] for i in medium_idx]
        medium_components = [*cheap_components, ActivityHemolysisScorer()]
        medium_scores = MultiObjectiveScorer(medium_components, config.score_weights, config.score_combine_mode).score(medium_candidates, context)
        scores[medium_idx] = medium_scores
        final_local = np.argsort(-medium_scores, kind="mergesort")[: min(len(medium_candidates), config.final_shortlist_size)]
        final_idx = medium_idx[final_local]
        final_candidates = [candidates[i] for i in final_idx]
        final_components = [*medium_components, MaskedPseudoLikelihoodScorer(), ESM2PseudoPerplexityScorer()]
        final_scorer = MultiObjectiveScorer(final_components, config.score_weights, config.score_combine_mode)
        final_table = final_scorer.score_table(final_candidates, context)
        rank_columns = [
            (name, float(config.score_weights.get(name, 0.0)), final_table.columns.get(f"{name}.rank"))
            for name in config.score_weights
        ]
        rank_columns = [(name, weight, values) for name, weight, values in rank_columns if weight > 0 and values is not None]
        if config.score_combine_mode == "geometric":
            final_scores = np.exp(sum(weight * np.log(np.clip(values, 1e-6, 1.0)) for _, weight, values in rank_columns)).astype(np.float32)
        else:
            final_scores = sum(weight * values for _, weight, values in rank_columns).astype(np.float32)
        scores[final_idx] = final_scores

        # Stage 2: novelty filter against known AMPs + antibacterial refs
        novelty_references = sorted(set(training_sequences).union(antibacterial_sequences))
        novelty_similarity = self._max_similarity_to_references(candidates, novelty_references)
        novelty_keep = novelty_similarity <= config.novelty_similarity_ceiling
        novelty_candidates = [seq for seq, keep in zip(candidates, novelty_keep.tolist()) if keep]
        novelty_scores = scores[novelty_keep]
        if len(novelty_candidates) < config.top_k:
            raise RuntimeError("Novelty filtering left insufficient candidates for top-k selection")

        # Stage 3: diversity-aware ranked shortlist.
        mmr = GreedyMMRSelector(config.mmr_lambda_diversity)
        deduped = mmr.select(novelty_candidates, novelty_scores, min(len(novelty_candidates), config.top_k * 10))
        deduped = validator.filter_valid(deduped)
        score_by_seq = {seq: float(score) for seq, score in zip(candidates, scores.tolist())}
        deduped = sorted(deduped, key=lambda seq: (-score_by_seq[seq], seq))

        # Stage 4: top-list final identity guard against antibacterial references
        guarded: list[str] = []
        for sequence in deduped:
            if len(guarded) >= config.top_k:
                break
            max_identity = float(
                self._max_similarity_to_references([sequence], antibacterial_sequences)[0]
                if antibacterial_sequences
                else 0.0
            )
            if max_identity <= config.top_identity_ceiling:
                guarded.append(sequence)
        if len(guarded) < config.top_k:
            raise RuntimeError("Final identity guard left insufficient top-k sequences")

        # Full library begins with the ranked top list, enforcing subset membership.
        ordered_idx = sorted(range(len(candidates)), key=lambda i: (-float(scores[i]), candidates[i]))
        top = guarded[: config.top_k]
        top_set = set(top)
        library = list(top)
        library.extend(candidates[idx] for idx in ordered_idx if candidates[idx] not in top_set and len(library) < config.n_sequences)
        library = validator.filter_valid(library)
        if len(library) < config.n_sequences:
            raise RuntimeError("Could not produce enough library sequences after validation")
        library = library[: config.n_sequences]
        validator.assert_library(library, expected_size=config.n_sequences)

        validator.assert_library(top, expected_size=config.top_k)
        if not set(top).issubset(library):
            raise AssertionError("ranked top list must be a subset of the submitted library")

        selected_lengths = np.array([len(seq) for seq in library], dtype=np.float32)
        tracker.log(
            {
                "generation/final_candidate_count": len(candidates),
                "generation/final_library_count": len(library),
                "generation/final_top_count": len(top),
                "generation/score_mean": float(np.mean(scores)),
                "generation/score_std": float(np.std(scores)),
                "generation/selected_length_mean": float(selected_lengths.mean()),
                "generation/selected_length_std": float(selected_lengths.std()),
            }
        )
        output_dir = config.generate_dir
        library_path = output_dir / "library.fasta"
        top_path = output_dir / "top.fasta"
        self.repo.write_fasta(library_path, library, id_prefix="amp")
        self.repo.write_fasta(top_path, top, id_prefix="top_amp")
        final_table.write_csv(self.repo.resolve(output_dir / "scores.csv"))
        self.repo.write_json_atomic(output_dir / "score_components.json", final_table.metadata)
        if official_identity is None:
            compliance_report = {
                "pass": False, "official_validation": "not_configured",
                "reason": "The organizer identity function is absent; edit config to point at the pinned official validator.",
                "library_count": len(library), "top_count": len(top),
                "top_is_library_subset": set(top).issubset(library),
            }
        else:
            try:
                compliance_report = validate_submission(
                    library, top, antibacterial_sequences, identity_function=official_identity,
                    threshold=config.top_identity_ceiling, min_length=config.min_length,
                    max_length=config.max_length, alphabet=config.vocab,
                    expected_library_count=config.n_sequences, expected_top_count=config.top_k,
                )
            except Exception as exc:
                self.repo.write_json_atomic(output_dir / "compliance_report.json", {
                    "pass": False, "error": str(exc), "library_count": len(library), "top_count": len(top),
                    "top_is_library_subset": set(top).issubset(library),
                })
                raise
        self.repo.write_json_atomic(output_dir / "compliance_report.json", compliance_report)
        self.repo.write_json_atomic(output_dir / "run_manifest.json", {
            "seed": config.seed, "config": config.model_dump(mode="json"),
            "candidate_count": len(candidates), "library_count": len(library),
            "top_count": len(top), "official_identity_function": config.official_identity_function,
            "scoring_stages": {"medium": len(medium_candidates), "final": len(final_candidates)},
            "compliance_pass": compliance_report["pass"],
        })
        tracker.finish()
        logger.info(
            "Generated library=%d (%s) top=%d (%s) length mean=%.1f std=%.1f",
            len(library),
            self.repo.resolve(library_path),
            len(top),
            self.repo.resolve(top_path),
            float(selected_lengths.mean()),
            float(selected_lengths.std()),
        )
