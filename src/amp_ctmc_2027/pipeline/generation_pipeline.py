from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from amp_ctmc_2027.config import AMPConfig, set_global_determinism
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
    MultiObjectiveScorer,
    NoveltyScorer,
    QualityScorer,
    RealismScorer,
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
        forbidden = set(antibacterial_sequences)
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

        # Stage 1: multi-objective scoring
        components = [
            DiscriminatorScorer(),
            ConformityScorer(),
            NoveltyScorer(),
            QualityScorer(),
            RealismScorer(),
            ActivityHemolysisScorer(),
            ESM2PseudoPerplexityScorer(),
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
        scorer = MultiObjectiveScorer(
            components=components,
            weights=config.score_weights,
            combine_mode=config.score_combine_mode,
        )
        scores = scorer.score(candidates, context)

        # Stage 2: novelty filter against known AMPs + antibacterial refs
        novelty_references = sorted(set(training_sequences).union(antibacterial_sequences))
        novelty_similarity = self._max_similarity_to_references(candidates, novelty_references)
        novelty_keep = novelty_similarity <= config.novelty_similarity_ceiling
        novelty_candidates = [seq for seq, keep in zip(candidates, novelty_keep.tolist()) if keep]
        novelty_scores = scores[novelty_keep]
        if len(novelty_candidates) < config.top_k:
            raise RuntimeError("Novelty filtering left insufficient candidates for top-k selection")

        # Stage 3: diversity dedup with score-priority keep rule
        deduped = self._diversity_dedup(
            novelty_candidates,
            novelty_scores,
            similarity_ceiling=config.diversity_similarity_ceiling,
        )
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

        # Full library output at challenge scale
        ordered_idx = sorted(range(len(candidates)), key=lambda i: (-float(scores[i]), candidates[i]))
        library = [candidates[idx] for idx in ordered_idx[: config.n_sequences]]
        library = validator.filter_valid(library)
        if len(library) < config.n_sequences:
            raise RuntimeError("Could not produce enough library sequences after validation")
        library = library[: config.n_sequences]
        validator.assert_library(library, expected_size=config.n_sequences)

        top = guarded[: config.top_k]
        validator.assert_library(top, expected_size=config.top_k)

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