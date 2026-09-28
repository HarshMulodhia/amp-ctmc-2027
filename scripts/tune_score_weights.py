#!/usr/bin/env python
"""Grid-search score weights utility.

Usage:
    python scripts/tune_score_weights.py

This script loads a checkpoint, generates a candidate pool, computes per-component scores,
and searches for optimal weight combinations that maximize the composite score while
satisfying novelty and diversity constraints.

After running, inspect the tuned_weights.json and manually integrate into config.json
if satisfied with the results.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from amp_ctmc_2027.config import AMPConfig, set_global_determinism
from amp_ctmc_2027.core import (
    CTMCDenoiser,
    ReverseGenerationConfig,
    SinSquaredSchedule,
    TauLeapingSampler,
)
from amp_ctmc_2027.data.dataset import AMPCanvasEncoder
from amp_ctmc_2027.data.fasta_io import FastaRepository
from amp_ctmc_2027.discriminator import (
    DiscriminatorEnsemble,
    FeatureStats,
    PeptideFeatureExtractor,
)
from amp_ctmc_2027.infra import ConstraintValidator, log_device, resolve_device
from amp_ctmc_2027.objectives import (
    ConformityScorer,
    DiscriminatorScorer,
    MaskedPseudoLikelihoodScorer,
    MultiObjectiveScorer,
    NoveltyScorer,
    QualityScorer,
    ScoringContext,
    grid_search_weights,
)

logger = logging.getLogger(__name__)


def load_training_stats(path: Path) -> FeatureStats:
    """Load feature stats from training_stats.json."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    stats = payload["feature_stats"]
    return FeatureStats(
        means={k: float(v) for k, v in stats["means"].items()},
        stds={k: float(v) for k, v in stats["stds"].items()},
        q05={k: float(v) for k, v in stats["q05"].items()},
        q95={k: float(v) for k, v in stats["q95"].items()},
    )


def length_prior(
    training_sequences: list[str], min_length: int, max_length: int
) -> list[float]:
    """Build length prior from training set."""
    train_lengths = np.array([len(seq) for seq in training_sequences], dtype=np.int64)
    counts = np.bincount(train_lengths, minlength=max_length + 1)
    prior = counts[min_length : max_length + 1].astype(np.float64)
    prior = np.maximum(prior, 1e-6)
    prior = prior / prior.sum()
    return prior.tolist()


def run_tune_weights() -> None:
    """Execute weight tuning."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s"
    )

    fasta_repo = FastaRepository(Path.cwd())
    checkpoint_dir = fasta_repo.resolve(Path("checkpoint"))
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
    set_global_determinism(ckpt_config.seed)

    tokenizer_meta = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    encoder = AMPCanvasEncoder.from_metadata(tokenizer_meta)

    device = resolve_device(ckpt_config)
    log_device(device)

    model = CTMCDenoiser.load(
        model_path,
        config=ckpt_config,
        vocab_size=encoder.vocab_size,
        pad_idx=encoder.pad_idx,
        device=device,
    )
    model.eval()

    feature_extractor = PeptideFeatureExtractor(alphabet=ckpt_config.vocab)
    feature_stats = load_training_stats(stats_path)
    discriminator = DiscriminatorEnsemble.load(
        discriminator_path, extractor=feature_extractor
    )

    training_sequences = fasta_repo.read_sequences(ckpt_config.training_fasta_path)
    forbidden = set(fasta_repo.read_sequences(ckpt_config.antibacterial_fasta_path))
    train_lengths = np.array([len(seq) for seq in training_sequences], dtype=np.int64)
    length_lo, length_hi = np.quantile(train_lengths, [0.05, 0.95])
    len_prior = length_prior(
        training_sequences,
        ckpt_config.min_length,
        ckpt_config.max_length,
    )

    logger.info("Generating candidate pool...")
    validator = ConstraintValidator(
        alphabet=ckpt_config.vocab,
        min_len=ckpt_config.min_length,
        max_len=ckpt_config.max_length,
        forbidden=forbidden,
    )
    sampler = TauLeapingSampler(
        model=model,
        encoder=encoder,
        schedule=SinSquaredSchedule(),
        min_length=ckpt_config.min_length,
        length_prior=len_prior,
    )

    seen: set[str] = set()
    candidates: list[str] = []
    for attempt in range(ckpt_config.max_generation_attempts):
        for idx, (temp, steps, share) in enumerate(
            zip(
                ckpt_config.generation_temperatures,
                ckpt_config.generation_step_counts,
                ckpt_config.generation_shares,
            )
        ):
            batch_n = max(1, round(ckpt_config.generation_batch_size * share))
            run_config = ReverseGenerationConfig(
                steps=int(steps),
                temperature=float(temp),
                seed=ckpt_config.seed + attempt * 10_000 + idx,
                confidence_reveal_fraction=ckpt_config.confidence_reveal_fraction,
            )
            sampled = sampler.sample_batch(run_config, batch_n)
            sampled = [seq for seq in sampled if length_lo <= len(seq) <= length_hi]
            valid = validator.filter_valid(sampled, seen=seen)
            candidates.extend(valid)

        if len(candidates) >= ckpt_config.candidate_pool_size:
            break

    logger.info(
        "Generated %d candidates. Computing component scores...", len(candidates)
    )

    components = [
        DiscriminatorScorer(),
        ConformityScorer(),
        NoveltyScorer(),
        QualityScorer(),
        MaskedPseudoLikelihoodScorer(),
    ]
    context = ScoringContext(
        training_sequences=training_sequences,
        antibacterial_sequences=list(forbidden),
        feature_extractor=feature_extractor,
        feature_stats=feature_stats,
        discriminator=discriminator,
        model=model,
        encoder=encoder,
    )

    component_scores: dict[str, np.ndarray] = {}
    for component in components:
        logger.info("Scoring with %s...", component.name)
        raw = component.score(candidates, context).astype(np.float32)
        norm = MultiObjectiveScorer._rank_normalize(raw)
        component_scores[component.name] = norm

    def combine_geometric(
        scores_dict: dict[str, np.ndarray], weights: dict[str, float]
    ) -> np.ndarray:
        eps = 1e-6
        log_sum = None
        for name, scores in scores_dict.items():
            clipped = np.clip(scores, eps, 1.0)
            w = float(weights.get(name, 0.0))
            term = w * np.log(clipped)
            log_sum = term if log_sum is None else log_sum + term
        return np.exp(log_sum)

    logger.info("Searching weight combinations...")
    best_score, best_weights = grid_search_weights(
        component_scores=component_scores,
        combine_fn=combine_geometric,
        novelty_floor=0.4,
        step=0.1,
    )

    logger.info("Best composite score: %.4f", best_score)
    logger.info("Best weights: %s", best_weights)

    tuned_path = checkpoint_dir / "tuned_weights.json"
    tuned_path.write_text(json.dumps(best_weights, indent=2), encoding="utf-8")
    logger.info("Tuned weights written to %s", tuned_path)
    logger.info("Review results and manually update config.json if satisfied")


if __name__ == "__main__":
    run_tune_weights()
