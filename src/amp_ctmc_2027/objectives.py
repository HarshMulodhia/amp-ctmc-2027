"""Objectives: scoring components, multi-objective optimization, and selection."""
from __future__ import annotations

import abc
import itertools
from typing import Callable, Literal

import numpy as np
import torch
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from amp_ctmc_2027.core import CTMCDenoiser
from amp_ctmc_2027.data.dataset import AMPCanvasEncoder
from amp_ctmc_2027.discriminator import DiscriminatorEnsemble, FeatureStats, PeptideFeatureExtractor
from amp_ctmc_2027.external_scorer import VendoredScorerClient


class ScoringContext:
    """Container for all scorer dependencies."""

    def __init__(
        self,
        training_sequences: list[str],
        antibacterial_sequences: list[str],
        feature_extractor: PeptideFeatureExtractor,
        feature_stats: FeatureStats,
        discriminator: DiscriminatorEnsemble,
        model: CTMCDenoiser,
        encoder: AMPCanvasEncoder,
        external_scorer: VendoredScorerClient | None = None,
    ) -> None:
        self.training_sequences = training_sequences
        self.antibacterial_sequences = antibacterial_sequences
        self.feature_extractor = feature_extractor
        self.feature_stats = feature_stats
        self.discriminator = discriminator
        self.model = model
        self.encoder = encoder
        self.external_scorer = external_scorer


class ScoreComponent(abc.ABC):
    """Abstract score component."""

    name: str

    @abc.abstractmethod
    def score(self, candidates: list[str], context: ScoringContext) -> np.ndarray:
        """Compute one raw score per candidate."""


class DiscriminatorScorer(ScoreComponent):
    """Scores by AMP discriminator probability."""

    name = "discriminator"

    def score(self, candidates: list[str], context: ScoringContext) -> np.ndarray:
        return context.discriminator.predict_proba(candidates)


class NoveltyScorer(ScoreComponent):
    """Scores by nearest-neighbor normalized edit distance to training set."""

    name = "novelty"

    def score(self, candidates: list[str], context: ScoringContext) -> np.ndarray:
        if not context.training_sequences:
            return np.zeros(len(candidates), dtype=np.float32)
        chunk_size = 256
        out = np.zeros(len(candidates), dtype=np.float32)
        for start in range(0, len(candidates), chunk_size):
            chunk = candidates[start : start + chunk_size]
            dists = process.cdist(
                chunk,
                context.training_sequences,
                scorer=Levenshtein.normalized_distance,
                dtype=np.float32,
            )
            out[start : start + len(chunk)] = np.min(dists, axis=1)
        return out


class ConformityScorer(ScoreComponent):
    """Scores closeness to training-set feature statistics."""

    name = "conformity"

    def score(self, candidates: list[str], context: ScoringContext) -> np.ndarray:
        feats = context.feature_extractor.extract_batch(candidates)
        idxs = [0, 1, 2, 3, 4]
        means = np.array([context.feature_stats.means[context.feature_extractor.feature_names[idx]] for idx in idxs])
        stds = np.array([context.feature_stats.stds[context.feature_extractor.feature_names[idx]] for idx in idxs])
        z = (feats[:, idxs] - means[None, :]) / stds[None, :]
        return np.exp(-0.5 * np.mean(z**2, axis=1)).astype(np.float32)


class QualityScorer(ScoreComponent):
    """Penalizes extreme physicochemical values outside broad quantile ranges."""

    name = "quality"

    def score(self, candidates: list[str], context: ScoringContext) -> np.ndarray:
        feats = context.feature_extractor.extract_batch(candidates)
        idxs = [0, 1, 2, 3, 4]
        penalties = np.zeros(len(candidates), dtype=np.float32)
        for idx in idxs:
            name = context.feature_extractor.feature_names[idx]
            low = context.feature_stats.q05[name]
            high = context.feature_stats.q95[name]
            width = max(high - low, 1e-6)
            below = np.maximum(0.0, low - feats[:, idx])
            above = np.maximum(0.0, feats[:, idx] - high)
            penalties += (below + above) / width
        return (1.0 / (1.0 + penalties)).astype(np.float32)


class RealismScorer(ScoreComponent):
    """Scores denoiser confidence on clean-time reconstructions."""

    name = "realism"

    def score(self, candidates: list[str], context: ScoringContext) -> np.ndarray:
        encoded = torch.stack([context.encoder.encode(seq) for seq in candidates]).to(next(context.model.parameters()).device)
        t = torch.ones((encoded.size(0),), device=encoded.device)
        with torch.inference_mode():
            logits = context.model(encoded, t)
            probs = torch.softmax(logits, dim=-1)
            conf = torch.max(probs, dim=-1).values
            non_pad = encoded != context.encoder.pad_idx
            numer = (conf * non_pad).sum(dim=1)
            denom = non_pad.sum(dim=1).clamp_min(1)
            score = numer / denom
        return score.detach().cpu().numpy().astype(np.float32)


class ActivityHemolysisScorer(ScoreComponent):
    """Scores candidates using vendored pretrained activity/hemolysis predictions."""

    name = "activity_hemolysis"

    def score(self, candidates: list[str], context: ScoringContext) -> np.ndarray:
        if context.external_scorer is None:
            return np.zeros(len(candidates), dtype=np.float32)
        payload = context.external_scorer.score(candidates)
        return payload["activity_hemolysis"]


class ESM2PseudoPerplexityScorer(ScoreComponent):
    """Scores candidates from ESM-2 pseudo-perplexity (lower perplexity is better)."""

    name = "esm2_pseudo_perplexity"

    def score(self, candidates: list[str], context: ScoringContext) -> np.ndarray:
        if context.external_scorer is None:
            return np.zeros(len(candidates), dtype=np.float32)
        payload = context.external_scorer.score(candidates)
        return -payload["esm2_pseudo_perplexity"]


class MultiObjectiveScorer:
    """Weighted rank-normalized multi-objective scorer."""

    def __init__(
        self,
        components: list[ScoreComponent],
        weights: dict[str, float],
        combine_mode: Literal["arithmetic", "geometric"] = "geometric",
    ) -> None:
        self.components = components
        self.weights = weights
        self.combine_mode = combine_mode

    @staticmethod
    def _rank_normalize(values: np.ndarray) -> np.ndarray:
        n = len(values)
        if n <= 1:
            return np.ones(n, dtype=np.float32)
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(n, dtype=np.float32)
        cursor = 0
        while cursor < n:
            end = cursor + 1
            while end < n and values[order[end]] == values[order[cursor]]:
                end += 1
            avg_rank = 0.5 * (cursor + end - 1)
            ranks[order[cursor:end]] = avg_rank
            cursor = end
        return (ranks / (n - 1)).astype(np.float32)

    def score(self, candidates: list[str], context: ScoringContext) -> np.ndarray:
        """Compute weighted combined score for candidates."""
        if not candidates:
            return np.array([], dtype=np.float32)

        if self.combine_mode == "geometric":
            return self._score_geometric(candidates, context)
        else:
            return self._score_arithmetic(candidates, context)

    def _score_geometric(self, candidates: list[str], context: ScoringContext) -> np.ndarray:
        """Compute weighted geometric mean of component scores."""
        eps = 1e-6
        log_sum = None
        for component in self.components:
            raw = component.score(candidates, context).astype(np.float32)
            norm = self._rank_normalize(raw)
            clipped = np.clip(norm, eps, 1.0)
            w = float(self.weights.get(component.name, 0.0))
            term = w * np.log(clipped)
            log_sum = term if log_sum is None else log_sum + term
        return np.exp(log_sum).astype(np.float32)

    def _score_arithmetic(self, candidates: list[str], context: ScoringContext) -> np.ndarray:
        """Compute weighted arithmetic mean of component scores."""
        total = np.zeros(len(candidates), dtype=np.float32)
        for component in self.components:
            raw = component.score(candidates, context).astype(np.float32)
            norm = self._rank_normalize(raw)
            total += float(self.weights.get(component.name, 0.0)) * norm
        return total


class GreedyMMRSelector:
    """Deterministic greedy MMR (Maximal Marginal Relevance) sequence selector."""

    def __init__(self, lambda_diversity: float) -> None:
        self.lambda_diversity = float(lambda_diversity)

    def select(self, candidates: list[str], scores: np.ndarray, target: int) -> list[str]:
        """Select target candidates with quality-diversity trade-off via MMR."""
        if len(candidates) != len(scores):
            raise ValueError("candidates and scores must have same length")
        if target <= 0:
            return []
        if len(candidates) <= target:
            ordered = sorted(zip(candidates, scores.tolist()), key=lambda x: (-x[1], x[0]))
            return [seq for seq, _ in ordered[:target]]

        ordered_idx = sorted(range(len(candidates)), key=lambda i: (-float(scores[i]), candidates[i]))
        cands = [candidates[i] for i in ordered_idx]
        score_arr = scores[np.array(ordered_idx)]

        n = len(cands)
        selected_mask = np.zeros(n, dtype=bool)
        max_similarity = np.zeros(n, dtype=np.float32)
        selected: list[str] = []

        for _ in range(min(target, n)):
            mmr = score_arr - self.lambda_diversity * max_similarity
            mmr[selected_mask] = -np.inf
            best_value = float(np.max(mmr))
            best_idxs = np.where(np.isclose(mmr, best_value))[0]
            best_idx = min(best_idxs.tolist(), key=lambda idx: cands[idx])
            selected_mask[best_idx] = True
            selected_seq = cands[best_idx]
            selected.append(selected_seq)

            remaining = np.where(~selected_mask)[0]
            if remaining.size == 0:
                continue
            remaining_sequences = [cands[idx] for idx in remaining.tolist()]
            distances = process.cdist(
                [selected_seq],
                remaining_sequences,
                scorer=Levenshtein.normalized_distance,
                dtype=np.float32,
            )[0]
            similarities = 1.0 - distances
            max_similarity[remaining] = np.maximum(max_similarity[remaining], similarities)

        return selected


def grid_search_weights(
    component_scores: dict[str, np.ndarray],
    combine_fn: Callable[[dict[str, np.ndarray], dict[str, float]], np.ndarray],
    novelty_floor: float = 0.4,
    identity_ceiling: float = 0.3,
    pairwise_identity_fn: Callable[[], float] | None = None,
    step: float = 0.1,
) -> tuple[float, dict[str, float]]:
    """Grid search over simplex-constrained weights to maximize mean composite score
    subject to novelty and diversity floors. Returns (best_score, best_weights).

    Args:
        component_scores: Dict mapping component names to score arrays [n_candidates]
        combine_fn: Function that combines component scores given weights
        novelty_floor: Minimum mean novelty score to satisfy
        identity_ceiling: Maximum mean pairwise identity to satisfy
        pairwise_identity_fn: Optional callable that returns current mean pairwise identity
        step: Grid resolution (e.g., 0.1 gives weights in [0.0, 0.1, 0.2, ...])

    Returns:
        (best_mean_score, best_weights_dict)
    """
    names = list(component_scores.keys())
    grid = np.arange(0.0, 1.0 + step, step)
    best: tuple[float, dict[str, float]] | None = None

    for combo in itertools.product(grid, repeat=len(names)):
        if not np.isclose(sum(combo), 1.0, atol=step / 2):
            continue
        weights = dict(zip(names, combo))
        composite = combine_fn(component_scores, weights)

        if "novelty" in component_scores and component_scores["novelty"].mean() < novelty_floor:
            continue
        if pairwise_identity_fn is not None and pairwise_identity_fn() > identity_ceiling:
            continue

        mean_score = float(composite.mean())
        if best is None or mean_score > best[0]:
            best = (mean_score, weights)

    if best is None:
        raise RuntimeError("No weight combination satisfied the constraints")
    return best
