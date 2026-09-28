"""Objectives: scoring components, multi-objective optimization, and selection."""
from __future__ import annotations

import abc
import itertools
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
import torch
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from amp_ctmc_2027.core import CTMCDenoiser
from amp_ctmc_2027.data.dataset import AMPCanvasEncoder
from amp_ctmc_2027.discriminator import DiscriminatorEnsemble, FeatureStats, PeptideFeatureExtractor
from amp_ctmc_2027.external_scorer import VendoredScorerClient


@dataclass
class ScoreTable:
    """Auditable raw, rank-normalized scores with component metadata."""
    sequences: list[str]
    columns: dict[str, np.ndarray] = field(default_factory=dict)
    metadata: dict[str, dict] = field(default_factory=dict)

    def write_csv(self, path) -> None:
        import csv
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        keys = list(self.columns)
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["sequence", *keys])
            for i, seq in enumerate(self.sequences):
                writer.writerow([seq, *(float(self.columns[k][i]) for k in keys)])
        temporary.replace(path)


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
    direction: Literal["maximize", "minimize"] = "maximize"
    required_artifacts: tuple[str, ...] = ()
    cost_tier: Literal["cheap", "medium", "expensive"] = "cheap"
    batch_size: int = 1024
    raw_units: str = "unitless"
    failure_policy: Literal["raise", "skip_if_zero_weight"] = "raise"

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
    direction = "maximize"
    raw_units = "normalized_edit_distance_design_proxy"

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


class MaskedPseudoLikelihoodScorer(ScoreComponent):
    """Observed-token masked log likelihood from the lightweight student."""
    name = "masked_pseudo_likelihood"
    cost_tier = "expensive"
    raw_units = "mean_log_probability_per_residue"
    batch_size = 128

    def score(self, candidates: list[str], context: ScoringContext) -> np.ndarray:
        device = next(context.model.parameters()).device
        scores: list[float] = []
        for start in range(0, len(candidates), self.batch_size):
            encoded = torch.stack([context.encoder.encode(s) for s in candidates[start:start+self.batch_size]]).to(device)
            batch_scores = torch.zeros(encoded.shape[0], device=device)
            count = torch.zeros_like(batch_scores)
            residue_mask = encoded.lt(len(context.encoder.vocab))
            for pos in range(context.encoder.max_length):
                selected = residue_mask[:, pos]
                if not selected.any():
                    continue
                masked = encoded[selected].clone()
                masked[:, pos] = context.encoder.mask_idx
                t = torch.full((masked.shape[0],), 0.5, device=device)
                with torch.inference_mode():
                    logp = torch.log_softmax(context.model(masked, t), dim=-1)
                target = encoded[selected, pos]
                batch_scores[selected] += logp[torch.arange(target.numel(), device=device), pos, target]
                count[selected] += 1
            scores.extend((batch_scores / count.clamp_min(1)).cpu().tolist())
        return np.asarray(scores, dtype=np.float32)


class ActivityHemolysisScorer(ScoreComponent):
    """Scores candidates using vendored pretrained activity/hemolysis predictions."""

    name = "activity_hemolysis"
    cost_tier = "medium"
    required_artifacts = ("scorer/models",)

    def score(self, candidates: list[str], context: ScoringContext) -> np.ndarray:
        if context.external_scorer is None:
            return np.zeros(len(candidates), dtype=np.float32)
        payload = context.external_scorer.score(candidates)
        return payload["activity_hemolysis"]


class ESM2PseudoPerplexityScorer(ScoreComponent):
    """Scores candidates from ESM-2 pseudo-perplexity (lower perplexity is better)."""

    name = "esm2_pseudo_perplexity"
    direction = "maximize"
    cost_tier = "expensive"
    required_artifacts = ("scorer/models",)
    raw_units = "negative_pseudo_perplexity"

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

        active = [c for c in self.components if float(self.weights.get(c.name, 0.0)) > 0.0]
        if not active:
            raise ValueError("At least one enabled score component must have a positive weight")
        for component in active:
            if component.failure_policy == "raise" and component.required_artifacts and context.external_scorer is None and any("scorer" in p for p in component.required_artifacts):
                raise RuntimeError(f"Score component {component.name} requires artifacts {component.required_artifacts}")

        if self.combine_mode == "geometric":
            return self._score_geometric(candidates, context)
        else:
            return self._score_arithmetic(candidates, context)

    def score_table(self, candidates: list[str], context: ScoringContext) -> ScoreTable:
        """Evaluate active components and retain raw and rank-normalized columns."""
        table = ScoreTable(list(candidates))
        for component in self.components:
            weight = float(self.weights.get(component.name, 0.0))
            if weight == 0.0:
                table.metadata[component.name] = {
                    "status": "skipped_zero_weight", "weight": 0.0, "model_version": None,
                    "uncertainty": None, "calibration": None, "failure_state": None,
                }
                continue
            raw = np.asarray(component.score(candidates, context), dtype=np.float32)
            if raw.shape != (len(candidates),) or not np.isfinite(raw).all():
                raise ValueError(f"Score component {component.name} returned invalid values")
            table.columns[f"{component.name}.raw"] = raw
            oriented = -raw if component.direction == "minimize" else raw
            table.columns[f"{component.name}.rank"] = self._rank_normalize(oriented)
            table.metadata[component.name] = {
                "status": "ok", "weight": weight, "direction": component.direction,
                "cost_tier": component.cost_tier, "raw_units": component.raw_units,
                "required_artifacts": list(component.required_artifacts), "failure_policy": component.failure_policy,
                "model_version": "not_recorded", "uncertainty": "unavailable", "calibration": "not_fitted",
                "failure_state": None,
            }
        return table

    def _score_geometric(self, candidates: list[str], context: ScoringContext) -> np.ndarray:
        """Compute weighted geometric mean of component scores."""
        eps = 1e-6
        log_sum = None
        for component in self.components:
            w = float(self.weights.get(component.name, 0.0))
            if w == 0:
                continue
            raw = component.score(candidates, context).astype(np.float32)
            if raw.shape != (len(candidates),) or not np.isfinite(raw).all():
                raise ValueError(f"Score component {component.name} returned invalid values")
            if component.direction == "minimize":
                raw = -raw
            norm = self._rank_normalize(raw)
            clipped = np.clip(norm, eps, 1.0)
            term = w * np.log(clipped)
            log_sum = term if log_sum is None else log_sum + term
        return np.exp(log_sum).astype(np.float32)

    def _score_arithmetic(self, candidates: list[str], context: ScoringContext) -> np.ndarray:
        """Compute weighted arithmetic mean of component scores."""
        total = np.zeros(len(candidates), dtype=np.float32)
        for component in self.components:
            w = float(self.weights.get(component.name, 0.0))
            if w == 0:
                continue
            raw = component.score(candidates, context).astype(np.float32)
            if raw.shape != (len(candidates),) or not np.isfinite(raw).all():
                raise ValueError(f"Score component {component.name} returned invalid values")
            if component.direction == "minimize":
                raw = -raw
            norm = self._rank_normalize(raw)
            total += w * norm
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
