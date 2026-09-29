"""Discriminator for AMP classification and feature extraction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import xgboost as xgb


@dataclass
class FeatureStats:
    """Summary statistics used by scoring components."""

    means: dict[str, float]
    stds: dict[str, float]
    q05: dict[str, float]
    q95: dict[str, float]


class PeptideFeatureExtractor:
    """Deterministic handcrafted AMP feature extractor."""

    hydrophobic_set: ClassVar[set[str]] = set("AVILMFWY")
    aromatic_set: ClassVar[set[str]] = set("FWY")

    def __init__(self, alphabet: str = "ACDEFGHIKLMNPQRSTVWY") -> None:
        self.alphabet = alphabet
        self.aa_index = {aa: idx for idx, aa in enumerate(alphabet)}
        self.feature_names = [
            "length",
            "net_charge",
            "hydrophobic_fraction",
            "composition_entropy",
            "aromatic_fraction",
            *[f"comp_{aa}" for aa in alphabet],
        ]

    def extract(self, sequence: str) -> np.ndarray:
        """Extract a single fixed-order feature vector."""
        return self.extract_batch([sequence])[0]

    def extract_batch(self, sequences: list[str]) -> np.ndarray:
        """Extract feature vectors for many sequences."""
        n = len(sequences)
        feats = np.zeros((n, len(self.feature_names)), dtype=np.float32)

        for row, sequence in enumerate(sequences):
            seq = sequence.strip().upper()
            length = max(len(seq), 1)
            counts = np.zeros(len(self.alphabet), dtype=np.float32)
            for aa in seq:
                idx = self.aa_index.get(aa)
                if idx is not None:
                    counts[idx] += 1

            composition = counts / length
            non_zero = composition > 0
            entropy = float(
                -np.sum(composition[non_zero] * np.log(composition[non_zero]))
            )
            net_charge = float(
                seq.count("K")
                + seq.count("R")
                + 0.1 * seq.count("H")
                - seq.count("D")
                - seq.count("E")
            )
            hydrophobic_fraction = float(
                sum(seq.count(aa) for aa in self.hydrophobic_set) / length
            )
            aromatic_fraction = float(
                sum(seq.count(aa) for aa in self.aromatic_set) / length
            )

            feats[row, 0] = float(len(seq))
            feats[row, 1] = net_charge
            feats[row, 2] = hydrophobic_fraction
            feats[row, 3] = entropy
            feats[row, 4] = aromatic_fraction
            feats[row, 5:] = composition

        return feats

    def fit_stats(self, sequences: list[str]) -> FeatureStats:
        """Fit robust feature statistics from training peptides."""
        feats = self.extract_batch(sequences)
        means = {
            name: float(feats[:, idx].mean())
            for idx, name in enumerate(self.feature_names)
        }
        stds = {
            name: float(feats[:, idx].std() + 1e-6)
            for idx, name in enumerate(self.feature_names)
        }
        q05 = {
            name: float(np.quantile(feats[:, idx], 0.05))
            for idx, name in enumerate(self.feature_names)
        }
        q95 = {
            name: float(np.quantile(feats[:, idx], 0.95))
            for idx, name in enumerate(self.feature_names)
        }
        return FeatureStats(means=means, stds=stds, q05=q05, q95=q95)


class AMPDiscriminator:
    """XGBoost binary classifier for AMP-vs-background prediction."""

    def __init__(
        self, extractor: PeptideFeatureExtractor, seed: int = 42, rounds: int = 200
    ) -> None:
        self.extractor = extractor
        self.seed = seed
        self.rounds = rounds
        self.booster: xgb.Booster | None = None

    def fit(
        self, positive_seqs: list[str], negative_seqs: list[str]
    ) -> AMPDiscriminator:
        """Train discriminator from scratch."""
        sequences = positive_seqs + negative_seqs
        labels = np.concatenate(
            [
                np.ones(len(positive_seqs), dtype=np.float32),
                np.zeros(len(negative_seqs), dtype=np.float32),
            ]
        )
        features = self.extractor.extract_batch(sequences)
        dtrain = xgb.DMatrix(
            features, label=labels, feature_names=self.extractor.feature_names
        )
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "seed": self.seed,
            "tree_method": "hist",
            "nthread": -1,
        }
        self.booster = xgb.train(
            params=params, dtrain=dtrain, num_boost_round=self.rounds
        )
        return self

    def predict_proba(self, sequences: list[str]) -> np.ndarray:
        """Predict AMP probabilities for sequences."""
        if self.booster is None:
            raise RuntimeError("Discriminator has not been trained")
        features = self.extractor.extract_batch(sequences)
        dtest = xgb.DMatrix(features, feature_names=self.extractor.feature_names)
        probs = self.booster.predict(dtest)
        return probs.astype(np.float32)

    def save(self, path: Path) -> None:
        """Serialize the discriminator model."""
        if self.booster is None:
            raise RuntimeError("Discriminator has not been trained")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
        self.booster.save_model(str(tmp))
        tmp.replace(path)

    @classmethod
    def load(
        cls, path: Path, extractor: PeptideFeatureExtractor, seed: int = 42
    ) -> AMPDiscriminator:
        """Load a serialized discriminator model."""
        model = cls(extractor=extractor, seed=seed)
        booster = xgb.Booster()
        booster.load_model(str(path))
        model.booster = booster
        return model

    def to_dict(self) -> dict:
        """Serialize to a JSON-embeddable XGBoost model string."""
        if self.booster is None:
            raise RuntimeError("Discriminator has not been trained")
        raw = bytes(self.booster.save_raw(raw_format="json")).decode("utf-8")
        return {
            "seed": self.seed,
            "rounds": self.rounds,
            "booster_json": raw,
        }

    def load_from_dict(self, data: dict) -> None:
        """Load from dictionary written by to_dict()."""
        payload = data["booster_json"]
        if isinstance(payload, list):
            raise TypeError(
                "discriminator.json was saved with get_dump() and cannot be loaded. "
                "Re-save the ensemble with save_raw(raw_format='json')."
            )
        if isinstance(payload, bytes):
            raw = payload
        else:
            raw = str(payload).encode("utf-8")
        booster = xgb.Booster()
        booster.load_model(bytearray(raw))
        self.booster = booster
        if "seed" in data:
            self.seed = int(data["seed"])
        if "rounds" in data:
            self.rounds = int(data["rounds"])


class DiscriminatorEnsemble:
    """Ensemble of discriminators for robust AMP probability estimates."""

    def __init__(self, extractor: PeptideFeatureExtractor, seeds: list[int]) -> None:
        self.extractor = extractor
        self.seeds = seeds
        self.members: list[AMPDiscriminator] = [
            AMPDiscriminator(extractor=extractor, seed=seed) for seed in seeds
        ]

    def fit(
        self,
        positive_sequences: list[str],
        background_sequences: list[str],
    ) -> DiscriminatorEnsemble:
        """Train all ensemble members."""
        for member in self.members:
            member.fit(positive_sequences, background_sequences)
        return self

    def predict_proba(self, sequences: list[str]) -> np.ndarray:
        """Average probabilities across ensemble members."""
        probs = np.stack(
            [member.predict_proba(sequences) for member in self.members],
            axis=0,
        )
        return probs.mean(axis=0).astype(np.float32)

    def save(self, path: Path) -> None:
        """Serialize all ensemble members to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
        payload = {
            "seeds": self.seeds,
            "members": [member.to_dict() for member in self.members],
        }
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(
        cls, path: Path, extractor: PeptideFeatureExtractor
    ) -> DiscriminatorEnsemble:
        """Load all ensemble members from disk."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        ensemble = cls(extractor=extractor, seeds=payload["seeds"])
        for member, member_dict in zip(ensemble.members, payload["members"]):
            member.load_from_dict(member_dict)
        return ensemble
