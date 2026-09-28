#!/usr/bin/env python3
"""Evaluate a generated AMP library against training statistics and hard constraints.

Run from the repo root:

    uv run python eval_library.py
    uv run python eval_library.py --library generate/library.fasta
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from amp_ctmc_2027.config import AMPConfig
from amp_ctmc_2027.core import CTMCDenoiser
from amp_ctmc_2027.data.dataset import AMPCanvasEncoder
from amp_ctmc_2027.data.fasta_io import FastaRepository
from amp_ctmc_2027.discriminator import (
    DiscriminatorEnsemble,
    FeatureStats,
    PeptideFeatureExtractor,
)
from amp_ctmc_2027.infra import ConstraintValidator, resolve_device
from amp_ctmc_2027.objectives import (
    DiscriminatorScorer,
    MaskedPseudoLikelihoodScorer,
    NoveltyScorer,
    ScoringContext,
)


def _summarize(name: str, values: np.ndarray) -> str:
    return (
        f"{name:22s} mean={values.mean():.3f}  "
        f"p10={np.quantile(values, 0.10):.3f}  "
        f"p50={np.quantile(values, 0.50):.3f}  "
        f"p90={np.quantile(values, 0.90):.3f}  "
        f"min={values.min():.3f}  max={values.max():.3f}"
    )


def _length_histogram(lengths: np.ndarray) -> str:
    bins = [(8, 12), (13, 20), (21, 30), (31, 40), (41, 48), (49, 50)]
    parts = []
    n = max(len(lengths), 1)
    for lo, hi in bins:
        count = int(((lengths >= lo) & (lengths <= hi)).sum())
        parts.append(f"{lo}-{hi}:{count}({100.0 * count / n:.1f}%)")
    return "  ".join(parts)


def _load_model(
    ckpt: Path, cfg: AMPConfig, encoder: AMPCanvasEncoder, device
) -> CTMCDenoiser:
    kwargs = {
        "path": ckpt / "model.pt",
        "config": cfg,
        "vocab_size": encoder.vocab_size,
        "device": device,
    }
    if "pad_idx" in inspect.signature(CTMCDenoiser.load).parameters:
        kwargs["pad_idx"] = encoder.pad_idx
    if "path" not in inspect.signature(CTMCDenoiser.load).parameters:
        model = CTMCDenoiser.load(
            kwargs["path"],
            config=kwargs["config"],
            vocab_size=kwargs["vocab_size"],
            **(
                {"pad_idx": encoder.pad_idx}
                if "pad_idx" in inspect.signature(CTMCDenoiser.load).parameters
                else {}
            ),
            device=kwargs["device"],
        )
        return model
    try:
        return CTMCDenoiser.load(**kwargs)
    except TypeError:
        return CTMCDenoiser.load(
            kwargs["path"],
            config=kwargs["config"],
            vocab_size=kwargs["vocab_size"],
            device=kwargs["device"],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated AMP library")
    parser.add_argument(
        "--library", type=Path, default=Path("generate_broad_spectrum/library.fasta")
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoint"))
    parser.add_argument("--pairwise-subset", type=int, default=400)
    parser.add_argument("--train-score-sample", type=int, default=2000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path.cwd()
    repo = FastaRepository(root)
    ckpt = repo.resolve(args.checkpoint_dir)
    library_path = repo.resolve(args.library)

    cfg = AMPConfig.from_json_file(ckpt / "config.json")
    lib = repo.read_sequences(library_path)
    train = repo.read_sequences(cfg.training_fasta_path)
    forbidden = set(repo.read_sequences(cfg.antibacterial_fasta_path))
    train_set = set(train)

    print(f"library: {library_path}")
    print(f"checkpoint: {ckpt}")
    print(f"n_library={len(lib)}  n_train={len(train)}  n_forbidden={len(forbidden)}")

    validator = ConstraintValidator(
        cfg.vocab, cfg.min_length, cfg.max_length, forbidden
    )
    try:
        validator.assert_library(lib, expected_size=len(lib))
        print("HARD CONSTRAINTS: PASS")
    except AssertionError as exc:
        print(f"HARD CONSTRAINTS: FAIL ({exc})")
        return 1

    encoder = AMPCanvasEncoder.from_metadata(
        json.loads((ckpt / "tokenizer.json").read_text())
    )
    device = resolve_device(cfg)
    model = _load_model(ckpt, cfg, encoder, device)
    model.eval()

    extractor = PeptideFeatureExtractor(alphabet=cfg.vocab)
    stats_payload = json.loads((ckpt / "training_stats.json").read_text())[
        "feature_stats"
    ]
    feature_stats = FeatureStats(
        means={k: float(v) for k, v in stats_payload["means"].items()},
        stds={k: float(v) for k, v in stats_payload["stds"].items()},
        q05={k: float(v) for k, v in stats_payload["q05"].items()},
        q95={k: float(v) for k, v in stats_payload["q95"].items()},
    )
    disc = DiscriminatorEnsemble.load(ckpt / "discriminator.json", extractor=extractor)
    ctx = ScoringContext(
        train, list(forbidden), extractor, feature_stats, disc, model, encoder
    )

    train_score_n = min(args.train_score_sample, len(train))
    p_lib = DiscriminatorScorer().score(lib, ctx)
    p_train = DiscriminatorScorer().score(train[:train_score_n], ctx)
    novelty = NoveltyScorer().score(lib, ctx)
    pseudo_likelihood = MaskedPseudoLikelihoodScorer().score(lib, ctx)
    feats = extractor.extract_batch(lib)
    feats_tr = extractor.extract_batch(train[:train_score_n])

    lengths = np.array([len(s) for s in lib], dtype=np.float32)
    train_lengths = np.array([len(s) for s in train], dtype=np.float32)
    length_lo, length_hi = np.quantile(train_lengths, [0.05, 0.95])
    in_train_length = float(((lengths >= length_lo) & (lengths <= length_hi)).mean())
    frac_long = float((lengths >= 49).mean())

    subset_n = min(args.pairwise_subset, len(lib))
    subset = lib[:subset_n]
    pair = process.cdist(
        subset, subset, scorer=Levenshtein.normalized_similarity, dtype=np.float32
    )
    np.fill_diagonal(pair, 0.0)
    mean_id = float(pair.mean()) if subset_n > 1 else 0.0
    frac_near = float((pair > 0.70).mean()) if subset_n > 1 else 0.0
    kmers = [s[i : i + 4] for s in lib for i in range(max(len(s) - 3, 0))]
    unique_4mers = len(set(kmers))

    print("\n=== RAW SCORES (not rank-normalized) ===")
    print(_summarize("disc P(AMP) library", p_lib))
    print(_summarize("disc P(AMP) train", p_train))
    print(_summarize("novelty vs train", novelty))
    print(_summarize("masked pseudo-likelihood", pseudo_likelihood))
    print(f"exact train copies   {sum(seq in train_set for seq in lib)}")
    print(f"exact forbidden      {sum(seq in forbidden for seq in lib)}")

    print("\n=== FEATURES vs TRAIN ===")
    feature_names = [
        "length",
        "net_charge",
        "hydrophobic_fraction",
        "composition_entropy",
        "aromatic_fraction",
    ]
    for idx, name in enumerate(feature_names):
        print(
            f"{name:22s} lib={feats[:, idx].mean():.3f}±{feats[:, idx].std():.3f}  "
            f"train={feats_tr[:, idx].mean():.3f}±{feats_tr[:, idx].std():.3f}"
        )

    print("\n=== LENGTH ===")
    print(
        f"library mean/std/min/max {lengths.mean():.1f} {lengths.std():.1f} "
        f"{lengths.min():.0f} {lengths.max():.0f}"
    )
    print(
        f"train   mean/std/min/max {train_lengths.mean():.1f} {train_lengths.std():.1f} "
        f"{train_lengths.min():.0f} {train_lengths.max():.0f}"
    )
    print(f"train 5-95% length band  {length_lo:.0f}-{length_hi:.0f}")
    print(f"fraction in train 5-95%  {in_train_length:.3f}")
    print(f"fraction length >= 49    {frac_long:.3f}")
    print(f"histogram                {_length_histogram(lengths)}")

    print("\n=== DIVERSITY ===")
    print(f"unique sequences     {len(set(lib))}/{len(lib)}")
    print(f"mean pairwise identity ({subset_n} subset) {mean_id:.3f}")
    print(f"fraction of pairs >70% identity    {frac_near:.3f}")
    print(f"unique 4-mers                      {unique_4mers}")
    print("first 8 sequences:")
    for seq in lib[:8]:
        print(f"  {len(seq):2d}  {seq}")

    print("\n=== VERDICT ===")
    flags: list[str] = []
    if frac_long > 0.5 or abs(float(lengths.mean()) - float(train_lengths.mean())) > 10:
        flags.append(
            "LENGTH COLLAPSE: library is stuck near the 50-token canvas. "
            "Fix EOS sampling; do not add Metropolis-Hastings."
        )
    if float(p_lib.mean()) > 0.97 and float(p_lib.mean()) - float(p_train.mean()) > 0.1:
        flags.append(
            "DISCRIMINATOR SATURATION: P(AMP)~1.0 is classifier exploitation, not AMP quality. "
            "Do not optimize MH against discriminator probability."
        )
    if float(novelty.mean()) < 0.10 or sum(seq in train_set for seq in lib) > 0:
        flags.append("TRAIN COPIES: library is too close to training sequences.")
    if mean_id > 0.45 or frac_near > 0.05:
        flags.append("MODE COLLAPSE: sequences are near-duplicates of each other.")
    if not flags:
        flags.append(
            "HEALTHY ENOUGH: constraints pass, diversity looks fine, length matches training. "
            "Keep this library; MH is not required."
        )
    for flag in flags:
        print(f"- {flag}")

    print("\nRecommended next step:")
    if frac_long > 0.5:
        print(
            f"- Reject candidates outside training length band {length_lo:.0f}-{length_hi:.0f} "
            "and force EOS using a training length prior during reverse sampling."
        )
    else:
        print("- Keep the library. Re-run scoring only if you change sampling.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
