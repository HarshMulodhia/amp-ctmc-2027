#!/usr/bin/env python3
"""Entry point for versioned teacher caches; requires a pinned local ESM checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from amp_ctmc_2027.data.fasta_io import FastaRepository
from amp_ctmc_2027.data.manifests import sha256_file
from amp_ctmc_2027.models.esm_multitask import ESMMultiTaskPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    metadata = json.loads(
        (args.checkpoint / "model_metadata.json").read_text(encoding="utf-8")
    )
    model, tokenizer = ESMMultiTaskPredictor.from_pretrained(
        metadata["model_name"], metadata.get("revision"), len(metadata["strain_ids"])
    )
    model.load_state_dict(
        torch.load(args.checkpoint / "model.pt", map_location="cpu", weights_only=True)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    sequences = FastaRepository(args.sequences.parent).read_sequences(args.sequences)
    args.output.mkdir(parents=True, exist_ok=True)
    token_count = min(max(map(len, sequences)) + 2, tokenizer.model_max_length)
    hidden_size = int(model.backbone.config.hidden_size)
    cache = np.lib.format.open_memmap(
        args.output / "teacher_residue_hidden.npy",
        mode="w+",
        dtype=np.float16,
        shape=(len(sequences), token_count, hidden_size),
    )
    for start in range(0, len(sequences), args.batch_size):
        batch = sequences[start : start + args.batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=token_count,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.inference_mode():
            output = model.backbone(**encoded).last_hidden_state
        cache[start : start + len(batch)] = output.cpu().numpy().astype(np.float16)
    cache.flush()
    (args.output / "teacher_cache_manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "residue_hidden_states",
                "model": metadata["model_name"],
                "revision": metadata.get("revision"),
                "sequence_count": len(sequences),
                "token_count": token_count,
                "hidden_size": hidden_size,
                "dtype": "float16",
                "checkpoint_sha256": sha256_file(args.checkpoint / "model.pt"),
                "sequence_sha256": [
                    hashlib.sha256(seq.encode("ascii")).hexdigest() for seq in sequences
                ],
                "cache_sha256": sha256_file(args.output / "teacher_residue_hidden.npy"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
