#!/usr/bin/env python3
"""Verify the bundled ESM-2 150M snapshot (offline check) or download it from HuggingFace.

The production snapshot (facebook/esm2_t30_150M_UR50D) is committed to this
repository under data/pretrained/esm2_t30_150M_UR50D/ via Git LFS.  A clean
clone should already have all files — this script is provided only as a
convenience for environments where Git LFS was not pulled or where the snapshot
must be refreshed from the canonical HuggingFace revision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

MODEL_ID = "facebook/esm2_t30_150M_UR50D"
REVISION = "a695f6045e2e32885fa60af20c13cb35398ce30c"
DESTINATION = (
    Path(__file__).resolve().parents[1] / "data/pretrained/esm2_t30_150M_UR50D"
)
RESIDUES = "ACDEFGHIKLMNPQRSTVWY"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path) -> None:
    from transformers import AutoTokenizer, EsmModel

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    EsmModel.from_pretrained(str(path), local_files_only=True)
    ids = [tokenizer.encode(char, add_special_tokens=False) for char in RESIDUES]
    if (
        any(len(tokens) != 1 for tokens in ids)
        or len({tokens[0] for tokens in ids}) != 20
    ):
        raise ValueError(
            "Pinned tokenizer does not encode the 20 canonical amino acids distinctly"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline-check", action="store_true")
    parser.add_argument("--output", type=Path, default=DESTINATION)
    args = parser.parse_args()
    path = args.output
    if not args.offline_check:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "The already-installed huggingface_hub is required"
            ) from exc
        path.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=MODEL_ID,
            revision=REVISION,
            local_dir=str(path),
            local_dir_use_symlinks=False,
        )
    if not path.is_dir():
        raise FileNotFoundError(
            f"ESM-2 150M snapshot is absent: {path}\n"
            "Run `git lfs pull` to restore the bundled snapshot, or run this "
            "script without --offline-check to download from HuggingFace."
        )
    verify(path)
    files = [
        {
            "path": item.relative_to(path).as_posix(),
            "sha256": sha256(item),
            "size_bytes": item.stat().st_size,
        }
        for item in sorted(path.rglob("*"))
        if item.is_file() and item.name != "file_manifest.json"
    ]
    manifest = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "source": f"https://huggingface.co/{MODEL_ID}/tree/{REVISION}",
        "files": files,
    }
    manifest_path = path / "file_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {MODEL_ID}@{REVISION} in {path} ({len(files)} files)")


if __name__ == "__main__":
    main()
