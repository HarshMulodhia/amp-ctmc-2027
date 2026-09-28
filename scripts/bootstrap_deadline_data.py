#!/usr/bin/env python3
"""Download exact pinned deadline datasets with bounded, hash-checked writes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(source: dict, directory: Path) -> dict:
    path = directory / source["filename"]
    expected = source["sha256"]
    if path.is_file() and sha256(path) == expected:
        actual = expected
    elif path.exists():
        raise ValueError(
            f"Existing pinned file has invalid SHA-256: {path}; expected {expected}; move it aside and rerun"
        )
    else:
        part = path.with_name(path.name + ".part")
        part.unlink(missing_ok=True)
        digest = hashlib.sha256()
        request = urllib.request.Request(
            source["url"], headers={"User-Agent": "amp-ctmc-deadline-bootstrap/1"}
        )
        try:
            with (
                urllib.request.urlopen(request, timeout=45) as response,
                part.open("wb") as out,
            ):
                length = int(response.headers.get("Content-Length", 0))
                count = 0
                while block := response.read(1024 * 1024):
                    out.write(block)
                    digest.update(block)
                    count += len(block)
                    if length:
                        print(
                            f"\r{source['id']}: {count}/{length} bytes ({100 * count / length:.1f}%)",
                            end="",
                            flush=True,
                        )
                    else:
                        print(f"\r{source['id']}: {count} bytes", end="", flush=True)
            print()
            actual = digest.hexdigest()
            if actual != expected:
                part.unlink(missing_ok=True)
                raise ValueError(
                    f"Pinned source hash mismatch: {source['url']} expected {expected}, got {actual}"
                )
            os.replace(part, path)
        except Exception:
            part.unlink(missing_ok=True)
            raise
    return {
        "id": source["id"],
        "filename": path.name,
        "url": source["url"],
        "source_revision": source["revision"],
        "expected_sha256": expected,
        "actual_sha256": actual,
        "size_bytes": path.stat().st_size,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config/deadline_sources.json"
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/raw/deadline")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for source in config["sources"]:
        if source.get("local_only"):
            local_path = ROOT / source["filename"]
            if not local_path.is_file():
                raise FileNotFoundError(
                    f"Required compliance reference is absent: {local_path}"
                )
            records.append(
                {
                    "id": source["id"],
                    "filename": source["filename"],
                    "url": source["url"],
                    "source_revision": source["revision"],
                    "expected_sha256": None,
                    "actual_sha256": sha256(local_path),
                    "size_bytes": local_path.stat().st_size,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "use": source["use"],
                }
            )
        else:
            records.append(download(source, args.output_dir))
    manifest = {
        "schema_version": config["schema_version"],
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": records,
    }
    (args.output_dir / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Verified {len(records)} pinned source files in {args.output_dir}")


if __name__ == "__main__":
    main()
