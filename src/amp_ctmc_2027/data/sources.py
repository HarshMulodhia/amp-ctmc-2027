"""Versioned source-catalog validation and idempotent downloads."""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path


def download_verified(url: str, destination: Path, expected_sha256: str) -> str:
    """Download once and verify SHA-256; never overwrite a mismatched existing artifact."""
    if not expected_sha256 or len(expected_sha256) != 64:
        raise ValueError("A published SHA-256 checksum is required before downloading")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        actual = hashlib.sha256(destination.read_bytes()).hexdigest()
        if actual != expected_sha256:
            raise ValueError(f"Existing file {destination} has SHA-256 {actual}, expected {expected_sha256}; refusing to replace it")
        return actual
    tmp = destination.with_suffix(destination.suffix + ".download")
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(url) as response, tmp.open("wb") as output:
            while block := response.read(1024 * 1024):
                digest.update(block)
                output.write(block)
        actual = digest.hexdigest()
        if actual != expected_sha256:
            raise ValueError(f"Downloaded checksum mismatch for {url}: got {actual}, expected {expected_sha256}")
        tmp.replace(destination)
        return actual
    finally:
        tmp.unlink(missing_ok=True)
