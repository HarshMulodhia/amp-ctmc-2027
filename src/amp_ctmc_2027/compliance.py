"""Submission checks and adapter for the organizer-provided identity validator."""
from __future__ import annotations

import hashlib
import importlib
from collections import Counter
from pathlib import Path


def load_official_identity(spec: str | None):
    if not spec or ":" not in spec:
        raise RuntimeError("Official identity validator is not configured; do not claim novelty compliance")
    module_name, function_name = spec.split(":", 1)
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise TypeError(f"Official identity adapter {spec} is not callable")
    return function


def validate_submission(library: list[str], top: list[str], references: list[str], *,
                       identity_function, threshold: float, min_length: int = 8,
                       max_length: int = 50, alphabet: str = "ACDEFGHIKLMNPQRSTVWY",
                       expected_library_count: int | None = 50000,
                       expected_top_count: int | None = 100) -> dict:
    """Enforce sequence rules, top membership, and organizer identity threshold.

    identity_function(candidate, reference) must return the organizer's [0,1] identity.
    """
    invalid = [(i, seq) for i, seq in enumerate(library) if not min_length <= len(seq) <= max_length or not seq or any(c not in alphabet for c in seq)]
    if invalid:
        raise ValueError(f"Invalid library sequences: {invalid[:3]}")
    if len(set(library)) != len(library):
        raise ValueError("Library contains duplicate sequences")
    if expected_library_count is not None and len(library) != expected_library_count:
        raise ValueError(f"Expected exactly {expected_library_count} library sequences; found {len(library)}")
    if len(set(top)) != len(top):
        raise ValueError("Top list contains duplicate sequences")
    if expected_top_count is not None and len(top) != expected_top_count:
        raise ValueError(f"Expected exactly {expected_top_count} ranked sequences; found {len(top)}")
    overlap = set(library).intersection(references)
    if overlap:
        raise ValueError(f"Library contains {len(overlap)} exact organizer reference sequence(s)")
    if not set(top).issubset(library):
        raise ValueError("Every ranked top sequence must occur in the library")
    nearest = []
    max_identity = 0.0
    for seq in top:
        if references:
            scored = [(float(identity_function(seq, ref)), ref) for ref in references]
            identity, reference = max(scored, key=lambda item: item[0])
        else:
            identity, reference = 0.0, None
        if not 0.0 <= identity <= 1.0:
            raise ValueError("Official identity function must return values in [0,1]")
        max_identity = max(max_identity, identity)
        nearest.append({"sequence": seq, "reference": reference, "identity": identity})
    if max_identity > threshold:
        raise ValueError(f"Top-list identity {max_identity:.6f} exceeds official threshold {threshold:.6f}")
    return {
        "pass": True, "library_count": len(library), "library_unique_count": len(set(library)),
        "top_count": len(top), "top_unique_count": len(set(top)), "top_is_library_subset": True,
        "max_top_identity": max_identity, "identity_threshold": threshold, "nearest_reference": nearest,
        "length_histogram": dict(sorted(Counter(map(len, library)).items())),
        "reference_sha256": hashlib.sha256("\n".join(references).encode()).hexdigest(),
    }
