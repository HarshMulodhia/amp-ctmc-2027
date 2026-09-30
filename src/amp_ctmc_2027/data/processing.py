"""Consolidated data processing, observation schema, validation, and splitting."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import urllib.request
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

CANONICAL = frozenset("ACDEFGHIKLMNPQRSTVWY")

MODIFICATION_TERMS = (
    "cyclic",
    "branched",
    "stapled",
    "lipid",
    "glycosyl",
    "peg",
    "dendrimer",
    "terminal",
    "noncanonical",
)

DEADLINE_STRAIN_COLUMNS = {
    "A_baumannii_ATCC19606": "A. baumannii ATCC19606",
    "E_coli_ATCC11775": "E. coli ATCC11775",
    "E_coli_AIC221": "E. coli AIG221",
    "E_coli_AIC222": "E. coli AIG222",
    "K_pneumoniae_ATCC13883": "K. pneumoniae ATCC13883",
    "P_aeruginosa_PAO1": "P. aeruginosa PAO1",
    "P_aeruginosa_PA14": "P. aeruginosa PA14",
    "S_aureus_ATCC12600": "S. aureus ATCC12600",
    "S_aureus_ATCC_BAA1556": "S. aureus (ATCC BAA-1556) - MRSA",
    "E_faecalis_ATCC700802": "vancomycin-resistant E. faecalis ATCC700802",
    "E_faecium_ATCC700221": "vancomycin-resistant E. faecium ATCC700221",
}

DEADLINE_STRAIN_GROUPS = {
    "gram_negative": [
        "A_baumannii_ATCC19606",
        "E_coli_ATCC11775",
        "E_coli_AIC221",
        "E_coli_AIC222",
        "K_pneumoniae_ATCC13883",
        "P_aeruginosa_PAO1",
        "P_aeruginosa_PA14",
    ],
    "gram_positive": [
        "S_aureus_ATCC12600",
        "S_aureus_ATCC_BAA1556",
        "E_faecalis_ATCC700802",
        "E_faecium_ATCC700221",
    ],
    "mdr": [
        "E_coli_AIC222",
        "S_aureus_ATCC_BAA1556",
        "E_faecalis_ATCC700802",
        "E_faecium_ATCC700221",
    ],
}


@dataclass(frozen=True)
class AMPObservation:
    """Canonical observation schema for AMP sequences and assay labels."""

    sequence: str
    sequence_sha256: str
    source: str
    source_record_id: str | None = None
    is_amp: float | None = None
    is_hemolytic: float | None = None
    organism: str | None = None
    strain: str | None = None
    panel_target: str | None = None
    mic_value: float | None = None
    mic_unit_original: str | None = None
    mic_uM: float | None = None
    mic_upper_uM: float | None = None
    mic_censor: Literal["none", "left", "right", "interval"] | None = None
    hc50_uM: float | None = None
    hc50_upper_uM: float | None = None
    hc50_censor: Literal["none", "left", "right", "interval"] | None = None
    terminal_modification: str | None = None
    other_modification: str | None = None
    is_linear: bool | None = None
    split_cluster_id: str | None = None
    split: Literal["train", "val", "test"] | None = None
    label_provenance: Literal["measured", "aggregated", "pseudo", "missing"] = "missing"

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CleaningResult:
    observations: tuple[AMPObservation, ...]
    rejected: tuple[dict, ...]
    conflicts: tuple[dict, ...]
    duplicate_count: int


# --- Hashing & Manifest Utilities ---


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_manifest(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**data, "code_commit": data.get("code_commit", git_commit())}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    tmp.replace(path)


# --- Download Utilities ---


def download_verified(url: str, destination: Path, expected_sha256: str) -> str:
    """Download once and verify SHA-256; never overwrite a mismatched existing artifact."""
    if not expected_sha256 or len(expected_sha256) != 64:
        raise ValueError("A published SHA-256 checksum is required before downloading")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        actual = hashlib.sha256(destination.read_bytes()).hexdigest()
        if actual != expected_sha256:
            raise ValueError(
                f"Existing file {destination} has SHA-256 {actual}, expected {expected_sha256}; refusing to replace it"
            )
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
            raise ValueError(
                f"Downloaded checksum mismatch for {url}: got {actual}, expected {expected_sha256}"
            )
        tmp.replace(destination)
        return actual
    finally:
        tmp.unlink(missing_ok=True)


# --- Unit Conversion & Parsing Helpers ---


def concentration_to_um(
    value: float, unit: str, molecular_weight_g_mol: float | None = None
) -> float:
    """Convert molar concentration to µM; mass units require explicit molecular weight."""
    normalized = (
        unit.strip().lower().replace("μ", "u").replace("µ", "u").replace(" ", "")
    )
    molar_factors = {"m": 1_000_000.0, "mm": 1_000.0, "um": 1.0, "nm": 1e-3, "pm": 1e-6}
    if normalized in molar_factors:
        return float(value) * molar_factors[normalized]
    mass_as_g_l = {"mg/l": 1e-3, "ug/ml": 1e-3, "g/l": 1.0, "mg/ml": 1.0}
    if normalized not in mass_as_g_l:
        raise ValueError(f"Unsupported concentration unit: {unit}")
    if molecular_weight_g_mol is None or molecular_weight_g_mol <= 0:
        raise ValueError(
            "Mass concentration conversion requires a positive molecular weight in g/mol"
        )
    return (
        float(value)
        * mass_as_g_l[normalized]
        / float(molecular_weight_g_mol)
        * 1_000_000.0
    )


def grampa_log10_mic_to_um(value: float) -> float:
    result = 10.0 ** float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("GRAMPA log10 MIC must convert to finite positive micromolar")
    return result


def hc50_log10_to_um(value: float) -> float:
    result = 10.0 ** float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("log10 HC50 must convert to finite positive micromolar")
    return result


def hemopi2_hemolysis_label(value: float) -> float:
    numeric = float(value)
    if numeric not in (0.0, 1.0):
        raise ValueError("HemoPI2 label must be 0 or 1")
    return numeric


# --- Sequence Splitting & Homology Clustering ---


def deterministic_sequence_split(sequence: str, seed: int = 42) -> str:
    """Stable exact-sequence hash split with nominal 80/10/10 proportions."""
    value = (
        int.from_bytes(
            hashlib.sha256(f"{seed}:{sequence}".encode()).digest()[:8], "big"
        )
        / 2**64
    )
    return "train" if value < 0.8 else "val" if value < 0.9 else "test"


def assign_cluster_splits(
    cluster_ids: list[str],
    seed: int = 42,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> list[str]:
    """Assign cluster groups deterministically without splitting any cluster."""
    if (
        len(fractions) != 3
        or any(f < 0 for f in fractions)
        or abs(sum(fractions) - 1.0) > 1e-8
    ):
        raise ValueError(
            "split fractions must be three nonnegative values summing to one"
        )
    unique = sorted(
        set(cluster_ids),
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).digest(),
    )
    n_train = int(len(unique) * fractions[0])
    n_val = int(len(unique) * fractions[1])
    mapping = {cluster: "train" for cluster in unique[:n_train]}
    mapping.update({cluster: "val" for cluster in unique[n_train : n_train + n_val]})
    mapping.update({cluster: "test" for cluster in unique[n_train + n_val :]})
    result = [mapping[c] for c in cluster_ids]
    groups: dict[str, set[str]] = {}
    for cluster, split in zip(cluster_ids, result):
        groups.setdefault(cluster, set()).add(split)
    if any(len(v) != 1 for v in groups.values()):
        raise AssertionError("a homology cluster crossed splits")
    return result


def cluster_fasta(
    input_fasta: Path, output_dir: Path, identity: float = 0.5, coverage: float = 0.8
) -> dict[str, str]:
    """MMseqs2 sequence clustering adapter."""
    from amp_ctmc_2027.data.fasta_io import FastaRepository

    if not 0.0 < identity <= 1.0 or not 0.0 < coverage <= 1.0:
        raise ValueError("identity and coverage must be in (0, 1]")
    exe = shutil.which("mmseqs")
    if exe is None:
        raise RuntimeError(
            "MMseqs2 is required for homology-aware splits (install the pinned version from configs)"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    repo = FastaRepository(input_fasta.parent)
    sequences = repo.read_sequences(input_fasta)
    ids = {f"seq{i}": seq for i, seq in enumerate(sequences)}
    fasta = output_dir / "cluster_input.fasta"
    fasta.write_text(
        "".join(f">seq{i}\n{seq}\n" for i, seq in enumerate(sequences)),
        encoding="utf-8",
    )
    db, clu, tmp, tsv = (
        output_dir / name for name in ("db", "cluster", "tmp", "clusters.tsv")
    )
    subprocess.run(
        [exe, "createdb", str(fasta), str(db)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            exe,
            "cluster",
            str(db),
            str(clu),
            str(tmp),
            "--min-seq-id",
            str(identity),
            "-c",
            str(coverage),
            "--cov-mode",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [exe, "createtsv", str(db), str(db), str(clu), str(tsv)],
        check=True,
        capture_output=True,
        text=True,
    )
    assignments: dict[str, str] = {}
    for line in tsv.read_text().splitlines():
        rep_id, member_id = line.split("\t")[:2]
        assignments[member_id] = rep_id
    return {
        ids[key]: cluster_id for key, cluster_id in assignments.items() if key in ids
    }


# --- Sequence Cleaning & Observation Normalization ---


def _parse_bool(val: object, default: bool = True) -> bool:
    """Parse string or mixed-type boolean flags robustly."""
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("false", "0", "no", "off", "f"):
            return False
        if s in ("true", "1", "yes", "on", "t"):
            return True
    return bool(val)


def clean_sequence(seq: str) -> tuple[str | None, str | None]:
    """Clean and canonicalize amino acid sequence."""
    if not seq or not isinstance(seq, str):
        return None, "empty_sequence"
    cleaned = seq.strip().upper()
    if not cleaned:
        return None, "empty_sequence"
    if not set(cleaned).issubset(CANONICAL):
        return None, "noncanonical_residue"
    return cleaned, None


def clean_observations(
    rows: Iterable[Mapping[str, object]],
    compliance_references: set[str] | None = None,
) -> CleaningResult:
    """Canonicalize raw rows with deterministic conflict tracking."""
    retained: list[AMPObservation] = []
    rejected: list[dict] = []
    conflicts: list[dict] = []
    seen_sequences: set[str] = set()
    label_map: dict[str, set[float]] = defaultdict(set)
    duplicates = 0
    forbidden = compliance_references or set()

    for idx, row in enumerate(rows):
        raw_sequence = str(row.get("sequence", ""))
        canonical, reason = clean_sequence(raw_sequence)
        if reason is not None:
            rejected.append(
                {"row_index": idx, "sequence": raw_sequence, "reason": reason}
            )
            continue
        assert canonical is not None

        if canonical in forbidden:
            rejected.append(
                {
                    "row_index": idx,
                    "sequence": canonical,
                    "reason": "compliance_reference_overlap",
                }
            )
            continue

        # Reject non-linear sequence records
        is_linear = _parse_bool(row.get("is_linear"), default=True)
        if not is_linear:
            rejected.append(
                {
                    "row_index": idx,
                    "sequence": canonical,
                    "reason": "modified_or_non_linear",
                }
            )
            continue

        mod_desc = (
            str(row.get("terminal_modification", ""))
            + " "
            + str(row.get("other_modification", ""))
        ).lower()
        if any(term in mod_desc for term in MODIFICATION_TERMS):
            rejected.append(
                {
                    "row_index": idx,
                    "sequence": canonical,
                    "reason": "modified_or_non_linear",
                }
            )
            continue

        if canonical in seen_sequences:
            duplicates += 1
        seen_sequences.add(canonical)

        is_amp = row.get("is_amp")
        if is_amp is not None and str(is_amp).strip():
            label_map[canonical].add(float(is_amp))

        seq_sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

        retained.append(
            AMPObservation(
                sequence=canonical,
                sequence_sha256=seq_sha,
                source=str(row.get("source", "unspecified")),
                source_record_id=str(row.get("source_record_id"))
                if row.get("source_record_id") is not None
                else None,
                is_amp=float(is_amp)
                if is_amp is not None and str(is_amp).strip()
                else None,
                organism=str(row.get("organism"))
                if row.get("organism") is not None
                else None,
                strain=str(row.get("strain"))
                if row.get("strain") is not None
                else None,
                panel_target=str(row.get("panel_target"))
                if row.get("panel_target") is not None
                else None,
                mic_value=float(row["mic_value"])
                if row.get("mic_value") is not None and str(row["mic_value"]).strip()
                else None,
                mic_unit_original=str(row.get("mic_unit_original"))
                if row.get("mic_unit_original") is not None
                else None,
                mic_uM=float(row["mic_uM"])
                if row.get("mic_uM") is not None and str(row["mic_uM"]).strip()
                else None,
                mic_censor=row.get("mic_censor"),  # type: ignore[arg-type]
                hc50_uM=float(row["hc50_uM"])
                if row.get("hc50_uM") is not None and str(row["hc50_uM"]).strip()
                else None,
                hc50_censor=row.get("hc50_censor"),  # type: ignore[arg-type]
                terminal_modification=str(row.get("terminal_modification"))
                if row.get("terminal_modification") is not None
                else None,
                other_modification=str(row.get("other_modification"))
                if row.get("other_modification") is not None
                else None,
                is_linear=True,
                label_provenance=row.get("label_provenance", "measured"),  # type: ignore[arg-type]
            )
        )

    for sequence, labels in label_map.items():
        if len(labels) > 1:
            conflicts.append(
                {
                    "sequence": sequence,
                    "labels": sorted(labels),
                    "kind": "contradictory_amp_labels",
                    "reason": "contradictory_amp_labels",
                }
            )

    return CleaningResult(
        observations=tuple(retained),
        rejected=tuple(rejected),
        conflicts=tuple(conflicts),
        duplicate_count=duplicates,
    )
