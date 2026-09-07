from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _features(sequence: str) -> dict[str, float]:
    seq = sequence.strip().upper()
    length = max(len(seq), 1)
    charge = (seq.count("K") + seq.count("R") + 0.1 * seq.count("H") - seq.count("D") - seq.count("E")) / length
    hydrophobic = sum(seq.count(aa) for aa in "AVILMFWY") / length
    aromatic = sum(seq.count(aa) for aa in "FWY") / length
    composition = [seq.count(aa) / length for aa in "ACDEFGHIKLMNPQRSTVWY"]
    entropy = 0.0
    for p in composition:
        if p > 0:
            entropy += -p * math.log(p)
    return {
        "length": float(len(seq)),
        "charge": float(charge),
        "hydrophobic": float(hydrophobic),
        "aromatic": float(aromatic),
        "entropy": float(entropy),
    }


def _load_coefficients() -> dict[str, float]:
    coeff_path = Path(__file__).resolve().parents[2] / "models" / "pretrained_coefficients.json"
    return json.loads(coeff_path.read_text(encoding="utf-8"))


def _score_one(seq: str, coef: dict[str, float]) -> tuple[float, float, float]:
    feat = _features(seq)
    activity_logit = (
        coef["activity_bias"]
        + coef["activity_length"] * feat["length"]
        + coef["activity_charge"] * feat["charge"]
        + coef["activity_hydrophobicity"] * feat["hydrophobic"]
        + coef["activity_aromatic"] * feat["aromatic"]
    )
    hemolysis_logit = (
        coef["hemolysis_bias"]
        + coef["hemolysis_length"] * feat["length"]
        + coef["hemolysis_charge"] * feat["charge"]
        + coef["hemolysis_hydrophobicity"] * feat["hydrophobic"]
        + coef["hemolysis_aromatic"] * feat["aromatic"]
    )
    activity = _sigmoid(activity_logit)
    hemolysis = _sigmoid(hemolysis_logit)
    pseudo_perplexity = (
        coef["esm2_ppl_base"]
        + coef["esm2_ppl_length_scale"] * abs(feat["length"] - 24.0)
        + coef["esm2_ppl_composition_scale"] * abs(feat["entropy"] - 2.4)
    )
    return activity, hemolysis, pseudo_perplexity


def main() -> int:
    payload = json.loads(sys.stdin.read() or "{}")
    sequences = payload.get("sequences", [])
    coef = _load_coefficients()
    activity: list[float] = []
    hemolysis: list[float] = []
    pseudo_perplexity: list[float] = []
    combined: list[float] = []

    for sequence in sequences:
        a, h, ppl = _score_one(sequence, coef)
        activity.append(float(a))
        hemolysis.append(float(h))
        pseudo_perplexity.append(float(ppl))
        combined.append(float((0.7 * a) + (0.3 * (1.0 - h))))

    result = {
        "activity": activity,
        "hemolysis": hemolysis,
        "activity_hemolysis": combined,
        "esm2_pseudo_perplexity": pseudo_perplexity,
    }
    sys.stdout.write(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
