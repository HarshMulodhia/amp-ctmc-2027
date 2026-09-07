from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class VendoredScorerClient:
    """APEX-style subprocess scorer wrapper with isolated dependencies."""

    def __init__(self, project_dir: Path, timeout_sec: int = 600) -> None:
        self.project_dir = project_dir
        self.timeout_sec = timeout_sec
        self._cache: dict[tuple[str, ...], dict[str, np.ndarray]] = {}

    def score(self, sequences: list[str]) -> dict[str, np.ndarray]:
        if not sequences:
            empty = np.array([], dtype=np.float32)
            return {
                "activity": empty,
                "hemolysis": empty,
                "activity_hemolysis": empty,
                "esm2_pseudo_perplexity": empty,
            }
        key = tuple(sequences)
        if key in self._cache:
            return self._cache[key]

        payload = {"sequences": sequences}
        try:
            result = self._invoke(payload)
        except Exception as exc:  # pragma: no cover - depends on runtime env/setup
            logger.warning("Vendored scorer failed (%s); falling back to neutral scores", exc)
            neutral = np.full(len(sequences), 0.5, dtype=np.float32)
            high_ppl = np.full(len(sequences), 100.0, dtype=np.float32)
            output = {
                "activity": neutral,
                "hemolysis": neutral,
                "activity_hemolysis": neutral,
                "esm2_pseudo_perplexity": high_ppl,
            }
            self._cache[key] = output
            return output

        output = {
            "activity": np.asarray(result["activity"], dtype=np.float32),
            "hemolysis": np.asarray(result["hemolysis"], dtype=np.float32),
            "activity_hemolysis": np.asarray(result["activity_hemolysis"], dtype=np.float32),
            "esm2_pseudo_perplexity": np.asarray(result["esm2_pseudo_perplexity"], dtype=np.float32),
        }
        self._cache[key] = output
        return output

    def _invoke(self, payload: dict[str, list[str]]) -> dict:
        commands = [
            [
                "uv",
                "run",
                "--project",
                str(self.project_dir),
                sys.executable,
                "-m",
                "amp_scorer.score",
            ],
            [
                sys.executable,
                "-m",
                "amp_scorer.score",
            ],
        ]
        last_exc: Exception | None = None
        for command in commands:
            try:
                pythonpath = str(self.project_dir / "src")
                if os.environ.get("PYTHONPATH"):
                    pythonpath = f"{pythonpath}:{os.environ['PYTHONPATH']}"
                completed = subprocess.run(
                    command,
                    check=True,
                    text=True,
                    capture_output=True,
                    input=json.dumps(payload),
                    timeout=self.timeout_sec,
                    cwd=str(self.project_dir),
                    env={**os.environ, "PYTHONPATH": pythonpath},
                )
                return json.loads(completed.stdout)
            except Exception as exc:  # pragma: no cover - runtime dependent
                last_exc = exc
                continue
        raise RuntimeError(f"Unable to run vendored scorer subprocess: {last_exc}")
