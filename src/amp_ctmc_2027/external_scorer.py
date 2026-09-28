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

    def __init__(
        self, project_dir: Path, timeout_sec: int = 600, chunk_size: int = 256
    ) -> None:
        self.project_dir = project_dir
        self.timeout_sec = timeout_sec
        self.chunk_size = chunk_size
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
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        key = tuple(sequences)
        if key in self._cache:
            return self._cache[key]

        chunks: dict[str, list[np.ndarray]] = {}
        expected = {
            "activity",
            "hemolysis",
            "activity_hemolysis",
            "esm2_pseudo_perplexity",
        }
        for start in range(0, len(sequences), self.chunk_size):
            chunk = sequences[start : start + self.chunk_size]
            # A configured, weighted scorer is mandatory: failure must not alter rank silently.
            result = self._invoke({"sequences": chunk})
            if not expected.issubset(result):
                raise ValueError(
                    f"Scorer response missing fields: {sorted(expected - set(result))}"
                )
            for name in expected:
                values = np.asarray(result[name], dtype=np.float32)
                if values.shape != (len(chunk),) or not np.isfinite(values).all():
                    raise ValueError(
                        f"Invalid scorer output {name!r}: expected {len(chunk)} finite values"
                    )
                chunks.setdefault(name, []).append(values)
        output = {name: np.concatenate(parts) for name, parts in chunks.items()}
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
            except (
                OSError,
                subprocess.SubprocessError,
                json.JSONDecodeError,
                ValueError,
            ) as exc:  # pragma: no cover
                last_exc = exc
                continue
        raise RuntimeError(f"Unable to run vendored scorer subprocess: {last_exc}")
