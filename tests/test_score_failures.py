import numpy as np
import pytest

from amp_ctmc_2027.external_scorer import VendoredScorerClient


def test_external_scorer_failure_is_not_replaced_by_neutral(monkeypatch, tmp_path):
    client = VendoredScorerClient(tmp_path)

    def fail(_payload):
        raise RuntimeError("offline")

    monkeypatch.setattr(client, "_invoke", fail)
    with pytest.raises(RuntimeError, match="offline"):
        client.score(["ACDEFGHI"])


def test_external_scorer_rejects_malformed_output(monkeypatch, tmp_path):
    client = VendoredScorerClient(tmp_path, chunk_size=2)
    monkeypatch.setattr(
        client,
        "_invoke",
        lambda payload: {
            "activity": np.ones(1),
            "hemolysis": np.ones(1),
            "activity_hemolysis": np.ones(1),
            "esm2_pseudo_perplexity": np.ones(0),
        },
    )
    with pytest.raises(ValueError, match="Invalid scorer output"):
        client.score(["ACDEFGHI"])
