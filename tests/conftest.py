import pytest

from amp_ctmc_2027.config import AMPConfig
from amp_ctmc_2027.core import CTMCDenoiser
from amp_ctmc_2027.data.dataset import AMPCanvasEncoder


@pytest.fixture
def encoder():
    return AMPCanvasEncoder(max_length=50)


@pytest.fixture
def tiny_model():
    config = AMPConfig(
        d_model=16,
        n_heads=4,
        n_layers=1,
        d_ff=32,
        dropout=0.0,
        batch_size=2,
        n_epochs=1,
        device="cpu",
    )
    enc = AMPCanvasEncoder(max_length=50)
    return CTMCDenoiser(config, enc.vocab_size, enc.pad_idx)
