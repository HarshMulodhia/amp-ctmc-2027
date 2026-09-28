import pytest
import torch

from amp_ctmc_2027.config import AMPConfig
from amp_ctmc_2027.core import CTMCDenoiser
from amp_ctmc_2027.data.dataset import AMPCanvasEncoder


@pytest.mark.parametrize("length", [8, 49, 50])
def test_canvas_has_eos_after_all_supported_lengths(length):
    encoder = AMPCanvasEncoder()
    sequence = "A" * length
    canvas = encoder.encode(sequence)
    assert canvas.shape == (51,)
    assert canvas[length].item() == encoder.eos_idx
    assert torch.all(canvas[length + 1 :] == encoder.pad_idx)
    assert encoder.decode(canvas) == sequence


def test_legacy_token_metadata_is_rejected():
    with pytest.raises(ValueError, match="Legacy tokenizer"):
        AMPCanvasEncoder.from_metadata(
            {"max_length": 50, "vocab": "ACDEFGHIKLMNPQRSTVWY"}
        )


def test_legacy_model_checkpoint_is_rejected(tmp_path):
    encoder = AMPCanvasEncoder()
    config = AMPConfig(d_model=16, n_heads=4, n_layers=1, d_ff=32, device="cpu")
    path = tmp_path / "old.pt"
    torch.save({"old.weight": torch.zeros(1)}, path)
    with pytest.raises(ValueError, match="Unsupported legacy"):
        CTMCDenoiser.load(
            path, config, encoder.vocab_size, encoder.pad_idx, torch.device("cpu")
        )


def test_version_two_checkpoint_round_trip(tmp_path, tiny_model, encoder):
    path = tmp_path / "model.pt"
    tiny_model.save(path)
    restored = CTMCDenoiser.load(
        path,
        tiny_model.config,
        encoder.vocab_size,
        encoder.pad_idx,
        torch.device("cpu"),
    )
    for name, tensor in tiny_model.state_dict().items():
        torch.testing.assert_close(tensor, restored.state_dict()[name])
