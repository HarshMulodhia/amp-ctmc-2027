from types import SimpleNamespace

import torch
from torch import nn

from amp_ctmc_2027.models.esm_multitask import ESMMultiTaskPredictor


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(bos_token_id=1, eos_token_id=2)
        self.embedding = nn.Embedding(16, 12)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def test_output_shapes_and_missing_labels_have_zero_loss_gradient():
    model = ESMMultiTaskPredictor(TinyBackbone(), hidden_size=12, strain_count=2)
    ids = torch.tensor([[1, 4, 5, 2, 0], [1, 6, 2, 0, 0]])
    mask = ids.ne(0)
    output = model(ids, mask, torch.tensor([0, 2]))
    assert output.pooled.shape == (2, 12)
    assert output.amp_probability.shape == (2,)
    empty = {
        "amp": torch.zeros(2),
        "amp_mask": torch.zeros(2, dtype=torch.bool),
        "hemolysis": torch.zeros(2),
        "hemolysis_mask": torch.zeros(2, dtype=torch.bool),
        "mic": torch.zeros(2),
        "mic_mask": torch.zeros(2, dtype=torch.bool),
        "hc50": torch.zeros(2),
        "hc50_mask": torch.zeros(2, dtype=torch.bool),
        "mic_censor": torch.zeros(2, dtype=torch.long),
        "hc50_censor": torch.zeros(2, dtype=torch.long),
    }
    losses = model.multitask_loss(output, empty)
    losses["total"].backward()
    assert losses["total"].item() == 0
    assert model.amp_head.weight.grad is not None
    assert torch.count_nonzero(model.amp_head.weight.grad) == 0
