import torch

from amp_ctmc_2027.core import CONDITION_NAMES, ConditionVector


def test_conditional_unconditional_and_cfg_formula(tiny_model, encoder):
    tokens = torch.stack([encoder.encode("ACDEFGHI"), encoder.encode("KLMNPQRS")])
    time = torch.tensor([0.3, 0.7])
    values = torch.ones((2, len(CONDITION_NAMES)))
    observed = torch.ones_like(values, dtype=torch.bool)
    conditions = ConditionVector(values, observed)
    tiny_model.eval()
    with torch.inference_mode():
        conditional = tiny_model(tokens, time, conditions)
        unconditional = tiny_model(tokens, time)
        guided = tiny_model.guided_logits(tokens, time, conditions, 1.5)
    assert conditional.shape == (2, 51, encoder.vocab_size)
    torch.testing.assert_close(
        guided, unconditional + 1.5 * (conditional - unconditional)
    )


def test_unobserved_condition_values_do_not_affect_logits(tiny_model, encoder):
    tokens = encoder.encode("ACDEFGHI").unsqueeze(0)
    time = torch.tensor([0.4])
    values = torch.zeros((1, len(CONDITION_NAMES)))
    observed = torch.zeros_like(values, dtype=torch.bool)
    first = ConditionVector(values.clone(), observed)
    changed_values = values.clone()
    changed_values[0, 0] = 1000
    changed = ConditionVector(changed_values, observed)
    tiny_model.eval()
    with torch.inference_mode():
        left = tiny_model(tokens, time, first)
        right = tiny_model(tokens, time, changed)
    torch.testing.assert_close(left, right)
