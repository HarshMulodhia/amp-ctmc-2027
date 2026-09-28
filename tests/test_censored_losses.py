import pytest
import torch

from amp_ctmc_2027.models.esm_multitask import censored_gaussian_nll


def loss(mean, censor, target, upper=None):
    return censored_gaussian_nll(
        torch.tensor([mean]),
        torch.tensor([0.0]),
        torch.tensor([target]),
        torch.tensor([censor]),
        torch.tensor([True]),
        None if upper is None else torch.tensor([upper]),
    )


def test_right_censored_observation_rewards_larger_mean():
    assert loss(3, 1, 2) < loss(0, 1, 2)


def test_left_censored_observation_rewards_smaller_mean():
    assert loss(0, -1, 2) < loss(3, -1, 2)


def test_interval_censored_observation_rewards_mean_inside_interval():
    inside = loss(1.5, 2, 1, 2)
    outside = loss(0, 2, 1, 2)
    assert inside < outside


def test_interval_requires_upper_bound():
    with pytest.raises(ValueError, match="upper-bound"):
        loss(1, 2, 1)
