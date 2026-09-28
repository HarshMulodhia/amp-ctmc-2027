from amp_ctmc_2027.core import (
    ReverseGenerationConfig,
    SinSquaredSchedule,
    TauLeapingSampler,
)


def test_same_seed_repeats_small_generation(tiny_model, encoder):
    sampler = TauLeapingSampler(tiny_model, encoder, SinSquaredSchedule(), min_length=8)
    config = ReverseGenerationConfig(steps=4, temperature=1.0, seed=27)
    first = sampler.sample_batch(config, 3)
    second = sampler.sample_batch(config, 3)
    assert first == second
    assert all(8 <= len(seq) <= 50 for seq in first)


def test_different_seed_changes_small_generation(tiny_model, encoder):
    sampler = TauLeapingSampler(tiny_model, encoder, SinSquaredSchedule(), min_length=8)
    one = sampler.sample_batch(
        ReverseGenerationConfig(steps=4, temperature=1.0, seed=27), 3
    )
    two = sampler.sample_batch(
        ReverseGenerationConfig(steps=4, temperature=1.0, seed=28), 3
    )
    assert one != two
