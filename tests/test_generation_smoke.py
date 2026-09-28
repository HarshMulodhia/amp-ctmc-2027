from amp_ctmc_2027.core import (
    ReverseGenerationConfig,
    SinSquaredSchedule,
    TauLeapingSampler,
)


def test_small_generation_obeys_sequence_constraints(tiny_model, encoder):
    sampler = TauLeapingSampler(tiny_model, encoder, SinSquaredSchedule(), min_length=8)
    generated = sampler.sample_batch(
        ReverseGenerationConfig(steps=6, temperature=0.9, seed=91), 4
    )
    assert len(generated) == 4
    assert all(8 <= len(sequence) <= 50 for sequence in generated)
    assert all(set(sequence).issubset(set(encoder.vocab)) for sequence in generated)
