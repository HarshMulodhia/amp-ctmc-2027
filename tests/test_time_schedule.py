from amp_ctmc_2027.core import SinSquaredSchedule


def test_time_orientation_noise_to_clean():
    schedule = SinSquaredSchedule()
    assert schedule.kappa(0.0) == 0.0
    assert schedule.kappa(1.0) == 1.0
    assert schedule.kappa(0.25) < schedule.kappa(0.75)
