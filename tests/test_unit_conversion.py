import pytest

from amp_ctmc_2027.data.preprocess import concentration_to_um


def test_molar_unit_conversions():
    assert concentration_to_um(2, "mM") == 2000
    assert concentration_to_um(16, "uM") == 16
    assert concentration_to_um(500, "nM") == pytest.approx(0.5)


def test_mass_units_require_documented_molecular_weight():
    with pytest.raises(ValueError, match="molecular weight"):
        concentration_to_um(1, "mg/L")
    assert concentration_to_um(1, "mg/L", molecular_weight_g_mol=1000) == pytest.approx(
        1
    )
