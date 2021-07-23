# %%
from src.utils.normalization.EnergyNormalizer import EnergyNormalizer
import numpy as np
import pandas as pd
import pytest

@pytest.fixture()
def resource():
    # Setup -> create random time series
    sig1 = (1, 1, -1, 1, -1)  # energy 5
    sig2 = (1, 2, 3, 4, 5) # energy 55
    data = np.array([sig1, sig2]).T
    df = pd.DataFrame(data)
    normalizer = EnergyNormalizer(df)
    yield normalizer
    # Teardown

class TestEnergyNormalizer(object):

    def test_calculated_energies_equality_correct(self, resource):
        real = resource.calculate_energies_()
        expected = [5, 55]
        # check equality
        np.testing.assert_equal(real, expected)

    def test_normalized_energies_are_one(self, resource):
        resource.normalize()
        real = resource.calculate_energies_()
        expected = [1, 1]
        assert np.all(np.isclose(real, expected))

    def test_calculate_energies_equality_incorrect(self, resource):
        real = resource.calculate_energies_()
        expected = [12, 57]
        # check non-equality
        np.testing.assert_raises(AssertionError,
                                 np.testing.assert_equal,
                                 real, expected)

    def test_normalize_energies_are_ones(self, resource):
        resource.set_energies()
        resource.normalize()
        real = resource.calculate_energies_()
        expected = (1, 1)
        np.testing.assert_array_almost_equal(real, expected)

    def test_normalize_energy_first_is_one_second_is_not_one(self, resource):
        resource.set_energies()
        resource.normalize()
        resource.data.iloc[2,1] *= 1.1
        real = resource.calculate_energies_()
        expected = (1, 1)
        assert (np.isclose(real[0], expected[0])
                and
                not(np.isclose(real[1], expected[1]))
                )
