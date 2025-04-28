import pytest
from multirex.spectra import Atmosphere

def test_atmosphere_validation():
    # Instantiate an atmosphere with valid parameters
    atm = Atmosphere(seed=42, temperature=300, base_pressure=1000, top_pressure=100,
                     composition={"H2O": -3}, fill_gas="N2")
    assert atm.validate() is True

def test_invalid_pressure():
    # Base pressure lower than top pressure should raise an error
    with pytest.raises(ValueError):
         Atmosphere(seed=42, temperature=300, base_pressure=(100, 100), top_pressure=(1000, 1000),
                  composition={"H2O": -3}, fill_gas="N2")
