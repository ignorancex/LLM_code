import numpy as np
import pytest
from multirex.spectra import Planet, Atmosphere, Star, System, Physics

def create_sample_system():
    # Helper function to create a system with fixed parameters
    atm = Atmosphere(seed=42, temperature=300, base_pressure=1000, top_pressure=100,
                     composition={"H2O": -3}, fill_gas="N2")
    planet = Planet(seed=42, radius=1, mass=1, atmosphere=atm)
    star = Star(seed=42, temperature=5800, radius=1, mass=1)
    system = System(planet, star, seed=42, sma=1.0)
    return system

def test_system_make_tm():
    system = create_sample_system()
    # Call make_tm() and verify that the transmission model is generated
    system.make_tm()
    assert system.transmission is not None

def test_generate_spectrum():
    system = create_sample_system()
    system.make_tm()
    wn_grid = Physics.wavenumber_grid(1, 10, 100)
    bin_wn, bin_rprs = system.generate_spectrum(wn_grid)
    assert isinstance(bin_wn, np.ndarray)
    assert isinstance(bin_rprs, np.ndarray)
