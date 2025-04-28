# tests/test_planet_atmosphere.py

import pytest
from multirex.spectra import Planet, Atmosphere

def test_planet_creation_without_atmosphere():
    # Create a planet without an atmosphere
    planet = Planet(seed=42, radius=1, mass=1)
    assert planet.radius == 1
    assert planet.mass == 1
    assert planet.atmosphere is None

def test_planet_with_atmosphere():
    # Create an atmosphere with fixed parameters
    atm = Atmosphere(seed=42, temperature=300, base_pressure=1000, top_pressure=100,
                     composition={"H2O": -3}, fill_gas="N2")
    planet = Planet(seed=42, radius=1, mass=1, atmosphere=atm)
    # Verify that the atmosphere is assigned and its parameters are correct
    assert planet.atmosphere.temperature == 300
    assert planet.validate() is True

def test_planet_setters():
    planet = Planet(seed=42, radius=1, mass=1)
    planet.set_radius((2, 2))
    planet.set_mass((2, 2))
    assert planet.radius == 2
    assert planet.mass == 2
