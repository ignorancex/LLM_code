# tests/test_atmosphere_extended.py

import pytest
import numpy as np
from multirex.spectra import Atmosphere

def test_atmosphere_initialization_with_ranges():
    # Test initialization with range values
    atm = Atmosphere(seed=42, 
                     temperature=(250, 350), 
                     base_pressure=(800, 1200), 
                     top_pressure=(50, 150),
                     composition={"H2O": (-4, -2), "CO2": -3}, 
                     fill_gas="N2")
    
    assert atm.validate() is True
    assert 250 <= atm.temperature <= 350
    assert 800 <= atm.base_pressure <= 1200
    assert 50 <= atm.top_pressure <= 150
    assert -4 <= atm.composition["H2O"] <= -2
    assert atm.composition["CO2"] == -3
    assert atm.fill_gas == "N2"

def test_atmosphere_with_multiple_fill_gases():
    # Test with a list of fill gases
    atm = Atmosphere(seed=42, temperature=300, base_pressure=1000, top_pressure=100,
                     composition={"H2O": -3, "CO2": -4}, fill_gas=["N2", "O2"])
    assert atm.validate() is True
    assert isinstance(atm.fill_gas, list)
    assert "N2" in atm.fill_gas
    assert "O2" in atm.fill_gas

def test_atmosphere_pressure_validation():
    # Test various pressure validation scenarios
    
    # Test with base pressure equal to top pressure
    with pytest.raises(ValueError):
        Atmosphere(seed=42, temperature=300, base_pressure=100, top_pressure=100,
                  composition={"H2O": -3}, fill_gas="N2")
    
    # Test with base pressure less than top pressure
    with pytest.raises(ValueError):
        Atmosphere(seed=42, temperature=300, base_pressure=50, top_pressure=100,
                  composition={"H2O": -3}, fill_gas="N2")
    
    # Test with negative pressures
    with pytest.raises(ValueError):
        Atmosphere(seed=42, temperature=300, base_pressure=-100, top_pressure=-50,
                  composition={"H2O": -3}, fill_gas="N2")

def test_atmosphere_temperature_validation():
    # Test temperature validation
    
    # Test with negative temperature
    with pytest.raises(ValueError):
        Atmosphere(seed=42, temperature=-300, base_pressure=1000, top_pressure=100,
                  composition={"H2O": -3}, fill_gas="N2")
    
    # Test with temperature range containing negative values
    with pytest.raises(ValueError):
        Atmosphere(seed=42, temperature=(-300, 300), base_pressure=1000, top_pressure=100,
                  composition={"H2O": -3}, fill_gas="N2")

def test_atmosphere_missing_attributes():
    # Test validation with missing attributes
    
    # Missing temperature
    atm = Atmosphere(seed=42, base_pressure=1000, top_pressure=100,
                    composition={"H2O": -3}, fill_gas="N2")
    assert atm.validate() is False
    
    # Missing base_pressure
    atm = Atmosphere(seed=42, temperature=300, top_pressure=100,
                    composition={"H2O": -3}, fill_gas="N2")
    assert atm.validate() is False
    
    # Missing top_pressure
    atm = Atmosphere(seed=42, temperature=300, base_pressure=1000,
                    composition={"H2O": -3}, fill_gas="N2")
    assert atm.validate() is False
    
    # Missing fill_gas
    atm = Atmosphere(seed=42, temperature=300, base_pressure=1000, top_pressure=100,
                    composition={"H2O": -3})
    assert atm.validate() is False

def test_atmosphere_reshuffle():
    # Test reshuffle functionality
    atm = Atmosphere(seed=42, 
                     temperature=(250, 350), 
                     base_pressure=(800, 1200), 
                     top_pressure=(50, 150),
                     composition={"H2O": (-4, -2)}, 
                     fill_gas="N2")
    
    original_temp = atm.temperature
    original_base_pressure = atm.base_pressure
    original_top_pressure = atm.top_pressure
    original_h2o = atm.composition["H2O"]
    
    # Force different random values by changing seed temporarily
    atm.set_seed(43)
    atm.reshuffle()
    
    # Values should be different after reshuffle but still within ranges
    assert atm.temperature != original_temp
    assert atm.base_pressure != original_base_pressure
    assert atm.top_pressure != original_top_pressure
    assert atm.composition["H2O"] != original_h2o
    
    # Values should still be within the original ranges
    assert 250 <= atm.temperature <= 350
    assert 800 <= atm.base_pressure <= 1200
    assert 50 <= atm.top_pressure <= 150
    assert -4 <= atm.composition["H2O"] <= -2