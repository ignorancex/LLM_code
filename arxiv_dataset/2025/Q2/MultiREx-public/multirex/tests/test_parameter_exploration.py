import numpy as np
import pandas as pd
import pytest
from unittest import mock
from multirex.spectra import Planet, Atmosphere, Star, System, Physics


def test_parameter_space_exploration_single_parameter():
    """Test parameter space exploration with a single parameter."""
    # Create a system for testing
    atm = Atmosphere(seed=42, temperature=300, base_pressure=1000, top_pressure=100,
                     composition={"H2O": -3}, fill_gas="N2")
    planet = Planet(seed=42, radius=1, mass=1, atmosphere=atm)
    star = Star(seed=42, temperature=5800, radius=1, mass=1)
    system = System(planet, star, seed=42, sma=1.0)
    
    # Create the transmission model
    system.make_tm()
    
    # Create a wavenumber grid for testing
    wn_grid = Physics.wavenumber_grid(1, 10, 100)
    
    # Define a parameter space with a single parameter (H2O concentration)
    parameter_space = {
        'planet.atmosphere.composition.H2O': {
            'min': -10,
            'max': -3,
            'n': 4,
            'distribution': 'linear'
        }
    }
    
    # Explore the parameter space
    results = system.explore_parameter_space(
        wn_grid=wn_grid,
        parameter_space=parameter_space,
        snr=20,
        n_observations=1,
        header = True
    )
    
    # Verify the results
    assert 'spectra' in results
    assert 'observations' in results
    
    # Check the spectra DataFrame
    spectra = results['spectra']
    assert isinstance(spectra.data, pd.DataFrame)
    assert isinstance(spectra.params, pd.DataFrame)
    
    # Verify we have the expected number of spectra (4 in this case)
    assert len(spectra.data) == 4
    assert len(spectra.params) == 4
    
    # Verify the parameter values in the params DataFrame
    # The parameter values are stored in the params DataFrame with the full parameter path
    h2o_values = spectra.params['atm H2O'].values
    assert len(h2o_values) == 4
    assert min(h2o_values) >= -10
    assert max(h2o_values) <= -3


def test_parameter_space_exploration_multiple_parameters():
    """Test parameter space exploration with multiple parameters."""
    # Create a system for testing
    atm = Atmosphere(seed=42, temperature=300, base_pressure=1000, top_pressure=100,
                     composition={"H2O": -3}, fill_gas="N2")
    planet = Planet(seed=42, radius=1, mass=1, atmosphere=atm)
    star = Star(seed=42, temperature=5800, radius=1, mass=1)
    system = System(planet, star, seed=42, sma=1.0)
    
    # Create the transmission model
    system.make_tm()
    
    # Create a wavenumber grid for testing
    wn_grid = Physics.wavenumber_grid(1, 10, 100)
    
    # Define a parameter space with multiple parameters
    parameter_space = {
        'planet.atmosphere.temperature': {
            'min': 200,
            'max': 400,
            'n': 3,
            'distribution': 'linear'
        },
        'planet.atmosphere.composition.CO2': {
            'min': -6,
            'max': -2,
            'n': 2,
            'distribution': 'linear'
        }
    }
    
    # Explore the parameter space
    results = system.explore_parameter_space(
        wn_grid=wn_grid,
        parameter_space=parameter_space,
        snr=20,
        n_observations=1,
        header = True
    )
    
    # Verify the results
    assert 'spectra' in results
    assert 'observations' in results
    
    # Check the spectra DataFrame
    spectra = results['spectra']
    assert isinstance(spectra.data, pd.DataFrame)
    assert isinstance(spectra.params, pd.DataFrame)
    
    # Verify we have the expected number of spectra (3 temperatures × 2 CO2 values = 6)
    assert len(spectra.data) == 6
    assert len(spectra.params) == 6
    
    # Verify the parameter values in the params DataFrame
    temp_values = sorted(set([params['atm temperature'] for _, params in spectra.params.iterrows()]))
    co2_values = sorted(set([params['atm CO2'] for _, params in spectra.params.iterrows()]))
    
    assert len(temp_values) == 3
    assert len(co2_values) == 2
    assert min(temp_values) >= 200
    assert max(temp_values) <= 400
    assert min(co2_values) >= -6
    assert max(co2_values) <= -2


def test_parameter_space_exploration_log_distribution():
    """Test parameter space exploration with logarithmic distribution."""
    # Create a system for testing
    atm = Atmosphere(seed=42, temperature=300, base_pressure=1000, top_pressure=100,
                     composition={"H2O": -3}, fill_gas="N2")
    planet = Planet(seed=42, radius=1, mass=1, atmosphere=atm)
    star = Star(seed=42, temperature=5800, radius=1, mass=1)
    system = System(planet, star, seed=42, sma=1.0)
    
    # Create the transmission model
    system.make_tm()
    
    # Create a wavenumber grid for testing
    wn_grid = Physics.wavenumber_grid(1, 10, 100)
    
    # Define a parameter space with logarithmic distribution
    parameter_space = {
        'planet.atmosphere.base_pressure': {
            'min': 1e5,
            'max': 1e8,
            'n': 3,
            'distribution': 'log'
        }
    }
    
    # Explore the parameter space
    results = system.explore_parameter_space(
        wn_grid=wn_grid,
        parameter_space=parameter_space,
        snr=20,
        n_observations=1,
        header=True
    )
    
    # Verify the results
    assert 'spectra' in results
    
    # Check the spectra DataFrame
    spectra = results['spectra']
    
    # Verify we have the expected number of spectra
    assert len(spectra.data) == 3
    
    # Verify the parameter values follow a logarithmic distribution
    pressure_values = [params['atm base_pressure'] for _, params in spectra.params.iterrows()]
    assert len(pressure_values) == 3
    
    # In a logarithmic distribution, the ratio between consecutive values should be approximately constant
    # Sort the values to ensure they're in order
    pressure_values.sort()
    
    # Calculate ratios between consecutive values
    ratios = [pressure_values[i+1] / pressure_values[i] for i in range(len(pressure_values)-1)]
    
    # Check that the ratios are approximately equal (within some tolerance)
    assert abs(ratios[0] - ratios[-1]) < 0.1


def test_parameter_space_exploration_with_discrete_values():
    """Test parameter space exploration with discrete parameter values."""
    # Create a system for testing
    atm = Atmosphere(seed=42, temperature=300, base_pressure=1000, top_pressure=100,
                     composition={"H2O": -3}, fill_gas="N2")
    planet = Planet(seed=42, radius=1, mass=1, atmosphere=atm)
    star = Star(seed=42, temperature=5800, radius=1, mass=1)
    system = System(planet, star, seed=42, sma=1.0)
    
    # Create the transmission model
    system.make_tm()
    
    # Create a wavenumber grid for testing
    wn_grid = Physics.wavenumber_grid(1, 10, 100)
    
    # Define a parameter space with discrete values
    parameter_space = {
        'planet.atmosphere.temperature': [200, 300, 400],
        'planet.radius': [0.8, 1.0, 1.2]
    }
    
    # Explore the parameter space
    results = system.explore_parameter_space(
        wn_grid=wn_grid,
        parameter_space=parameter_space,
        snr=20,
        n_observations=1,
        header=True
    )
    
    # Verify the results
    assert 'spectra' in results
    
    # Check the spectra DataFrame
    spectra = results['spectra']
    
    # Verify we have the expected number of spectra (3 temperatures × 3 radii = 9)
    assert len(spectra.data) == 9
    
    # Verify the parameter values match our discrete values
    temp_values = sorted(set([params['atm temperature'] for _, params in spectra.params.iterrows()]))
    radius_values = sorted(set([params['p_radius'] for _, params in spectra.params.iterrows()]))
    
    assert temp_values == [200, 300, 400]
    assert radius_values == [0.8, 1.0, 1.2]


def test_parameter_space_exploration_observations():
    """Test parameter space exploration with multiple observations."""
    # Create a system for testing
    atm = Atmosphere(seed=42, temperature=300, base_pressure=1000, top_pressure=100,
                     composition={"H2O": -3}, fill_gas="N2")
    planet = Planet(seed=42, radius=1, mass=1, atmosphere=atm)
    star = Star(seed=42, temperature=5800, radius=1, mass=1)
    system = System(planet, star, seed=42, sma=1.0)
    
    # Create the transmission model
    system.make_tm()
    
    # Create a wavenumber grid for testing
    wn_grid = Physics.wavenumber_grid(1, 10, 100)
    
    # Define a simple parameter space
    parameter_space = {
        'planet.atmosphere.temperature': [200, 300]
    }
    
    # Explore the parameter space with multiple observations per parameter combination
    results = system.explore_parameter_space(
        wn_grid=wn_grid,
        parameter_space=parameter_space,
        snr=20,
        n_observations=3,
        header=True
    )
    
    # Verify the results
    assert 'observations' in results
    
    # Check the observations DataFrame
    observations = results['observations']
    
    # Verify we have the expected number of observations (2 temperatures × 3 observations = 6)
    assert len(observations.data) == 6
    
    # Verify the SNR column exists
    assert 'SNR' in observations.params.columns
    
    # Verify the noise column exists
    assert 'noise' in observations.params.columns