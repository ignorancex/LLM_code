import pytest
from multirex.spectra import Star

def test_star_creation():
    star = Star(seed=42, temperature=5800, radius=1, mass=1)
    assert star.temperature == 5800
    assert star.radius == 1
    assert star.mass == 1
