import sys
import pytest
import minvime.generator as gen  # The file ../minvime/generator.py

def test_min_max_baseline():
    temp = gen.generate_min_max_baseline(10,100,1000)
    assert len(temp) == 1000
    assert min(temp) >= 10
    assert max(temp) <= 100

    temp = gen.generate_min_max_baseline(100,1000,1000)
    assert len(temp) == 1000
    assert min(temp) >= 100
    assert max(temp) <= 1000

    temp = gen.generate_min_max_baseline(-100,100,1000)
    assert len(temp) == 1000
    assert min(temp) >= -100
    assert max(temp) <= 100

    temp = gen.generate_min_max_baseline(-200,500,5000)
    assert len(temp) == 5000
    assert min(temp) >= -200
    assert max(temp) <= 500

