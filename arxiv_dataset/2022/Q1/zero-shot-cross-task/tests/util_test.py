import pytest

from src.common import util


@pytest.mark.parametrize("name,expected", [
    ["\u2026What could happen next, C1 or C2?", "What_could_happen_next_C1_or_C2"],
    ["C1 or C2? premise, so/because\u2026", "C1_or_C2_premise_so_because"]
])
def test_sanitize_name(name, expected):
    assert util.sanitize_name(name) == expected
