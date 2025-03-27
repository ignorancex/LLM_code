"""Integration test using turing_machine.py"""
# pylint: disable=pointless-statement,import-error
from external.bazel_python.pytest_helper import main
import turing_machine

def test_turing_machine():
    """Regression test for the Turing machine example."""
    ts = turing_machine.Main()
    state, tape, index = turing_machine.PrintTMState(ts)
    assert state == "/:State:B"
    assert tape == ["/:Symbol:1", "X"]
    assert index == 1

main(__name__, __file__)
