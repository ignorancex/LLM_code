"""Tests for cpp_structure.py"""
from collections import defaultdict
from external.bazel_python.pytest_helper import main
from ts_lib import TripletStructure
from runtime.cpp_structure import CPPStructure, CPPPattern

def test_simple_constraints():
    """Tests the CPPStructure class."""
    ts = TripletStructure()
    ts[":A"].map({ts[":B"]: ts[":C"]})
    ts_cpp = CPPStructure(ts)
    ts[":B"].map({ts[":C"]: ts[":A"]})
    ts[":B"].map({ts[":C"]: ts[":X"]})

    # Removed fact.
    ts[":B"].map({ts[":B"]: ts[":B"]})
    constraints = [(0, 0, 0)]
    assert list(ts_cpp.assignments(constraints)) == [dict({0: "/:B"})]
    ts.remove_fact(("/:B", "/:B", "/:B"))
    assert list(ts_cpp.assignments(constraints)) == []

    # Unquantified test.
    constraints = [("/:A", "/:B", "/:C"), ("/:B", "/:C","/:A")]
    assert list(ts_cpp.assignments(constraints)) == [dict({})]

    constraints = [("/:Wrong", "/:B", "/:C"), ("/:B", "/:C","/:A")]
    assert list(ts_cpp.assignments(constraints)) == []
    constraints = [("/:A", "/:B", "/:B")]
    assert list(ts_cpp.assignments(constraints)) == []

    # Maybe_equals test.
    constraints = [(5, "/:B", 6), (7, 0, 1), (7, 2, 3)]
    assert not list(ts_cpp.assignments(constraints))

    maybe_equal = defaultdict(set)
    for variable in set({5, 1}):
        maybe_equal[variable] = set({5, 1})
    for variable in set({6, 0, 2}):
        maybe_equal[variable] = set({6, 0, 2})
    truth = [dict({
        5: "/:A",
        6: "/:C",
        7: "/:B",
        0: "/:C",
        1: "/:A",
        2: "/:C",
        3: "/:X",
    })]
    assert list(ts_cpp.assignments(constraints, maybe_equal)) == truth
    # Test we can pull from the cache correctly.
    assert list(ts_cpp.assignments(constraints, maybe_equal)) == truth

main(__name__, __file__)
