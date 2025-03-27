"""Tests for ts_lib.py"""
# pylint: disable=pointless-statement,import-error
import itertools
from external.bazel_python.pytest_helper import main
from ts_lib import TripletStructure

def test_get_create_nodes():
    """Tests that ts.__getitem__ works correctly."""
    ts = TripletStructure()
    ts[":A"]
    assert ts.has_node("/:A")
    ts[":B, :C"]
    assert ts.has_node("/:B") and ts.has_node("/:C")
    b0, b1 = ts[":B:??, :B:??"]
    assert str(b0) == "/:B:0"
    assert str(b1) == "/:B:1"
    assert ts.has_node("/:B:0") and ts.has_node("/:B:1")
    # Should be lexicographical.
    assert b0 < b1

    assert b0.display_name() == "/:B:0"
    b0.display_name("b0")
    assert b0.display_name() == "b0"

def test_str():
    """Tests TripletStructure.__str__"""
    ts = TripletStructure()
    ts[":A"].map({ts[":B"]: ts[":C"]})
    truth = f"TripletStructure ({id(ts)}):\n\t('/:A', '/:B', '/:C')"
    assert str(ts) == truth

def test_shadow():
    """Tests that we can shadow operations on the structure."""
    # pylint: disable=missing-function-docstring
    class ShadowStructure():
        """Dummy shadower."""
        def __init__(self):
            self.log = []
        def add_node(self, node):
            self.log.append(("+", node))
        def remove_node(self, node):
            self.log.append(("-", node))
        def add_fact(self, fact):
            self.log.append(("+", fact))
        def remove_fact(self, fact):
            self.log.append(("-", fact))
    ts = TripletStructure()
    ts.shadow = ShadowStructure()
    ts[":A"].map({ts[":B"]: ts[":C"]})
    ts[":C"].remove_with_facts()
    # Shouldn't re-remove it.
    ts.remove_fact(("/:A", "/:B", "/:C"))
    assert len(ts.shadow.log) == 6
    assert (set(ts.shadow.log[:3])
            == set({("+", "/:A"), ("+", "/:B"), ("+", "/:C")}))
    assert ts.shadow.log[3] == ("+", ("/:A", "/:B", "/:C"))
    assert ts.shadow.log[4] == ("-", ("/:A", "/:B", "/:C"))
    assert ts.shadow.log[5] == ("-", "/:C")

def test_scope():
    """Tests that scopes work correctly."""
    ts = TripletStructure()
    scope = ts.scope("/:Scope")
    assert str(scope[":A"]) == "/:Scope:A"
    assert ts.has_node("/:Scope:A")
    assert scope.protected()[":A"] == "/:Scope:A"
    sub_scope = scope.scope(":Sub")
    assert str(sub_scope[":A"]) == "/:Scope:Sub:A"
    assert str(sub_scope["/:A"]) == "/:A"
    assert list(sub_scope) == [sub_scope[":A"]]
    assert scope[":Sub:A"] in sub_scope
    assert scope[":A"] not in sub_scope
    assert sub_scope[":A"] - scope == ":Sub:A"
    assert sub_scope[":A"] - ts.scope(":Hello") == "/:Scope:Sub:A"
    with ts.scope(":Scope"):
        assert ts.scope().prefix == "/:Scope"
        with ts.scope(":Sub"):
            assert ts.scope().prefix == "/:Scope:Sub"
            assert ts[":A"] == sub_scope[":A"]
        assert ts.scope().prefix == "/:Scope"
        assert ts[":Sub:A"] == sub_scope[":A"]

def test_freeze_frame():
    """Tests TSFreezeFrame."""
    ts = TripletStructure()
    ts[":A"].map({ts[":B"]: ts[":C"]})
    freeze_frame = ts.freeze_frame()
    assert freeze_frame.nodes == set({"/:A", "/:B", "/:C"})
    assert freeze_frame.facts == set({("/:A", "/:B", "/:C")})
    ts[":D"].map({ts[":E"]: ts[":B"]})
    ts[":C"].remove_with_facts()

    delta = freeze_frame.delta_to_reach(ts.freeze_frame())
    assert delta.add_nodes == set({"/:D", "/:E"})
    assert delta.add_facts == set({("/:D", "/:E", "/:B")})
    assert delta.remove_nodes == set({"/:C"})
    assert delta.remove_facts == set({("/:A", "/:B", "/:C")})

    ts.commit()
    ts.freeze_frame().delta_to_reach(freeze_frame).apply()
    assert freeze_frame == ts.freeze_frame()

def test_delta():
    """Tests TSDelta and TSRecorder."""
    ts = TripletStructure()
    assert not ts.buffer and ts.is_clean()
    ts.commit(commit_if_clean=False)
    assert ts.path == [None]
    ts.commit(commit_if_clean=True)
    assert not ts.path[1]

    ts[":A"].map({ts[":B"]: [ts[":C"], ts[":D"]]})
    assert ts.buffer and not ts.is_clean()
    assert ts.buffer.add_nodes == set({"/:A", "/:B", "/:C", "/:D"})
    assert ts.buffer.add_facts == set({("/:A", "/:B", "/:C"),
                                       ("/:A", "/:B", "/:D")})
    assert ts.buffer.remove_nodes == ts.buffer.remove_facts == set()
    ts.commit()
    assert ts.path[2]
    assert not ts.buffer
    assert ts.is_clean()

    recording = ts.start_recording()
    before = ts.freeze_frame()
    ts.rollback(0)
    assert before == ts.freeze_frame()

    ts[":D"].remove_with_facts()
    remove_D = ts.commit()
    assert recording.commits() == [remove_D]
    ts.rollback(-1)
    assert recording.commits() == []
    assert before == ts.freeze_frame()

    ts[":C"].remove_with_facts()
    remove_C = ts.commit()
    assert recording.commits() == [remove_C]
    ts.rollback(len(ts.path) - 1)
    assert recording.commits() == []
    assert before == ts.freeze_frame()

    ts[":A"].remove_with_facts()
    remove_A = ts.commit()
    assert recording.commits(rollback=True) == [remove_A]
    assert recording.commits() == []
    assert before == ts.freeze_frame()

    ts[":A"].map({ts[":D"]: ts[":E"]})
    ts[":C"].remove_with_facts()
    truth = f"TSDelta ({id(ts.buffer)}):"
    truth += "\n\t- Nodes: \n\t\t/:C"
    truth += "\n\t+ Nodes: \n\t\t/:E"
    truth += "\n\t- Facts: \n\t\t('/:A', '/:B', '/:C')"
    truth += "\n\t+ Facts: \n\t\t('/:A', '/:D', '/:E')"
    assert str(ts.buffer) == truth

def test_remove_node_without_facts():
    """Tests that a node cannot be removed while it has linked facts.

    Adapted from Zhe's example on PR #421.
    """
    ts = TripletStructure()
    ts[":A"].map({ts[":B"]: ts[":C"], ts[":C"]: ts[":B"]})
    for full_name in ("/:A", "/:B", "/:C"):
        try:
            ts.remove_node(full_name)
        except AssertionError:
            pass
        else:
            assert False, "TripletStructure let me remove a useful node."
        try:
            ts[full_name].remove()
        except AssertionError:
            pass
        else:
            assert False, "TripletStructure let me remove a useful node."

def test_remove_node_with_facts():
    """Tests that we can successfully remove a node with all of its facts.

    Adapted from Zhe's example on PR #421.
    """
    ts = TripletStructure()
    node_names = ("/:A", "/:B", "/:C")

    def run_for_node(ts, node_name):
        ts[":A"].map({ts[":B"]: ts[":C"], ts[":C"]: ts[":B"]})
        assert any(fact_list for fact_list in ts.facts.values())
        assert all(ts.has_node(full_name) for full_name in node_names)
        ts[node_name].remove_with_facts()

        assert not any(fact_list for fact_list in ts.facts.values())
        assert not ts.has_node(node_name)

    for node_name in node_names:
        run_for_node(ts, node_name)

def test_fact_invariants():
    """Tests that some desired invariants always hold."""
    ts = TripletStructure()
    # Try adding nodes
    node_names = ["/:{}".format(i) for i in range(1000)]
    for node_name, next_node_name in zip(node_names, node_names[1:]):
        ts[node_name]
        assert ts.has_node(node_name)
        assert not ts.has_node(next_node_name)
    assert not ts.has_node(node_names[-1])
    ts.add_node(node_names[-1])
    assert ts.has_node(node_names[-1])

    # Try bulk removing/adding nodes.
    ts.remove_nodes(node_names)
    assert not any(ts.has_node(name) for name in node_names)
    ts.add_nodes(node_names)
    assert all(ts.has_node(name) for name in node_names)

    # Try adding/removing facts and ensure the invariants hold.
    def assert_fact_invariant(ts, fact):
        if ts.facts[fact]:
            relevant_keys = list(ts._iter_subfacts(fact))
        else:
            relevant_keys = []
        for key in ts.facts.keys():
            assert (fact in ts.facts[key]) == (key in relevant_keys)

    fact = (node_names[0], node_names[1], node_names[2])
    ts.add_fact(fact)
    for maybe_fact in itertools.permutations(fact):
        assert_fact_invariant(ts, maybe_fact)
    for maybe_fact in itertools.permutations(node_names[155:160], 3):
        assert_fact_invariant(ts, maybe_fact)
        ts.add_fact(maybe_fact)
        assert_fact_invariant(ts, maybe_fact)
        ts.remove_fact(maybe_fact)
        assert_fact_invariant(ts, maybe_fact)

    # Bulk adding/removing facts.
    ts.add_facts(list(itertools.permutations(fact)))
    assert all(ts.lookup(*fact) for fact in itertools.permutations(fact))
    ts.remove_facts(list(itertools.permutations(fact)))
    assert not any(ts.lookup(*fact) for fact in itertools.permutations(fact))

main(__name__, __file__)
