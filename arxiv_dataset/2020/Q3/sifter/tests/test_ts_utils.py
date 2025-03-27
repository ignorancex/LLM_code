"""Tests for ts_utils.py"""
# pylint: disable=pointless-statement,import-error
from external.bazel_python.pytest_helper import main
from ts_lib import TripletStructure
from ts_utils import RegisterRule, RegisterPrototype, AssertNodesEqual

def test_register_rule():
    """Tests the RegisterRule(...) macro."""
    ts = TripletStructure()
    with ts.scope(":Rule"):
        with ts.scope(":MustMap") as exist:
            ts[":A"].map({ts[":B"]: ts[":C"]})
        with ts.scope(":Insert"):
            ts[":D"].map({exist[":A"]: ts["/:X"]})
        with ts.scope(":Hello"):
            ts[":E"].map({exist[":A"]: exist["/:B"]})
        ts.commit()
        RegisterRule(ts)
    # The 3 facts above + 5 below
    assert len(ts.lookup(None, None, None)) == 8
    assert ts.lookup("/:Rule:RuleMap:0", "/:Rule:_", "/RULE")
    assert ts.lookup("/:Rule:RuleMap:0", "/:Rule:MustMap:A", "/MUST_MAP")
    assert ts.lookup("/:Rule:RuleMap:0", "/:Rule:MustMap:B", "/MUST_MAP")
    assert ts.lookup("/:Rule:RuleMap:0", "/:Rule:MustMap:C", "/MUST_MAP")
    assert ts.lookup("/:Rule:RuleMap:0", "/:Rule:Insert:D", "/INSERT")
    freeze_frame = ts.freeze_frame()

    # Now test custom_qualifiers argument.
    ts.rollback(0) # Before registering the rule.
    with ts.scope(":Rule"):
        RegisterRule(ts, custom_qualifiers=dict({":Hello": "/INSERT"}))
    delta = ts.freeze_frame() - freeze_frame
    assert not (delta.add_nodes or delta.remove_nodes or delta.remove_facts)
    assert (delta.add_facts
            == set({("/:Rule:RuleMap:0", "/:Rule:Hello:E", "/INSERT")}))

    # Now test auto_assert_equal
    ts.rollback(0)
    with ts.scope(":Rule"):
        with ts.scope(":MustMap") as exist:
            ts[":A"].map({ts[":B"]: ts[":C"]})
        with ts.scope(":Insert"):
            ts[":D"].map({ts[":B"]: ts["/:X"]})
        RegisterRule(ts, auto_assert_equal=True)
    delta = ts.freeze_frame() - freeze_frame
    assert not (delta.remove_nodes or delta.remove_facts)
    assert (delta.add_nodes
            == set({"/=", "/:Rule:Insert:B", "/:Rule:Equivalence:0"}))
    assert (delta.add_facts
            == set({
                ("/:Rule:Equivalence:0", "/:Rule:Insert:B", "/="),
                ("/:Rule:Equivalence:0", "/:Rule:MustMap:B", "/="),
                ("/:Rule:Equivalence:0", "/:Rule:_", "/RULE"),
                ("/:Rule:Insert:D", "/:Rule:Insert:B", "/:X"),
                ("/:Rule:RuleMap:0", "/:Rule:Insert:B", "/INSERT"),
            }))

def test_register_prototype():
    """Tests the RegisterPrototype(...) macro."""
    ts = TripletStructure()
    with ts.scope(":Rule"):
        ts[":A"].map({ts[":B"]: ts[":C"]})
        ts[":D"].map({ts[":A"]: ts["/:X"]})
        ts.commit()
        RegisterPrototype(ts, dict({
            ":RecognizeAIsX": {ts["/INSERT"]: [ts[":D"]]},
        }), [])
    # 2 above + 6 below
    assert len(ts.lookup(None, None, None)) == 8
    assert ts.lookup("/:Rule:RecognizeAIsX:RuleMap:0",
                     "/:Rule:RecognizeAIsX:_",
                     "/RULE")
    assert ts.lookup("/:Rule:RecognizeAIsX:RuleMap:0", "/:Rule:D", "/INSERT")
    assert ts.lookup("/:Rule:RecognizeAIsX:RuleMap:0", "/:Rule:D", "/NO_MAP")
    assert ts.lookup("/:Rule:RecognizeAIsX:RuleMap:0", "/:Rule:A", "/MUST_MAP")
    assert ts.lookup("/:Rule:RecognizeAIsX:RuleMap:0", "/:Rule:B", "/MUST_MAP")
    assert ts.lookup("/:Rule:RecognizeAIsX:RuleMap:0", "/:Rule:C", "/MUST_MAP")

    with ts.scope(":Rule"):
        AssertNodesEqual(ts, ts[":B, :C"], ":RecognizeAIsX")
        AssertNodesEqual(ts, ts[":A, :B"], ":RecognizeAIsX",
                         equal_type="/MAYBE=")
    freeze_frame = ts.freeze_frame()
    ts.rollback()
    with ts.scope(":Rule"):
        RegisterPrototype(ts, dict({
            ":RecognizeAIsX": {ts["/INSERT"]: [ts[":D"]]},
        }), [ts[":B, :C"]], [ts[":A, :B"]])
    assert ts.freeze_frame() == freeze_frame

main(__name__, __file__)
