"""Example program implementing a Turing machine in TSLang.

Note that we use a few special names, namely "tc" representing the Triplet
Structure being operated on and "rt" representing the Runtime operating on that
structure.
"""
from ts_lib import TripletStructure
from ts_utils import RegisterRule
from runtime.runtime import TSRuntime

def Main():
    """Builds a basic TM with one transition rule.

    In effect, we're building up a 'graph' where nodes represent a few possible
    things:
    1. 'Prototypical types' representing concepts such as "the A state" or "the
           leftmost symbol."
    2. Transition rules, which are themselves composed of nodes that are mapped
           into both the prototypical types declared here and those of the
           'underlying type checker,' such as /RULE and /IMPLICANT.
    3. Nodes representing the current state and the machine tape.

    If @return_proposals=False, does not print the proposals. Useful for
    automated testing.
    """
    ts = TripletStructure()
    # Initialize prototypes for the two states. Note that this is not strictly
    # necessary, as nodes will be created implicitly when first referenced, but
    # it's nice to explicitly declare "standard" nodes in one place. Also note
    # that the ts.scope(...) blocks ensure these nodes are named eg.
    # ts["/:State:A"], not just ts["/:A"].
    # pylint: disable=pointless-statement
    with ts.scope(":State"):
        ts[":A, :B"]
    # Initialize prototypes for the types of symbols on the tape.
    with ts.scope(":Symbol"):
        ts[":0, :1, :2"]
    # Initialize a prototype representing the current tape symbol.
    ts[":Mark"]
    # Initialize prototypes for, effectively, the "next to relation." One of
    # the prototype nodes is "the one on the left" and the other is "the one on
    # the right."
    with ts.scope(":NextPair"):
        ts[":Left, :Right"]
    # Add a transition rules.
    TransitionRule(ts,
                   name=":Transition0A",
                   state=ts[":State:A"],
                   read_symbol=ts[":Symbol:2"],
                   write_symbol=ts[":Symbol:1"],
                   direction="R",
                   statep=ts[":State:B"])
    # Add a node corresponding to the current state and map it as an instance
    # of state A. The "??"s are filled in until a unique name is found. They
    # are convenient to use for avoiding possible name collisions (especially
    # in macro code).
    ts[":MState"].map({ts[":CurrentState"]: ts[":State:A"]})
    # Add the initial symbol to the tape, initialized to 2.
    ts[":MSymbolType"].map({ts[":OriginSymbol"]: ts[":Symbol:2"]})
    # Mark the current symbol.
    ts[":MSymbolMark"].map({ts[":OriginSymbol"]: ts[":Mark"]})
    # The TSRuntime will parse and apply the rules to the structure.
    rt = TSRuntime(ts)
    # Print the state of the TM.
    PrintTMState(ts)
    # Execute one step.
    proposals = list(rt.propose_all())
    assert len(proposals) == 1
    print("Taking one step...")
    _, delta = proposals[0]
    delta.apply()
    PrintTMState(ts)
    return ts

def TransitionRule(
        ts, name, state, read_symbol, write_symbol, direction, statep):
    """Adds a transition rule to the structure."""
    print(f"Adding Transition Rule {name}:\n" +
          f"\tFrom State: {state}, Read Symbol: {read_symbol}\n" +
          f"\tTo State: {statep}, Write Symbol: {write_symbol}\n" +
          f"\tMove: {direction}")
    with ts.scope(name):
        # We must have the current state and the current symbol.  We will
        # overwrite this information when the rule is applied.
        with ts.scope(":MustMap:Subtract"):
            ts[":MState"].map({ts[":State"]: state})
            ts[":MSymbol"].map({ts[":Symbol"]: read_symbol})
            ts[":MMarker"].map({ts[":Symbol"]: ts["/:Mark"]})

        # If it exists, we should map against the next symbol we want.  If it
        # does not exist, we should insert it.
        with ts.scope(":TryMap:OrInsert"):
            if direction == "L":
                ts[":MNewSymbol"].map({
                    ts[":NewSymbol"]: ts["/:NextPair:Left"],
                    ts[":Symbol"]: ts["/:NextPair:Right"],
                })
                new_symbol = ts[":NewSymbol"]
            elif direction == "R":
                ts[":MNewSymbol"].map({
                    ts[":Symbol"]: ts["/:NextPair:Left"],
                    ts[":NewSymbol"]: ts["/:NextPair:Right"],
                })
                new_symbol = ts[":NewSymbol"]
            else:
                new_symbol = ts[":Symbol"]

        with ts.scope(":Insert"):
            ts[":MState"].map({ts[":State"]: statep})
            ts[":MSymbol"].map({ts[":Symbol"]: write_symbol})
            ts[":MMarker"].map({new_symbol: ts["/:Mark"]})

        RegisterRule(ts, auto_assert_equal=True)

def PrintTMState(ts):
    """Pretty-print the current state of the TM."""
    print("Printing Current Turing Machine:")
    assert len(ts.lookup(None, "/:CurrentState", None)) == 1
    current_state = ts.lookup(None, "/:CurrentState", None)[0][2]
    print("\tCurrent state:", current_state)

    assert len(ts.lookup(None, None, "/:Mark")) == 1
    head_node = ts.lookup(None, None, "/:Mark")[0][1]

    symbols = ["/:OriginSymbol"]
    while True:
        left = GetNodeNeighbor(ts, symbols[0], "/:NextPair:Left")
        if left:
            symbols.insert(0, left)
        right = GetNodeNeighbor(ts, symbols[-1], "/:NextPair:Right")
        if right:
            symbols.append(right)
        if not (left or right):
            break
    head_index = symbols.index(head_node)
    symbols = [ts.lookup("/:MSymbolType", node, None)[0][2]
               if ts.lookup("/:MSymbolType", node, None) else "X"
               for node in symbols]
    print("\tCurrent Tape Contents:", symbols)
    print("\tCurrent Head Index:", head_index)
    return (current_state, symbols, head_index)

def GetNodeNeighbor(ts, node, side):
    """Returns node on the @side side of @node in @ts."""
    opposite_side = dict({
        "/:NextPair:Right": "/:NextPair:Left",
        "/:NextPair:Left": "/:NextPair:Right",
    })[side]
    try:
        fact = ts.lookup(None, node, opposite_side)[0][0]
        return ts.lookup(fact, None, side)[0][1]
    except IndexError:
        return None

if __name__ == "__main__":
    Main()
