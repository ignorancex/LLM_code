"""Heuristics for quickly solving letter analogy problems.

Matthew considers this to be written "in the DSL," although it's somewhat on
the border.
"""
from tactic_utils import ApplyRulesMatching, Fix, RuleFixedpoint
from analogy_utils import Analogy

def SolveLetterAnalogy(rt, verbose=True):
    """Applies tactics for quickly solving letter analogies.

    We have two sets of tactics. The first starts a mapping between two
    sub-structures (the examples), and the second then maps the prompt against
    that "abstraction."
    """
    maybe_print = lambda string: print(string) if verbose else None
    maybe_print("Labeling heads of letter groups...")
    RuleFixedpoint(rt, "/:HeadOfContainer:_")
    maybe_print("Identifying successor pairs...")
    ApplyRulesMatching(rt, "ConcretizePair")

    maybe_print("Mapping the examples...")
    no_slip = rt.ts.scope("/:Mapper:NoSlipRules", protect=True)
    partial = dict({
        no_slip[":C"]: "/:Analogy:From",
        no_slip[":A"]: "/:Analogy_abc_bcd:From:_",
        no_slip[":B"]: "/:Analogy_lmn_mno:From:_",
    })
    analogy = Analogy.Begin(rt, partial)
    ExtendAnalogyTactic(analogy)

    maybe_print("Mapping the prompt...")
    partial[no_slip[":B"]] = "/:Analogy_def_top:From:_"
    analogy = Analogy.Begin(rt, partial, exists=True)
    ExtendAnalogyTactic(analogy)

    maybe_print("Solving for letters...")
    analogy.state = "concretize"
    ExtendAnalogyTactic(analogy)
    ApplyRulesMatching(rt, "ConcretizePredecessor")
    ApplyRulesMatching(rt, "ConcretizeSuccessor")

def ExtendAnalogyTactic(analogy):
    """Heuristic for exploring a letter string.

    Basically, we first look at the head and then move rightward.
    """
    Fix(analogy.ExtendMap, ["/:HeadPair:Container", "/:NextPair:Left"])
    Fix(analogy.ExtendMap, ["/:SuccessorPair:Predecessor",
                            "/:SuccessorPair:Successor", "/:Owned"])
