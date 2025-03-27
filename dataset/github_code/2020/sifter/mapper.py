"""Rules to construct mappings between parts of a structure.

'Mappings', aka 'abstractions' or 'joins', are new structures which contain the
shared facts of two or more other sub-structures.

TODO(masotoud): Right now we build mappings up fact-by-fact. Instead, once we
build an initial mapping, we can find new 'instances' by turning that mapping
into a production rule.
"""
from ts_utils import RegisterRule, RegisterPrototype

def MapperCodelet(ts, name=":Mapper"):
    """Adds rules for the Mapper codelet to @ts.
    """
    with ts.scope(name) as scope:
        NoSlip(ts, scope)
        HelperRules(ts, scope)
    return scope

def NoSlip(ts, scope):
    """Prototypes extending a mapping without 'real slips.'

    Basically, these rules handle mappings where the relations line up, such as
    abc -> bcd, lmn -> mno and more complicated ones like abcefg -> fgh, xyzabc
    -> bcd.

    It *cannot* handle 'deeper' mappings, where the important relations
    actually change. Eg. abc -> bcd, nml -> mlk (here, successor is 'really
    slipping' with predecessor). Such mappings could be handled with a similar
    rule, but they're a lot harder to deal with because then the mapping chosen
    is more ambiguous.
    """
    with ts.scope(":NoSlipRules"):
        ts[":MA"].map({ts[":A"]: ts[":C"]})
        ts[":MB"].map({ts[":B"]: ts[":C"]})

        ts[":IsAbstractionA"].map({ts[":MAlphaA"]: scope[":Abstraction"]})
        ts[":IsAbstractionB"].map({ts[":MAlphaB"]: scope[":Abstraction"]})

        ts[":MAlphaA"].map({ts[":A"]: ts[":AlphaAB"]})
        ts[":MAlphaB1"].map({ts[":B"]: ts[":AlphaAB"]})

        ts[":MAlphaA"].map({ts[":MA"]: ts[":AlphaMAB"]})
        ts[":MAlphaB2"].map({ts[":MB1"]: ts[":AlphaMAB"]})

        ts[":NewAlphaMAB"].map({ts[":AlphaAB"]: ts[":C"]})

        RegisterPrototype(ts, dict({
            ":HotBegin": {ts["/MUST_MAP"]: ts[":A, :B, :C, :MA, :MB"]},
            ":NewConcrete": {ts["/INSERT"]: ts[":AlphaAB, :NewAlphaMAB"]},
            ":NewMap": {ts["/INSERT"]: ts[":AlphaMAB, :NewAlphaMAB"]},
            ":NewFact": {ts["/INSERT"]: [ts[":NewAlphaMAB"]]},
            ":IntoExisting": {
                ts["/INSERT"]:
                    ts[":IsAbstractionB, :MAlphaB, :MAlphaB1, :MAlphaB2"],
            },
            ":ExistingConcrete": {ts["/INSERT"]: [ts[":MAlphaB1"]]},
            ":ExistingMap": {ts["/INSERT"]: [ts[":MAlphaB2"]]},
            ":Concretize?Concrete": {
                ts["/INSERT"]: [ts[":B"], ts[":MAlphaB1"]],
            },
            ":Concretize?Map": {
                ts["/TRY_MAP"]: [ts[":MB"]],
                ts["/INSERT"]: [ts[":MB"], ts[":MB1"], ts[":MAlphaB2"]],
            },
            ":ConcretizeMap": {
                ts["/INSERT"]: [ts[":MB"], ts[":MB1"], ts[":MAlphaB2"]],
            },
            ":Concretize?Fact": {
                ts["/INSERT"]: [ts[":MB"]],
            },
        }), equal=[
            set({ts[":NewAlphaMAB"], ts[":AlphaMAB"]}),
            set({ts[":MAlphaB1"], ts[":MAlphaB2"], ts[":MAlphaB"]}),
            set({ts[":MB"], ts[":MB1"]}),
        ], maybe_equal=[
            set({ts[":A"], ts[":B"]}),
            set({ts[":MA"], ts[":MB"]}),
        ])

def HelperRules(ts, scope):
    """Helper patterns used by the analogy-making heuristics.

    See analogy_utils.py for more details on where these patterns are used.
    """
    with ts.scope(":NotFunction"):
        with ts.scope(":MustMap"):
            ts[":IsAbstractionA"].map({ts[":MAlphaA"]: scope[":Abstraction"]})
            ts[":MAlphaA"].map({ts[":A"]: ts[":AlphaA1"]})
            ts[":MAlphaA"].map({ts[":A"]: ts[":AlphaA2"]})
        RegisterRule(ts)

    with ts.scope(":NotInjective"):
        with ts.scope(":MustMap"):
            ts[":IsAbstractionA"].map({ts[":MAlphaA"]: scope[":Abstraction"]})
            ts[":MAlphaA"].map({
                ts[":A1"]: ts[":AlphaA"],
                ts[":A2"]: ts[":AlphaA"],
            })
        RegisterRule(ts)

    with ts.scope(":MissingFacts"):
        with ts.scope(":MustMap"):
            ts[":MA"].map({ts[":A"]: ts[":C"]})
            ts[":IsAbstractionA"].map({ts[":MAlphaA"]: scope[":Abstraction"]})
            # TODO: maybe this should be in NoMap?
            ts[":MAlphaA"].map({ts[":A"]: ts[":AlphaAB"]})
            ts[":MAlphaA"].map({ts[":MA"]: ts[":AlphaMAB"]})
        with ts.scope(":NoMap"):
            ts[":AlphaMAB"].map({ts[":AlphaAB"]: ts[":C"]})
        RegisterRule(ts, auto_assert_equal=True)

    with ts.scope(":UnlowerableFacts"):
        with ts.scope(":MustMap"):
            ts[":AlphaMAB"].map({ts[":AlphaAB"]: ts[":C"]})
            ts[":IsAbstractionA"].map({ts[":MAlphaA"]: scope[":Abstraction"]})
            ts[":MAlphaA"].map({ts[":A"]: ts[":AlphaAB"]})
            ts[":MAlphaA"].map({ts[":MA"]: ts[":AlphaMAB"]})
        with ts.scope(":NoMap"):
            ts[":MA"].map({ts[":A"]: ts[":C"]})
        RegisterRule(ts, auto_assert_equal=True)
