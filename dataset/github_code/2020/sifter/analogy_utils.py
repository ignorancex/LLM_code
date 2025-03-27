"""Python driver for interacting with the Mapper rules.

See AnalogyUtils.md for documentation.
"""
from collections import defaultdict
import itertools
from tactic_utils import RuleFixedpoint, RuleAny, GetMatcher

class Analogy:
    """Represents a single analogy between two things in the structure."""
    def __init__(self, rt, MAlphaA, MAlphaB, state="new"):
        """Initialize referring to an existing analogy in the structure."""
        self.rt = rt
        self.ts = rt.ts
        self.no_slip = rt.ts.scope("/:Mapper:NoSlipRules", protect=True)
        self.MAlphaA = MAlphaA
        self.MAlphaB = MAlphaB
        assert state in ("new", "concretize", "exists")
        self.state = state

    def ExtendMap(self, relations=None, partial=None, no_follow=False,
                  no_top=False):
        """Extend the analogy using a new map node.
        """
        partial = self.Partial(partial or dict())

        if relations:
            assert self.no_slip[":C"] not in partial
            for relation in relations:
                partial[self.no_slip[":C"]] = relation
                if self.ExtendMap(None, partial, no_follow):
                    return True
            return False

        if self.state == "concretize":
            proposals = itertools.chain(
                self.Propose(self.no_slip[":Concretize?Map:_"], partial),
                self.Propose(self.no_slip[":ConcretizeMap:_"], partial),
            )
        elif self.state == "exists":
            proposals = self.Propose(self.no_slip[":ExistingMap:_"], partial)
        else:
            proposals = self.Propose(self.no_slip[":NewMap:_"], partial)
        for assignment, _ in proposals:
            if no_top and self.IsTop(assignment[self.no_slip[":B"]]):
                continue
            alpha = assignment[self.no_slip[":AlphaMAB"]]
            if not self.Function(alpha) or not self.Injective(alpha):
                continue
            if self.ExtendFacts(alpha, no_follow=no_follow):
                return True
        return False

    def ExtendFacts(self, alpha_MAB, no_follow=False):
        """Extends the analogy to include facts about a new map node.

        If @no_follow, then it will not concretize any nodes, only lift/lower
        facts.
        """
        partial = self.Partial({self.no_slip[":AlphaMAB"]: alpha_MAB})
        recording = self.ts.start_recording()
        while not no_follow:
            if self.state == "concretize":
                proposals = self.Propose(self.no_slip[":Concretize?Concrete:_"], partial)
                for assignment, _ in proposals:
                    alpha = assignment[self.no_slip[":AlphaAB"]]
                    if (self.Injective(alpha) and self.Function(alpha)
                            and self.ConcretizeFacts(alpha_MAB)):
                        break
                else:
                    break
            elif self.state == "exists":
                proposals = self.Propose(self.no_slip[":ExistingConcrete:_"], partial)
                for assignment, _ in proposals:
                    alpha = assignment[self.no_slip[":AlphaAB"]]
                    if (self.Injective(alpha) and self.Function(alpha)
                            and self.FactsLower(alpha_MAB)):
                        break
                else:
                    break
            else:
                proposals = self.Propose(self.no_slip[":NewConcrete:_"], partial)
                for assignment, _ in proposals:
                    alpha = assignment[self.no_slip[":AlphaAB"]]
                    if self.Function(alpha) and self.LiftFacts(alpha_MAB):
                        break
                else:
                    break
        if self.state == "concretize":
            self.ConcretizeFacts(alpha_MAB)
        else:
            RuleFixedpoint(self.rt, self.no_slip[":NewFact:_"], partial)
        if self.FactsMissing(alpha_MAB) or not self.FactsLower(alpha_MAB):
            recording.rollback()
            return False
        return True

    def LiftFacts(self, alpha_M):
        """For every fact (m, ?, ?) try to add abstract fact (alpha_M, ?, ?).

        Only lifts a fact if all nodes are already abstracted. If all facts are
        lifted, returns True. If there are any facts which cannot be lifted, it
        undoes its changes and returns False to indicate that these two nodes
        probably shouldn't correspond to each other.
        """
        recording = self.ts.start_recording()
        RuleFixedpoint(self.rt, self.no_slip[":NewFact:_"], self.Partial({
            self.no_slip[":AlphaMAB"]: alpha_M,
        }))
        if self.FactsMissing(alpha_M):
            recording.rollback()
            return False
        return True

    def ConcretizeFacts(self, alpha_M):
        """Adds concrete facts corresponding to each (@alpha_M, ?, ?) fact.
        """
        RuleFixedpoint(self.rt, self.no_slip[":Concretize?Fact:_"],
                       self.Partial({
                           self.no_slip[":AlphaMAB"]: alpha_M,
                       }))
        return True

    def FactsMissing(self, alpha_m):
        """True iff there is some fact (m, ?, ?) not lifting to the abstract.

        Assumes m is abstracted to alpha_m.

        Used as a heuristics; we only lift 'm' to 'alpha_M' if we can lift all
        of the relevant facts.
        """
        missing = self.ts.scope("/:Mapper:MissingFacts:MustMap", protect=True)
        return RuleAny(self.rt, "/:Mapper:MissingFacts:_", dict({
            missing[":AlphaMAB"]: alpha_m,
        }))

    def FactsLower(self, alpha_m):
        """True iff all facts (@alpha_m, ?, ?) also hold in the concrete.
        """
        missing = self.ts.scope("/:Mapper:UnlowerableFacts:MustMap", protect=True)
        return not RuleAny(self.rt, "/:Mapper:UnlowerableFacts:_", dict({
            missing[":AlphaMAB"]: alpha_m,
        }))

    def Function(self, alpha):
        """True iff nothing mapped to @alpha is also mapped to something else.
        """
        not_fn = self.ts.scope("/:Mapper:NotFunction:MustMap", protect=True)
        return not RuleAny(self.rt, "/:Mapper:NotFunction:_", dict({
            not_fn[":AlphaA1"]: alpha,
        }))

    def Injective(self, alpha):
        """True iff at most one concrete node is mapped to @alpha.
        """
        not_inj = self.ts.scope("/:Mapper:NotInjective:MustMap", protect=True)
        return not RuleAny(self.rt, "/:Mapper:NotInjective:_", dict({
            not_inj[":AlphaA"]: alpha,
        }))

    def IsTop(self, node):
        """True iff some fact node claims @node is TOP."""
        return self.ts.lookup(None, node, "/:Mapper:TOP", read_direct=True)

    @classmethod
    def Begin(cls, rt, partial, exists=False, extend_here=True):
        """Begin an analogy.

        * Takes the first new analogy matching @partial.
        * If @exists, it assumes there is an existing analogy between two
          things and we want to include a third object in the comparison.
        * If @extend_here, it calls ExtendFacts on the map matched when
          starting the analogy.
        """
        no_slip = rt.ts.scope("/:Mapper:NoSlipRules", protect=True)
        proposals = rt.propose(no_slip[":HotBegin:_"], partial)
        if exists:
            proposals = rt.propose(no_slip[":IntoExisting:_"], partial)
        state = "exists" if exists else "new"
        for assignment, _ in proposals:
            analogy = cls(rt, assignment[no_slip[":MAlphaA"]],
                          assignment[no_slip[":MAlphaB"]], state=state)
            if extend_here:
                analogy.ExtendFacts(assignment[no_slip[":AlphaMAB"]])
            return analogy

    def Partial(self, start):
        """Helper to extend @start to specialize to this specific analogy."""
        start = start.copy()
        start[self.no_slip[":MAlphaA"]] = self.MAlphaA
        start[self.no_slip[":MAlphaB"]] = self.MAlphaB
        return start

    def Propose(self, rule, partial):
        """Helper to get assignments to a rule in the structure."""
        matcher = GetMatcher(self.rt, rule, partial)
        matcher.sync()
        return self.rt.matcher_propose(matcher)

    def Get(self):
        """Parse the analogy into a Python data structure.

        Returns three dicts, from_A, from_B, and from_abstract. from_A and
        from_B map concrete nodes to lists of abstract nodes. from_abstract maps
        abstract nodes to pairs (A, B) of lists of nodes in the A and B side.

        In both cases, the lists are empty or singletons unless the analogy is
        non-injective or not-a-function.
        """
        from_A, from_B = defaultdict(list), defaultdict(list)
        from_abstract = dict()

        facts_a = self.ts.lookup(self.MAlphaA, None, None)
        facts_b = self.ts.lookup(self.MAlphaB, None, None)
        abstract_nodes = set(fact[2] for fact in facts_a + facts_b)
        for abstract in sorted(abstract_nodes):
            concr_as = [fact[1] for fact in facts_a if fact[2] == abstract]
            concr_bs = [fact[1] for fact in facts_b if fact[2] == abstract]
            from_abstract[abstract] = (concr_as, concr_bs)
            for concr_a in concr_as:
                from_A[concr_a].append(abstract)
            for concr_b in concr_bs:
                from_B[concr_b].append(abstract)
        return from_A, from_B, from_abstract

    def Print(self):
        """Print a human-readable form of the analogy."""
        _, _, from_abstract = self.Get()
        for _, (concr_a, concr_b) in from_abstract.items():
            print(concr_a, "<==>", concr_b)
