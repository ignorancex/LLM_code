"""Methods for parsing and executing TripletStructures according to their rules.
"""
# pylint: disable=import-error,no-name-in-module
from runtime.cpp_structure import CPPStructure
from runtime.production_rule import ProductionRule
from runtime.interactive import TSREPL
from runtime.matcher import OneOffMatcher

class TSRuntime:
    """A runtime for interpreting and executing triplet structures.
    """
    def __init__(self, ts):
        """Initializes a new TSRuntime.
        """
        ts.commit(False)
        self.ts = ts
        # The Solver handles the 'dirty work' of actually finding matches to
        # rule implicants.
        self.solver = CPPStructure(self.ts)
        self.extract_rules()
        ts.commit(False)

    def interactive(self):
        """Begins an interactive runtime session.
        """
        repl = TSREPL(self)
        repl.run()

    def extract_rules(self):
        """Populates self.rules and deletes the structure nodes.

        NOTE: This function removes all rule-related nodes from the structure
        after parsing them into ProductionRules. This is a relatively cheap and
        easy way to ensure we don't have to worry about rules applying to other
        rules, etc. In the long run, though, it would be nice to support
        reflective rules.
        """
        ts = self.ts

        self.rules = []
        self.rules_by_name = dict()

        rule_nodes = set(fact[1] for fact in ts.lookup(None, None, "/RULE"))
        for rule_node in sorted(rule_nodes):
            rule = ProductionRule(self, rule_node)
            self.rules.append(rule)
            self.rules_by_name[rule.name] = rule

        avoid_nodes = set(avoid_node for rule in self.rules
                          for avoid_node in rule.all_nodes)
        for avoid_node in avoid_nodes:
            ts[avoid_node].remove_with_facts()
        for node in ts.nodes.copy():
            if not node.startswith("/:"):
                ts[node].remove()

    def matcher_propose(self, matcher):
        """Propose TSDeltas based on a Matcher.

        Yields (assignment, delta) pairs, where delta = assignment.produce().
        """
        assert self.ts.is_clean()

        for assignment in matcher.assignments():
            assignment = assignment.apply()
            delta = self.ts.commit(commit_if_clean=True)

            if delta:
                yield (assignment, delta)
            if self.ts.path[-1] is delta:
                self.ts.rollback(-1)

    def propose(self, rule, partial=None):
        """Propose TSDeltas based on the rules.

        Yields (assignment, delta) pairs, where delta = assignment.produce().
        """
        rule = self.rules_by_name[rule]
        partial = partial or dict()
        matcher = OneOffMatcher(self, rule, partial or dict())
        yield from self.matcher_propose(matcher)

    def propose_all(self, rules=None):
        """Helper to yield proposals from multiple ProductionRules.
        """
        if rules is None:
            rules = [rule.name for rule in self.rules]
        for rule in rules:
            yield from self.propose(rule)

    def get_rule(self, name):
        """Returns the ProductionRule associated with @name.

        In the structure, there is some fact (A, @node, "/RULE").
        """
        return next(rule for rule in self.rules if rule.name == name)
