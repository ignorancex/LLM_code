"""Defines a Matcher, which can efficiently enumerate ProductionRule matches.

This file defines the Matcher class, which keeps track of all matches to a
ProductionRule (optionally satisfying some partial assignment). It is
particularly optimized for applying rules until fixedpoint is reached, see eg.
../tactic_utils.py:RuleFixedpoint.

The main benefit is that Matcher can do "differential updates:" it keeps track
of all matches, and when the structure is modified it only needs to update
matches relevant to the delta between the old structure and modified one.

This works because ProductionRules are matched against in three 'layers:'

1. MustMap, 2. TryMap, 3. NoMap.

The key observation each one is *MONOTONIC*: eg. if you add a new fact, the
MustMap assignments in the new structure are a strict superset of those of the
old structure (and the new assignments *must* make use of the added fact). So,
when new facts are added, we can keep all of the old assignments and only check
new assignments which use the new facts.

Eg. given constraints [(1, 2, 3)] and new facts [(A, B, C), (B, C, D)], we only
need to check for new assignments which map {1: A, 2: B, 3: C} or {1: B, 2: C,
3: D}.

Note that profiling shows the vast majority of time is spent in MustMap, hence
this implementation only optimizes/caches that layer (it re-computes TryMap,
NoMap on every sync). In the future we can think through how to do differential
updates to the latter two layers as well.
"""
from collections import defaultdict
# pylint: disable=import-error,no-name-in-module
import runtime.utils as utils
from runtime.utils import freezedict, thawdict
from runtime.assignment import Assignment

class Matcher:
    """Keeps track of all assignments to a ProductionRule in the runtime."""
    def __init__(self, rt, rule, partial):
        """Initialize a Matcher."""
        self.rt = rt
        self.rule = rule
        self.freeze_frame = rt.ts.freeze_frame()
        self.partial = partial.copy()
        if any(isinstance(key, str) for key in partial.keys()):
            self.partial = dict({rule.node_to_variable[key]: value
                                 for key, value in partial.items()
                                 if key in rule.node_to_variable})
        # Assignments to the 'MustMap' pattern.
        self.must_matcher = PatternMatcher(rt, self.rule.must_pattern, self.partial)
        self.must_assignments = dict()
        for assignment in self.must_matcher.assignments:
            self._add_must(assignment)

    def assignments(self):
        """Yields assignments to self.rule satisfying self.partial.

        Will only yield assignments as of the last call to sync().
        """
        node_to_variable = utils.Translator(self.rule.node_to_variable)
        for must_assignment in sorted(self.must_assignments.keys()):
            entry = self.must_assignments[must_assignment]
            if any(never.assignments for never in entry["nevers"]):
                continue
            if entry["try"] is not None and entry["try"].assignments:
                for assign in map(thawdict, sorted(entry["try"].assignments)):
                    yield Assignment(self.rule, node_to_variable.compose(assign))
            else:
                assign = thawdict(must_assignment)
                yield Assignment(self.rule, node_to_variable.compose(assign))

    def sync(self):
        """Update the assignments."""
        current = self.rt.ts.freeze_frame()
        delta = self.freeze_frame.delta_to_reach(current)
        self.freeze_frame = current

        removed, added = self.must_matcher.sync(delta)

        for assign in removed:
            del self.must_assignments[assign]

        for existing in self.must_assignments:
            entry = self.must_assignments[existing]
            invalid = False
            for never in entry["nevers"]:
                never.sync(delta)
                invalid = invalid or bool(never.assignments)
            if invalid:
                entry["try"] = None
            elif entry["try"] is not None:
                entry["try"].sync(delta)
            else:
                entry["try"] = PatternMatcher(self.rt, self.rule.try_pattern, thawdict(existing))

        for assign in added:
            self._add_must(assign)

    def _add_must(self, frozen):
        """Adds an assignment to the list of must assignments."""
        assignment = thawdict(frozen)
        self.must_assignments[frozen] = dict({
            "nevers": [],
            "try": None,
        })
        invalid = False
        entry = self.must_assignments[frozen]
        for never in sorted(self.rule.never_patterns):
            never = self.rule.never_patterns[never]
            matcher = PatternMatcher(self.rt, never, assignment)
            entry["nevers"].append(matcher)
            invalid = invalid or bool(matcher.assignments)
            # TODO we could break if invalid, but then we'd need more logic
            # elsewhere.
        if not invalid:
            entry["try"] = PatternMatcher(self.rt, self.rule.try_pattern, assignment)

class PatternMatcher:
    """Keeps track of assignments to a single Pattern (existential formula).
    """
    def __init__(self, rt, pattern, partial):
        """Initialize the PatternMatcher.

        @pattern should be a Pattern instance, while @partial should be the
        same partial assignment on the owning Matcher instance.
        """
        self.rt = rt
        self.pattern = pattern
        self.partial = partial.copy()

        self.assignments = set()
        # Maps fact |-> set({assignments})
        self.assignments_relying_on_fact = defaultdict(set)
        # Maps assignment |-> set({facts})
        self.facts_used_in_assignment = defaultdict(set)

        # Initializes must, full.
        self.full_sync()

    def full_sync(self):
        """Initializes self.must, self.full (non-differential).
        """
        for assignment in self.pattern.assignments(self.partial):
            frozen = freezedict(assignment)
            for constraint in self.pattern.constraints:
                fact = tuple(assignment.get(arg, arg) for arg in constraint)
                self.assignments_relying_on_fact[fact].add(frozen)
                self.facts_used_in_assignment[frozen].add(fact)
            self.assignments.add(frozen)

    def sync(self, delta):
        """Updates the set of known assignments to match the current structure.
        """
        removed, added = set(), set()

        if not self.pattern.constraints:
            return removed, added

        # First, see if there are any assignments which rely on removed facts.
        for fact in delta.remove_facts:
            for relying in self.assignments_relying_on_fact[fact].copy():
                # NOTE: this call modifies assignments_relying_on_fact, which
                # is why we take a copy.
                self.remove_assignment(relying)
                removed.add(relying)
            self.assignments_relying_on_fact.pop(fact)

        # For all of the new facts, we find new assignments using them.
        partials = set()
        for fact in delta.add_facts:
            for constraint in self.pattern.constraints:
                assignment = self.unify(constraint, fact)
                if assignment:
                    partials.add(freezedict(assignment))
        partials -= set({freezedict(self.partial)})
        partials = [partial for partial in map(thawdict, sorted(partials))
                    if self.pattern.is_partial(partial)]
        for partial in partials:
            for new_must in self.pattern.assignments(partial):
                frozen = freezedict(new_must)
                if frozen not in self.assignments:
                    self.add_assignment(new_must, frozen)
                    added.add(frozen)

        assert not added & removed
        return removed, added

    def remove_assignment(self, frozen):
        """Remove an existing assignment from the set of valid assignments.

        Called by sync() when it is determined that the assignment is no longer
        valid.
        """
        self.assignments.remove(frozen)
        for fact in self.facts_used_in_assignment.pop(frozen):
            self.assignments_relying_on_fact[fact].remove(frozen)

    def add_assignment(self, assignment, frozen):
        """Add a newly-valid assignment to the set of valid assignments.
        """
        assert frozen not in self.assignments

        self.assignments.add(frozen)

        for constraint in self.pattern.constraints:
            fact = tuple(assignment.get(arg, arg) for arg in constraint)
            self.assignments_relying_on_fact[fact].add(frozen)
            self.facts_used_in_assignment[frozen].add(fact)

    def unify(self, constraint, fact):
        """Unifies a constraint with a fact into an assignment.

        Eg. constraint = (1, 2, 3), fact = (A, B, C) -> {1:A, 2:B, 3:C}.
        """
        assignment = self.partial.copy()
        inverse = defaultdict(list)
        for arg, var in zip(fact, constraint):
            if not isinstance(var, int):
                # Constants must match.
                if arg != var:
                    break
                continue
            if var in assignment and assignment[var] != arg:
                # Variables must match.
                break
            if any(other_var not in self.pattern.maybe_equal[var]
                   for other_var in inverse[arg]):
                break
            assignment[var] = arg
            inverse[arg].append(var)
        else:
            # We didn't break; @assignment is correct!
            return assignment
        return None

class OneOffMatcher:
    """Drop-in replacement for Matcher which does not save state."""
    def __init__(self, rt, rule, partial):
        """Initialize the OneOffMatcher."""
        self.rt = rt
        self.rule = rule
        self.partial = partial.copy()
        if any(isinstance(key, str) for key in partial.keys()):
            self.partial = dict({rule.node_to_variable[key]: value
                                 for key, value in partial.items()
                                 if key in rule.node_to_variable})

    def assignments(self):
        """Solves for and yields valid rule assignments."""
        node_to_variable = utils.Translator(self.rule.node_to_variable)

        must_assignments = self.rule.must_pattern.assignments(self.partial)
        for must_assignment in must_assignments:
            if self.rule.invalid(must_assignment):
                continue

            try_assignments = self.rule.try_pattern.assignments(must_assignment)
            any_assigned = False
            for try_assignment in try_assignments:
                any_assigned = True
                yield Assignment(
                    self.rule, node_to_variable.compose(try_assignment))
            # If there are no try_constraints, then try_assignments will still
            # output the 'trivial match,' and so it will have already been
            # checked against invalid in the above loop.
            if not self.rule.try_pattern.constraints:
                continue
            if not any_assigned:
                yield Assignment(
                    self.rule, node_to_variable.compose(must_assignment))

    def sync(self):
        """No-op for the OneOffMatcher, which is always up-to-date."""
