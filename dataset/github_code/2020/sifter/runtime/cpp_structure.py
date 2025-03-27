"""Python wrappers for the C++ solver."""
from collections import defaultdict
# pylint: disable=no-name-in-module
from ts_cpp import Structure, Triplet, Solver
import runtime.utils as utils

class CPPStructure:
    """Represents an optimized TripletStructure.

    Notably, the optimized TripletStructure is implemented in C++ and nodes are
    referenced by numerical indices, not strings.
    """
    def __init__(self, ts):
        """Initialize the CPPStructure."""
        self.ts = ts
        self.cpp = Structure()
        self.dictionary = dict({node: (i+1) for i, node in enumerate(ts.nodes)})
        self.dictionary_back = [None] + ts.nodes

        self.translator = utils.Translator(self.dictionary)
        existing = self.translator.translate_tuples(
            ts.lookup(None, None, None, read_direct=True))
        for fact in existing:
            self.cpp.addFact(*fact)

        ts.shadow = self

    def solve(self, pattern):
        """Given a CPPPattern, yields solutions to it in the structure."""
        if not pattern.valid:
            return
        if not pattern.sorted_variables:
            if all([self.ts.lookup(*fact, read_direct=True)
                    for fact in pattern.raw_constraints]):
                yield {}
            return

        solver = Solver(self.cpp, len(pattern.sorted_variables),
                        pattern.constraints, pattern.maybe_equal)

        while solver.isValid():
            assignment = solver.nextAssignment()
            if assignment:
                # Need to convert back to a dict with the original ordering.
                real_assignment = dict()
                for i, variable in enumerate(pattern.sorted_variables):
                    node = self.dictionary_back[assignment[i]]
                    real_assignment[variable] = node
                yield real_assignment
            else:
                return

    def assignments(self, constraints, maybe_equal=None):
        """Yields assignments to the constraints."""
        pattern = CPPPattern(self, constraints, maybe_equal)
        yield from self.solve(pattern)

    def add_node(self, node):
        """Add a node to the structure."""
        if node not in self.dictionary:
            self.dictionary[node] = len(self.dictionary) + 1
            self.dictionary_back.append(node)

    def remove_node(self, node):
        """No-op.

        Unconstrained nodes in patterns are not supported, hence for
        pattern-solving purposes a node is considered to be in the CPPStructure
        iff there are facts using it. the 'add_node' method above only assigns
        the node a numerical ID.
        """

    def add_fact(self, fact):
        """Add a fact to the structure."""
        self.cpp.addFact(*self.translator.translate_tuple(fact))

    def remove_fact(self, fact):
        """Remove a fact from the structure."""
        self.cpp.removeFact(*self.translator.translate_tuple(fact))

class CPPPattern:
    """Represents a pre-processed existential search query.

    For example, we might search for [(1, 3, 3), (1, 1, "/:A")] where 1 and 3
    are allowed to be equal. The C++ solver enforces a number of constraints
    that are not assumed on the Python side:
    1. Constants like "/:A" need to be replaced by their corresponding
       (positive) number in @cppstruct.dictionary.
    2. Variables need to be numbered in decreasing order starting from 0 ---
       no positive variable numbers and no gaps.
    3. Variables should be ordered in the order that they should be searched
       for in the structure (i.e., ordering heuristics must be computed on the
       Python side).
    For example, the [(1,3,3),(1,1,"/:A")] pattern might get pre-processed to
    the pattern [(-1,0,0),(-1,-1,1)], where -1<->1, 0<->3, and 1<->"/:A".
    """
    cached = dict()
    def __init__(self, cppstruct, constraints, maybe_equal):
        """Initialize and pre-process the pattern."""
        frozen = tuple(constraints)
        if (frozen in CPPPattern.cached
                and CPPPattern.cached[frozen][0] == maybe_equal):
            cached = CPPPattern.cached[frozen][1]
            self.raw_constraints = cached.raw_constraints
            self.constraints = cached.constraints
            self.valid = cached.valid
            self.sorted_variables = cached.sorted_variables
            self.maybe_equal = cached.maybe_equal
            return
        assert constraints
        self.raw_constraints = constraints
        # First, sort the variables.
        sorted_variables = []
        variables = sorted(set(i for fact in constraints for i in fact if isinstance(i, int)))
        n_fixed = [sum(isinstance(arg, str) for arg in constraint)
                   for constraint in constraints]
        def _compare_goodness_key(variable):
            return (n_fixed[variable] != 3,
                    n_fixed[variable],
                    str(constraints[variable]).count(":"))
        for _ in range(len(variables)):
            best_constraint = max(range(len(constraints)),
                                  key=_compare_goodness_key)
            arg = next(arg for arg in constraints[best_constraint]
                       if (not isinstance(arg, str) and
                           arg not in sorted_variables))
            sorted_variables.append(arg)
            for i, constraint in enumerate(constraints):
                if arg in constraint:
                    n_fixed[i] += 1
        # Then, rewrite the constraints with the sorted variables.
        translation = dict({old_var: -i
                            for i, old_var in enumerate(sorted_variables)})
        try:
            self.constraints = [
                Triplet(*[translation[arg] if arg in translation
                          else cppstruct.dictionary[arg]
                          for arg in constraint])
                for constraint in constraints]
        except KeyError:
            # E.g. the pattern uses a node that's not in the structure.
            self.valid = False
            return
        self.valid = True
        # Keep for the back-translation.
        self.sorted_variables = sorted_variables
        raw_maybe_equal = maybe_equal
        maybe_equal = maybe_equal or defaultdict(set)
        self.maybe_equal = [
            set({abs(translation[var]) for var in maybe_equal[v]
                 if var in translation})
            for v in sorted_variables]
        CPPPattern.cached[frozen] = (raw_maybe_equal, self)
