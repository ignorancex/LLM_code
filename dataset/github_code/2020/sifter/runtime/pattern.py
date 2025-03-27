"""Methods to simplify looking for patterns in a Structure.
"""
# pylint: disable=import-error
from collections import defaultdict
import runtime.utils as utils

class Pattern:
    """Represents an existential query/pattern to be matched against.
    """
    def __init__(self, runtime, constraints, maybe_equal, variable_names):
        """Initializes the Pattern.

        Arguments
        =========
        - @constraints should be a list of 3-tuples with strings (representing
            nodes) or integers (representing variables) as elements.
        - @maybe_equal should be a dictionary mapping {variable:
              maybe_equivalent_variables}.
        - @variable_names should be the corresponding names (in the structure)
              of all variables.
        """
        self.runtime = runtime
        self.constraints = constraints
        self.maybe_equal = maybe_equal
        self.variable_names = variable_names

    def named_assignment_to_vars(self, assignment):
        """Converts an assignment mapping node names to mapping numbers."""
        names_to_vars = dict({name: var
                              for var, name in self.variable_names.items()})
        return dict({names_to_vars[k]: v for k, v in assignment.items()})

    def n_variables(self):
        """Returns the number of variables to solve for in the pattern."""
        return len(set(
            arg for fact in self.constraints for arg in fact
            if isinstance(arg, int)))

    def assignments(self, partial_assignment=None):
        """Yields assignments satisfying the pattern.

        Each assignment is a dict {variable: node}. @partial_assignment, if
        provided, can be used to initialize some of the variables (see eg.
        production_rule.py:typecheck_with_facts).
        """
        if not self.constraints and partial_assignment is not None:
            yield partial_assignment
            return
        assert self.constraints

        partial_assignment = partial_assignment or dict()
        partial_assignment = utils.Translator(partial_assignment)

        constraints = partial_assignment.translate_tuples(self.constraints)
        assignments = self.runtime.solver.assignments(
            constraints, self.maybe_equal)
        for assignment in assignments:
            assignment = partial_assignment.concatenated_with(assignment)
            if self.valid_maybe_equals(assignment):
                yield assignment

    def equivalence_class(self, member):
        """Returns the equivalence class corresponding to variable @member.
        """
        if self.maybe_equal and member in self.maybe_equal:
            return self.maybe_equal[member]
        return set({member})

    def valid_maybe_equals(self, assignment):
        """Ensures any variables assigned together are marked maybe_equal.

        This happens particularly when using @partial_assignment.
        """
        preimages = defaultdict(set)
        for variable, node in assignment.items():
            preimages[node].add(variable)
        for variables in preimages.values():
            equivalence_class = self.equivalence_class(next(iter(variables)))
            if not variables <= equivalence_class:
                return False
        return True

    def add_constraint(self, constraint):
        """Adds more constraints to the Pattern.
        """
        self.constraints.append(constraint)

    def is_assignment(self, assignment):
        """True iff @assignment is a valid assignment. to the pattern."""
        return len(assignment) == self.n_variables() and self.is_partial(assignment)

    def is_partial(self, partial, check_eq=True):
        """If @partial can be extended to a valid assignment, returns True.

        NOTE: This is an *IF* not if*F*.
        If @check_eq=True, then it will always return False if @partial
        contradicts the equality constraints of the pattern.
        """
        assert not partial or isinstance(list(partial.keys())[0], int)
        if check_eq and not self.valid_maybe_equals(partial):
            return False
        partial = partial or dict()
        partial = utils.Translator(partial)
        constraints = partial.translate_tuples(self.constraints)
        for constraint in constraints:
            constraint = tuple(arg if isinstance(arg, str) else None
                               for arg in constraint)
            if not self.runtime.ts.lookup(*constraint, read_direct=True):
                return False
        return True

    def __str__(self):
        """Human-readable version of the Pattern."""
        return str(self.constraints)
