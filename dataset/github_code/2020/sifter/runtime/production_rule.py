"""Methods for parsing, mapping against, and executing structure /RULEs.
"""
# pylint: disable=no-name-in-module,import-error
from collections import defaultdict
from runtime.pattern import Pattern
import runtime.utils as utils

MAP_TYPES = ["/MUST_MAP", "/TRY_MAP", "/NO_MAP"]
ACTION_TYPES = ["/REMOVE", "/SUBTRACT", "/INSERT"]
NODE_TYPES = MAP_TYPES + ACTION_TYPES

class ProductionRule:
    """Represents a single /RULE in the TripletStructure.
    """
    def __init__(self, runtime, rule):
        """Initializes a ProductionRule given the corresponding node.

        NOTE: The structure can be modified arbitrarily once a ProductionRule
        is initialized; all relevant information is copied into the object
        itself.
        """
        self.runtime = runtime
        self.ts = runtime.ts
        self.name = rule
        self.is_backtracking = None
        self.parse_rule()
        self.assign_variables()
        self.prepare_constraints()
        self.facts = [fact for node in self.all_nodes
                      for fact in self.ts.lookup(node, None, None)]
        self.indexed_facts = dict({
            node: self.ts.lookup(node, None, None).copy()
            for node in self.all_nodes
        })

    def parse_rule(self):
        """Parses the relevant nodes to the rule (eg. MUST_MAP, etc.)
        """
        rule_node, solver = self.name, self.runtime.solver

        self.nodes_by_type = defaultdict(list)
        self.nodes_by_type["/NO_MAP"] = defaultdict(list)
        self.map_nodes = []
        self.all_nodes = set({rule_node})
        pattern = [(0, rule_node, "/RULE"), (0, 1, 2)]
        for assignment in solver.assignments(pattern):
            self.all_nodes.add(assignment[0])
            value, key = assignment[1], assignment[2]
            try:
                node_type = next(node_type for node_type in NODE_TYPES
                                 if key.startswith(node_type))
            except StopIteration:
                continue
            if node_type != "/NO_MAP":
                assert node_type == key
                self.nodes_by_type[key].append(value)
            else:
                index = int(key.split("/NO_MAP")[1].strip("_") or 0)
                self.nodes_by_type[node_type][index].append(value)
            if node_type in MAP_TYPES:
                self.map_nodes.append(value)
            self.all_nodes.add(value)

        self.equal = defaultdict(set)
        for assignment in solver.assignments([(0, rule_node, "/RULE"),
                                              (0, 1, "/="),
                                              (0, 2, "/=")]):
            # NOTE that assignments will call this for each permutation of the
            # two, so we don't actually need to special-case that.
            self.equal[assignment[1]].add(assignment[2])

        self.maybe_equal = defaultdict(set)
        for assignment in solver.assignments([(0, rule_node, "/RULE"),
                                              (0, 1, "/MAYBE="),
                                              (0, 2, "/MAYBE=")]):
            # NOTE see above.
            self.maybe_equal[assignment[1]].add(assignment[2])

    def assign_variables(self):
        """Gives each node in the mapping a variable name/ID/number.

        Populates node_to_variable, variable_to_node, and
        maybe_equal_variables.
        """
        # Every node that we might map against gets a variable name/ID.
        # Variable ID -> Node
        self.node_to_variable = dict()
        self.variable_to_node = dict()
        self.maybe_equal_variables = dict()
        # It's tempting here to give variables just to nodes marked /MUST_MAP,
        # but then you quickly run into problems because we often want to match
        # for the /NO_MAP nodes, etc. Plus other parts of the code will often
        # use variable numbers and node names interchangeably, so it's just
        # easier to support variable numbers for all relevant nodes. Unused
        # variable numbers don't have any negative impact.
        for node in sorted(self.all_nodes):
            variable = len(self.node_to_variable)
            for equivalent in self.equal[node]:
                try:
                    variable = self.node_to_variable[equivalent]
                    break
                except KeyError:
                    continue
            self.node_to_variable[node] = variable
            # The last one will take priority; we sort the iteration order so
            # this is deterministic.
            self.variable_to_node[variable] = node
            self.maybe_equal_variables[variable] = set({variable})

        for key, values in self.maybe_equal.items():
            key = self.node_to_variable[key]
            values = set(map(self.node_to_variable.get, values))
            self.maybe_equal_variables[key].update(values)

    def prepare_constraints(self):
        """Extracts the constraints corresponding to the rule.

        MUST be called after parse_rule().
        """
        ts = self.ts
        node_to_variable = utils.Translator(self.node_to_variable)

        # We will keep track of which nodes have constraints on them to avoid
        # free nodes.
        constrained = set()

        no_map = dict({
            node: index for index in self.nodes_by_type["/NO_MAP"].keys()
            for node in self.nodes_by_type["/NO_MAP"][index]})
        no_map_nodes = set(no_map.keys())

        def pattern():
            return Pattern(self.runtime, [], self.maybe_equal_variables,
                           self.variable_to_node)
        self.must_pattern = pattern()
        self.try_pattern = pattern()
        self.never_patterns = defaultdict(pattern)
        relevant_facts = (fact for node in self.map_nodes
                          for fact in ts.lookup(node, None, None))
        for fact in relevant_facts:
            constrained.update(fact)
            constraint = node_to_variable.translate_tuple(fact)
            arguments = set(fact)
            if arguments & no_map_nodes:
                index = next(no_map[argument] for argument in fact
                             if argument in no_map)
                self.never_patterns[index].add_constraint(constraint)
            elif arguments & set(self.nodes_by_type["/TRY_MAP"]):
                self.try_pattern.add_constraint(constraint)
            elif arguments & set(self.nodes_by_type["/INSERT"]):
                # We can try or fail to map against INSERT nodes, but they
                # should never be must_maps.
                pass
            else:
                assert set(fact) & set(self.nodes_by_type["/MUST_MAP"])
                self.must_pattern.add_constraint(constraint)

        assert not no_map_nodes & set(self.nodes_by_type["/TRY_MAP"])
        assert set(self.map_nodes) <= constrained

    def invalid(self, assignment):
        """True iff @assignment allows some of the /NEVER_MAPs to map.
        """
        return any(not utils.is_empty(pattern.assignments(assignment))
                   for pattern in self.never_patterns.values())
