"""Methods handling satisfying rule assignments.
"""
# pylint: disable=import-error,no-name-in-module
import runtime.utils as utils

class Assignment:
    """Represents a single satisfying assignment to a /RULE in the Structure.

    This class is effectively an intermediary between ProductionRule and
    TSDelta.
    """
    def __init__(self, rule, assignment):
        """Initialize the Assignment.
        """
        self.ts = rule.ts
        self.rule = rule
        self.assignment = assignment.copy()
        self.base_hash = utils.real_hash(assignment)

    def apply(self):
        """Applies the rule + assignment to the structure and returns the map.

        NOTE: Does *NOT* wrap the delta. The caller should do this if they want
        to.
        """
        # We will update this dictionary with newly-created nodes as necessary.
        running_assignment = self.assignment.copy()

        self.add_nodes(running_assignment)
        added_facts = self.add_facts(running_assignment)
        self.remove(running_assignment, added_facts)

        return running_assignment

    def add_nodes(self, running_assignment):
        """Adds /INSERT nodes and updates @running_assignment.
        """
        for node in self.unassigned_of_type(running_assignment, "/INSERT"):
            # NOTE: This also adds the node to the structure.
            new_node = self.ts[self.node_name(node) + ":??"].full_name
            running_assignment[node] = new_node
            for equivalent_node in self.rule.equal[node]:
                running_assignment[equivalent_node] = new_node

    def add_facts(self, running_assignment):
        """Adds facts to the structure.

        The add_facts are rule facts where all rule-nodes are assigned and not
        removed (/subtracted?).
        """
        translator = utils.Translator(running_assignment)
        ignore_nodes = set(self.rule.nodes_by_type["/REMOVE"])
        must_include = set(self.rule.nodes_by_type["/INSERT"])
        relevant_nodes = set(running_assignment.keys()) - ignore_nodes

        new_facts = []
        for fact in self.facts_of_nodes(sorted(relevant_nodes)):
            args = set(fact)
            if ((args & must_include) and
                    (set(fact) & self.rule.all_nodes) <= relevant_nodes):
                new_facts.append(translator.translate_tuple(fact))

        self.ts.add_facts(new_facts)
        return set(new_facts)

    def remove(self, running_assignment, added_facts):
        """Removes the relevant nodes & facts.

        We remove nodes that are marked /REMOVE or ones that are marked
        /SUBTRACT which have no facts after subtraction.
        """
        # (1) First, we just remove the /REMOVE nodes and any facts referencing
        # them.
        for node in self.assigned_of_type(running_assignment, "/REMOVE"):
            node = running_assignment[node]
            self.ts[node].remove_with_facts()

        # (2) Then we remove /SUBTRACT facts.
        # NOTE: Currently, addition takes precedence over removal. So if you
        # '/INSERT' a fact that already exists and '/REMOVE' or '/SUBTRACT' it
        # at the same time, it will end up *still in the structure*. The
        # example for where this semantics is useful is for something like the
        # Turing Machine example, where you might want to express keeping the
        # same head position as 'remove the current head position then put it
        # back in the same spot.'
        translator = utils.Translator(running_assignment)
        subtract = set(self.assigned_of_type(running_assignment, "/SUBTRACT"))
        for fact in self.assigned_rule_facts(running_assignment):
            if set(fact) & subtract and fact not in added_facts:
                self.ts.remove_fact(translator.translate_tuple(fact))

        # (3) Then, we remove /SUBTRACT nodes which have no facts.
        for node in self.assigned_of_type(running_assignment, "/SUBTRACT"):
            node = running_assignment[node]
            if not self.ts.facts_about_node(node, True):
                self.ts[node].remove()

    def node_name(self, node):
        """Returns the name to use for produced node @node.

        Ensures node names are deterministic and reproducible, regardless of
        when exactly the match happened.
        """
        return "/:" + utils.real_hash("{}{}".format(self.base_hash, node))

    def assigned_rule_facts(self, running_assignment):
        """Returns rule facts which do not involve remaining unassigned nodes.
        """
        assigned_rule_nodes = set(running_assignment.keys())
        for fact in self.rule.facts:
            fact_rule_nodes = set(fact) & self.rule.all_nodes
            if fact_rule_nodes <= assigned_rule_nodes:
                yield fact

    def assigned_of_type(self, running_assignment, of_type):
        """Returns nodes already assigned to of a particular type.
        """
        for node in self.rule.nodes_by_type[of_type]:
            if node in running_assignment:
                yield node

    def unassigned_of_type(self, running_assignment, of_type):
        """Returns nodes not yet assigned which are @of_type in self.rule.

        Eg., you could use this to return all nodes of_type=/INSERT which are
        not already constructed (i.e., in running_assignment).
        """
        for node in self.rule.nodes_by_type[of_type]:
            if node not in running_assignment:
                yield node

    def facts_of_nodes(self, nodes):
        """Helper method to return facts with one of @nodes in the first slot.

        NOTE: This uses the cached copy from @self.rule.indexed_facts, meaning
        these are such facts which were in the structure when @rule was
        initialized. We do this so we can remove rule-related nodes from the
        structure after initialization.
        """
        assert set(nodes) <= set(self.rule.all_nodes)
        return (fact for node in nodes
                for fact in self.rule.indexed_facts[node])

    def __str__(self):
        return str(self.assignment)
