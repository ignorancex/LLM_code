"""Macros which help building Triplet Structures.

Most of them are focused on automating the construction of production rules.
"""
from collections import defaultdict

def RegisterRule(
        ts, rule_name="", custom_qualifiers=None, auto_assert_equal=False):
    """A 'low-level' macro for building a rule based on node names.

    We assume that all nodes in the rule are within scopes:
    :[MapType]:[ActionType]
        - [ActionType] is applied to the node.
    :[MapType]
        - The node, if it exists, is left alone and anything done to it is
          added.
    :Insert
        - The node (and/or associated facts) are inserted.

    @custom_qualifiers can be used to extend the default list of scopes
    recognized. This is particularly useful when using the same set of nodes in
    multiple different rules (playing distinct roles in each).
    """
    qualifiers = dict({
        ":MustMap:": "/MUST_MAP",
        ":TryMap:": "/TRY_MAP",
        ":NoMap:": "/NO_MAP",
        ":NoMap1:": "/NO_MAP1",
        ":NoMap2:": "/NO_MAP2",
        ":NoMap3:": "/NO_MAP3",
        ":NoMap4:": "/NO_MAP4",
        ":NoMap5:": "/NO_MAP5",
        ":Remove:": "/REMOVE",
        ":Subtract:": "/SUBTRACT",
        ":Insert:": "/INSERT",
        ":OrInsert:": "/INSERT",
    })
    qualifiers.update(custom_qualifiers or dict())
    qualifiers = dict({key: value for key, value in qualifiers.items()
                       if value is not None})
    qualifiers_sorted = sorted(qualifiers.keys())

    equivalence_classes = defaultdict(set)

    node_scope = ts.scope()
    with ts.scope(rule_name) as rule_scope:
        mapnode = ts[":RuleMap:??"]
        for node in node_scope:
            name = node - node_scope
            for qualifier in qualifiers_sorted:
                if qualifier in name:
                    mapnode.map({
                        rule_scope[":_"]: ts["/RULE"],
                        node: ts[qualifiers[qualifier]],
                    })
                    first_name = name.split(":")[-1]
                    equivalence_classes[first_name].add(node)

    if auto_assert_equal:
        handled = set()
        for equivalence_class in equivalence_classes.values():
            if equivalence_class <= handled:
                # TODO(masotoud): is this reachable?
                continue
            handled.update(equivalence_class)
            if len(equivalence_class) == 1:
                continue
            AssertNodesEqual(ts, sorted(equivalence_class, key=str), rule_name)

def RegisterPrototype(ts, rules, equal, maybe_equal=None):
    """A higher-level macro for describing "simpler" rules.

    Essentially, Prototypes are rules where you map against some subset of the
    structure, then the other subset is inserted.

    The actual semantics of RegisterPrototype get a bit hairy, because a lot of
    things are assumed implicitly. For example, things you mark as /INSERT but
    not /TRY_MAP are implicitly declared /NO_MAP as well. TODO(masotoud):
    better document and simplify such behavior.

    Arguments
    =========
    - @rules should be a dictionary {name: rule}, where "rule" is tuple (nodes,
          action). @nodes is a list of nodes, while @action is the action to
          paply to those nodes (either /INSERT or /MUST_MAP). The other nodes
          in the scope are treated as the other of the two.
    - @equal is a list of tuples of nodes which should share an assignment.
    """
    scope = ts.scope()
    all_nodes = list(scope)
    maybe_equal = maybe_equal or []
    for rule_name, rule in sorted(rules.items(), key=lambda x: x[0]):
        rule_scope = ts.scope(rule_name)
        map_node = rule_scope[":RuleMap:??"]
        map_node.map({rule_scope[":_"]: ts["/RULE"]})

        if ts["/INSERT"] in rule.keys():
            assert ts["/NO_MAP"] not in rule
            if ts["/TRY_MAP"] in rule.keys():
                rule[ts["/NO_MAP"]] = [node for node in rule[ts["/INSERT"]]
                                       if node not in rule[ts["/TRY_MAP"]]]
            else:
                rule[ts["/NO_MAP"]] = rule[ts["/INSERT"]]

        remaining_nodes = set(all_nodes.copy())
        for node_type, nodes_of_type in rule.items():
            remaining_nodes = remaining_nodes - set(nodes_of_type)
            if node_type is not None:
                map_node.map({node: node_type for node in nodes_of_type})

        if ts["/INSERT"] not in rule:
            map_remaining_to = ts["/INSERT"]
        elif ts["/MUST_MAP"] not in rule:
            map_remaining_to = ts["/MUST_MAP"]
        else:
            map_remaining_to = None

        if map_remaining_to is not None:
            remaining_nodes = sorted(remaining_nodes)
            map_node.map({node: map_remaining_to for node in remaining_nodes})

        for node_set in equal:
            AssertNodesEqual(ts, node_set, rule_name)

        for node_set in maybe_equal:
            AssertNodesEqual(ts, node_set, rule_name, equal_type="/MAYBE=")

def AssertNodesEqual(ts, nodes, rule_scope, equal_type="/="):
    """Enforces that a set of nodes share an assignment in the rule.

    TODO(masotoud): use an actual rule_scope.
    """
    map_node = ts[":Equivalence:??"]
    map_node.map({ts.scope(rule_scope)[":_"]: ts["/RULE"]})
    for node in nodes:
        map_node.map({node: ts[equal_type]})
