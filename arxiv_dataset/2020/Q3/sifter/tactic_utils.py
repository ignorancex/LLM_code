"""Collection of macros for writing Tactics.

Matthew considers this to be written "in the DSL," although it's somewhat on
the border.
"""
from runtime.matcher import Matcher, OneOffMatcher

def SearchRules(rt, search_term):
    """Returns all rules with @search_term in their name.

    Used, for example, to quickly mark all successor pairs in a structure
    (since currently "X is A, Y is B -> Successor(X, Y)" is a separate rule for
    each A, B).
    """
    return [rule.name for rule in rt.rules if search_term in rule.name]

def Fix(function, *args, **kwargs):
    """Applies @function repeatedly until it returns False.

    Many tactic functions return True if they made a change. This method allows
    you to apply those tactics until fixedpoint is reached.
    """
    while function(*args, **kwargs):
        pass

def ApplyRulesMatching(rt, search_term, partial=None):
    """Calls RuleFixedpoint for every rule containing @search_term.

    Note that this is *all* it does --- it does not, e.g., ever repeat the 1st
    rule after applying the 3rd. If you want the fixedpoint for all of the
    rules together, you need to wrap this call in a Fix(...).
    """
    rules = SearchRules(rt, search_term)
    did_anything = False
    for rule in rules:
        did_anything = RuleFixedpoint(rt, rule, partial) or did_anything
    return did_anything

MATCHERS = dict()
def GetMatcher(rt, rule, partial, one_off=False):
    """Returns a Matcher keeping track of applications to @rule."""
    rule = rt.rules_by_name[rule]
    if one_off:
        return OneOffMatcher(rt, rule, partial)
    key = (id(rt), rule, tuple(sorted(partial.items())))
    if key not in MATCHERS:
        MATCHERS[key] = Matcher(rt, rule, partial)
    return MATCHERS[key]

def RuleFixedpoint(rt, rule, partial=None):
    """Given a rule, applies it repeatedly until fixedpoint is reached.
    """
    matcher = GetMatcher(rt, rule, partial or dict({}))

    did_anything = False
    while True:
        matcher.sync()
        # NOTE: for correctness, rt.matcher_propose assumes you only ever use
        # exactly one of the things it yields.
        try:
            _ = next(rt.matcher_propose(matcher))
        except StopIteration:
            break
        did_anything = True
    return did_anything

def RuleAny(rt, rule, partial, one_off=True):
    """True iff @rule has any matches extending @partial in the structure."""
    matcher = GetMatcher(rt, rule, partial, one_off=one_off)
    matcher.sync()
    try:
        _ = next(matcher.assignments())
        return True
    except StopIteration:
        pass
    return False
