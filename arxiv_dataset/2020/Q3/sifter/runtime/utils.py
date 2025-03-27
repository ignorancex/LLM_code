"""Assorted helper functions for the TSRuntime."""
import hashlib

def freezedict(dictionary):
    """Freezes a dict to a hashable form, e.g. to store in a set."""
    return tuple(sorted(dictionary.items()))

def thawdict(dictionary):
    """Thaws a previously-frozen dict."""
    return dict(dictionary)

def is_empty(generator):
    """True iff the generator is empty.

    Used primarily to check if there are any satisfying assignments to a
    pattern, eg. in production_rule.py.
    """
    try:
        next(generator)
        return False
    except StopIteration:
        return True

class Translator:
    """Helper class for translation dictionaries.

    Useful, eg., when you want to translate between a set of constraints (using
    variable names) and ``filled-in'' constraints according to some assignment.
    """
    def __init__(self, translation):
        """Initializes the Translator.

        @translation should be a dictionary.
        """
        self.translation = translation

    def translate(self, element):
        """Returns the translation of @element (or @element if no translation).
        """
        return self.translation.get(element, element)

    def translate_tuple(self, elements):
        """Translates all elements of tuple @elements in @self.translation."""
        return tuple(map(self.translate, elements))

    def translate_tuples(self, elements):
        """Translates a list of tuples."""
        return list(map(self.translate_tuple, elements))

    def translate_list(self, elements):
        """Translates each member of a list."""
        return list(map(self.translate, elements))

    def compose(self, after, default_identity=False):
        """Returns the composition of @self.translation with the dict @after.

        Used eg. when we have an assignment to _variables_ that we want to turn
        in to an assignment to _nodes_ using a node-to-variable map.
        """
        composed = dict()
        for key, value in self.translation.items():
            try:
                composed[key] = after[value]
            except KeyError:
                if default_identity:
                    composed[key] = value
        return composed

    def concatenated_with(self, other):
        """Returns the concatenation of @self.translation and @after."""
        concatenated = self.translation.copy()
        concatenated.update(other)
        return concatenated

def real_hash(item):
    """Returns a "cryptographically-secure-ish" hash of @item.

    This is used in particular in assignment.py for giving newly-created nodes
    unambiguous, reproducible names based only on their 'source' assignment.
    """
    if isinstance(item, str):
        return hashlib.sha224(item.encode()).hexdigest()
    if isinstance(item, dict):
        # NOTE: this assumes that the str(...) does not include any
        # non-deterministic information (eg. ids). Maybe it would be best to
        # let real_hash operate directly on the sorted list.
        return real_hash(str(sorted(item.items())))
    raise NotImplementedError
