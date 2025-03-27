"""An example TripletStructure for solving a letter analogy problem.

Throughout this codebase we use the shorthand `tc` to represent the
TripletStructure being operated on, and `rt` to represent the Runtime operating on
that TripletStructure. I have also standardized to upper-case function names to
differentiate the "front-end" code using ts_lib from the "plumbing" code that
actually implements the DSL. These decisions can be undone if no one else likes
them.
"""
import string # pylint: disable=deprecated-module
from timeit import default_timer as timer
from ts_lib import TripletStructure
from ts_utils import RegisterPrototype, RegisterRule
from runtime.runtime import TSRuntime
from mapper import MapperCodelet
from letter_tactics import SolveLetterAnalogy

def Main(verbose=True):
    """Initialize the structure and solve the letter analogy.
    """
    # This will hold our facts and nodes (including the rules and tactics!).
    ts = TripletStructure()
    # Add rules describing common letter relations, eg. 'Successor.'
    LetterRelations(ts)
    # Add (standardized) rules describing how to map/make analogies/join
    # sub-structures. This can handle eg. abc -> bcd and lmn -> mno being
    # 'abstracted' to x_1x_2x_3 -> y_1y_2y_3 with Succ(x_1, y_1), etc.
    MapperCodelet(ts)

    # Add three letter analogies, describing the problem to be solved. None
    # indicates that the 'solution' should go there (it's indicated by marking
    # the corresponding node with TOP).
    LetterAnalogy(ts, ":Analogy_abc_bcd", "abc", "bcd")
    LetterAnalogy(ts, ":Analogy_lmn_mno", "lmn", "mno")
    LetterAnalogy(ts, ":Analogy_def_top", "def", None)

    # Now that we have a structure (ts) which fully describes our problems,
    # rules, and heuristics, we can initialize a TSRuntime to modify the
    # structure according to the rules.
    rt = TSRuntime(ts)

    SolveLetterAnalogy(rt, verbose=verbose)

    solution = ExtractLetterGroup(ts, ts["/:Analogy_def_top:To:_"])

    if verbose:
        print(f"Proposed solution: {solution}")

    return solution

def ExtractLetterGroup(ts, letter_group):
    """Tries to extract a string representation of the letter group.

    This is not always successful, as the representation might be invalid (eg.
    cycles, or more than one NextPair:Right). In such cases it returns None.
    """
    try:
        # NOTE: These may be ambiguous, to be sure we should have a few
        # assert(len(...) == 1)s here.
        letter_group = letter_group.full_name
        head_map = ts.lookup(None, letter_group, "/:HeadPair:Container")[0][0]
        head = ts.lookup(head_map, None, "/:HeadPair:Head")[0][1]
        letters = ""
        visited = set()
        while True:
            if head in visited:
                return None
            visited.add(head)

            for letter in string.ascii_lowercase + string.ascii_uppercase:
                is_letter = ts.lookup(None, head, Letter(ts, letter).full_name)
                if is_letter:
                    letters += letter
                    break
            else:
                letters += "?"
            next_map = ts.lookup(None, head, "/:NextPair:Left")
            if not next_map:
                break
            next_map = next_map[0][0]
            head = ts.lookup(next_map, None, "/:NextPair:Right")[0][1]
        return letters
    except IndexError:
        return None

def LetterAnalogy(ts, name, analogy_from, analogy_to):
    """Adds nodes describing a particular letter analogy.

    @name should be a unique node prefix, while @analogy_from and @analogy_to
    should be strings describing the word analogy.
    """
    with ts.scope(name):
        LetterGroup(ts, analogy_from, ts.scope(":From"))
        LetterGroup(ts, analogy_to, ts.scope(":To"))
        ts[":AnalogyMap"].map({
            ts[":From:_"]: ts["/:Analogy:From"],
            ts[":To:_"]: ts["/:Analogy:To"]
        })

def LetterGroup(ts, letters, scope):
    """Adds nodes describing a string of letters (eg. one side of an analogy).

    The entire group is represented by a node scope[:_] which is marked as the
    owner of all other nodes.
    """
    with scope:
        if letters is None:
            ts[":IsTopMap:??"].map({ts[":_"]: ts["/:Mapper:TOP"]})
            return
        nodes = []
        for i, letter in enumerate(letters):
            node = ts[":Letter{}_{}".format(i, letter)]
            nodes.append(node)
            ts[":IsLetterMap:??"].map({node: Letter(ts, letter)})
            ts[":IsOwned:??"].map({
                scope[":_"]: ts["/:Owner"],
                node: ts["/:Owned"],
            })
        for left_node, right_node in zip(nodes[:-1], nodes[1:]):
            ts[":LRMap:??"].map({
                left_node: ts["/:NextPair:Left"],
                right_node: ts["/:NextPair:Right"],
            })

def Letter(ts, letter):
    """Returns a reference to the node corresponding to character @letter.

    This is sort of the "Platonic conception" of @letter.
    """
    return ts["/:Letter:{}".format(letter)]

def LetterRelations(ts):
    """Adds relations and rules describing strings of letters.
    """
    # Declaring these is not strictly necessary, as they will be created when
    # first referenced, but I am listing them here for reference.
    # pylint: disable=pointless-statement
    with ts.scope(":NextPair"):
        ts[":Left, :Right"]
    with ts.scope(":SuccessorPair"):
        ts[":Predecessor, :Successor"]
    with ts.scope(":UpperPair"):
        ts[":Lower, :Upper"]
    with ts.scope(":HeadPair"):
        ts[":Container, :Head"]

    Headify(ts)

    for predecessor, successor in zip(string.ascii_lowercase[:-1],
                                      string.ascii_lowercase[1:]):
        SuccessorPrototype(ts, predecessor, successor)
        SuccessorPrototype(ts, predecessor.upper(), successor.upper())

    for letter in string.ascii_lowercase:
        UpperPrototype(ts, letter, letter.upper())

def SuccessorPrototype(ts, predecessor, successor):
    """Rules that identify and concretize successor pairs.

    Note that this approach will create 3 rules for each pair (a, b); one that
    creates Successor(a, b) from a and b, one that creates `a` given
    Successor(a, b) and b, and one that creates `b` given Successor(a, b) and
    `a`.

    An alternative approach is to define three "Generic Binary Prototype"
    rules, and then expose these as "examples" that that rule then maps
    against. I think this is a more straight-forward approach for now, though
    in the future (or if there's too much overhead with parsing the rules) we
    can look at the other option.
    """
    with ts.scope(":Successor{}To{}".format(predecessor, successor)):
        ts[":MA"].map({ts[":A"]: Letter(ts, predecessor)})
        ts[":MB"].map({ts[":B"]: Letter(ts, successor)})
        ts[":PairMap"].map({ts[":A"]: ts["/:SuccessorPair:Predecessor"]})
        ts[":PairMap"].map({ts[":B"]: ts["/:SuccessorPair:Successor"]})

        RegisterPrototype(ts, dict({
            ":ConcretizePredecessor": {ts["/INSERT"]: [ts[":MA"]]},
            ":ConcretizeSuccessor": {ts["/INSERT"]: [ts[":MB"]]},
            ":ConcretizePair": {ts["/INSERT"]: [ts[":PairMap"]]},
        }), equal=[])

def UpperPrototype(ts, lowercase, uppercase):
    """Rules for identifying and concretizing Upper pairs.

    See SuccessorPrototype for notes on this implementation.
    """
    with ts.scope(":Upper{}To{}".format(lowercase, uppercase)):
        ts[":MA"].map({ts[":A"]: Letter(ts, lowercase)})
        ts[":MB"].map({ts[":B"]: Letter(ts, uppercase)})
        ts[":PairMap"].map({ts[":A"]: ts["/:UpperPair:Lower"]})
        ts[":PairMap"].map({ts[":B"]: ts["/:UpperPair:Upper"]})

        RegisterPrototype(ts, dict({
            ":ConcretizePredecessor": {ts["/INSERT"]: [ts[":MA"]]},
            ":ConcretizeSuccessor": {ts["/INSERT"]: [ts[":MB"]]},
            ":ConcretizePair": {ts["/INSERT"]: [ts[":PairMap"]]},
        }), equal=[])

def Headify(ts):
    """Rules that identify the head letter of a letter group.
    """
    with ts.scope("/:HeadOfContainer"):
        with ts.scope(":MustMap") as exist:
            ts[":IsLeftOf"].map({ts[":LeftMost"]: ts["/:NextPair:Left"]})
            ts[":IsMember"].map({
                ts[":Container"]: ts["/:Owner"],
                ts[":LeftMost"]: ts["/:Owned"],
            })
        with ts.scope(":NoMap"):
            ts[":IsRightOf"].map({exist[":LeftMost"]: ts["/:NextPair:Right"]})
        with ts.scope(":TryMap:Insert"):
            ts[":IsHead"].map({
                exist[":Container"]: ts["/:HeadPair:Container"],
                exist[":LeftMost"]: ts["/:HeadPair:Head"],
            })
        RegisterRule(ts)

if __name__ == "__main__":
    start = timer()
    Main()
    print(timer() - start)
