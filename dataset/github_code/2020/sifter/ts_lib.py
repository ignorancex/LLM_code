"""Core library for describing triplet-structures in Python."""
import itertools
from collections import defaultdict

class TripletStructure:
    """Represents a triplet structure. Instances are usually named 'ts'.

    A TripletStructure starts out empty, with no nodes and no facts. It can be
    modified using the syntax: `ts["/:A"].map({ts["/:B"]: ts["/:C"]})` which
    adds the fact `(/:A,/:B,/:C)` to the structure. By default, nodes are added
    automatically upon first reference.

    We want to be able to easily roll-back changes to the TripletStructure. Every
    direct modification of the TripletStructure is automatically registered in the
    TSDelta instance @ts.buffer. This acts as a buffer of changes.  The method
    @ts.commit(...) will commit this buffer, i.e., save it to the end of the
    list @ts.path and replace @ts.buffer with a fresh TSDelta instance. You can
    always re-construct the structure by applying the TSDeltas in @ts.path
    successively to an empty structure, then applying @ts.buffer.  When
    @ts.buffer is empty, we say the structure is 'clean.' @ts.rollback(...) can
    be used to restore the state of the structure to a particular commit in
    @ts.path.

    Generally, every change to a triplet structure is owned by some TSDelta
    instance. @ts.path gives a list [None, delta_1, delta_2, ..., delta_n] of
    TSDeltas.
    """
    def __init__(self):
        """Initializes a new triplet structure."""
        # A list of the names of all nodes in the structure.
        self.nodes = []
        # Maps full_name -> short_name. The short name will be used in
        # user-facing printouts. We should maintain the invariant
        # self.display_names.keys() == self.nodes.
        self.display_names = dict()
        # self.facts a pre-computed index of the facts in the structure. Keys
        # are of two types:
        # 1. Triplet keys with 'holes' represented by None. E.g.,
        #    self.facts[(None, x, None)] is a list of all facts containing `x`
        #    in the middle slot. To get a list of all facts, use
        #    self.facts[(None, None, None)].
        # 2. Single-node keys. For a node string @x, self.facts[x] is all facts
        #    with x in at least one slot (see facts_about_node(...)).
        # Notably, if a fact (A, B, C) is in the structure at all, then it
        # *MUST* be belong to exactly the 11 keys returned by
        # self._iter_subfacts((A,B,C)).
        self.facts = defaultdict(list)
        # A prefix applied to node lookups. See ts.scope(...) and
        # ts.__getitem__.
        self.current_scope = "/"
        # The historical and running deltas.
        self.path = [None]
        self.buffer = TSDelta(self)
        # (Optional) an object with [add,remove]_[node,fact] methods which will
        # shadow changes to the structure. Used to implement efficient solving
        # with the C++ extensions.
        self.shadow = None

    def __getitem__(self, node):
        """Returns a (list of) NodeWrapper(s) corresponding to @node.

        This is the main entrypoint to manipulation of the structure.

        @node should be a string containing either (i) the name of a node, or
        (ii) a comma-separated list of node names. Node names should not
        contain spaces or commas.

        If a node name ends in ":??", then the "??" will be replaced with the
        smallest number such that the resulting node name does not yet exist in
        the structure.  Its use is somewhat analagous to LISP's gensym. See
        ts_utils.py for example usage.

        NOTE: ts[...] *CAN HAVE SIDE-EFFECTS*, namely _it constructs nodes
        which don't already exist_. You may think of it as a Python
        defaultdict. This makes for simpler code, but has a slight drawback of
        making typos harder to catch.  We may decide to change this syntax in
        the future, to something like ts.node(name) or tc(name), but: (i) the
        former would make quickly understanding 'tc-dense' code (like
        mapper.py) difficult while (ii) the latter loses intuition.
        """
        if "," in node:
            return [self[subname.strip()] for subname in node.split(",")]
        full_name = self._full_name(node)
        if full_name.endswith(":??"):
            for i in itertools.count():
                filled_name = "{}:{}".format(full_name[:-3], i)
                if filled_name not in self.nodes:
                    full_name = filled_name
                    break
        self.add_node(full_name)
        return NodeWrapper(self, full_name)

    def lookup(self, *template, read_direct=False):
        """Returns all facts according to a given template.

        This method should be called like ts.lookup(A,B,C) where A, B, C can be
        either node names or Nones. Nones match against any node name.

        Setting @read_direct=True returns a reference to the corresponding list
        of facts stored on the Structure instance. _May_ sometimes improve
        performance, but in general should be avoided due to unexpected
        behavior when either this class or the returned list is modified.
        """
        if not read_direct:
            return self.lookup(*template, read_direct=True).copy()
        return self.facts[template]

    def facts_about_node(self, full_name, read_direct=False):
        """Returns all facts involving the node with name @full_name.

        See self.lookup for nodes about @read_direct.
        """
        if not read_direct:
            return self.facts_about_node(full_name, read_direct=True).copy()
        return self.facts[full_name]

    def scope(self, scope="", protect=False):
        """Returns a TSScope representing the given scope.

        Often used like with ts.scope(...): ... to automatically prefix node
        names, e.g., to prevent name collisions.
        """
        return TSScope(self, self._full_name(scope), protect)

    def is_clean(self):
        """True iff the current buffer is empty."""
        return not self.buffer

    def commit(self, commit_if_clean=True):
        """Commits self.buffer to self.path."""
        if self.is_clean() and not commit_if_clean:
            return False
        self.path.append(self.buffer)
        self.buffer = TSDelta(self)
        return self.path[-1]

    def rollback(self, to_time=0):
        """Restores the structure to a previously-committed state.

        to_time = 0 means roll back the current buffer.
        to_time > 0 means roll back so that len(path) == to_time.
        to_time < 0 means roll back so that len(path) == len(path) - to_time.
        NOTE: In the final case, len(path) does *not* include the buffer.
        NOTE: len(path) == 0 is invalid, as path[0] = None (the 'root delta').
        """
        old_running = self.buffer
        self.buffer = TSDelta(self)
        old_running.rollback()
        self.buffer = TSDelta(self)
        if to_time == 0:
            return

        if to_time >= 0:
            target_length = to_time
        else:
            target_length = len(self.path) + to_time
        assert len(self.path) >= target_length > 0

        while len(self.path) > target_length:
            self.path.pop().rollback()
        # buffer will have a bunch of changes which aren't needed. In
        # theory we can 'disable' the TSDelta instead of just overwriting it
        # here, which might improve performance for some such operations.
        self._force_clean()

    def start_recording(self):
        """Returns a new TSRecording to track changes to @self."""
        return TSRecording(self)

    def freeze_frame(self):
        """Returns a new TSFreezeFrame saving the state of the structure."""
        return TSFreezeFrame(self)

    def has_node(self, full_name):
        """True iff @full_name is a registered node in the structure."""
        assert isinstance(full_name, str)
        return full_name in self.nodes

    def add_node(self, full_name, display_name=None):
        """Low-level method to add a node to the structure."""
        if not self.has_node(full_name):
            self.nodes.append(full_name)
            self.display_names[full_name] = display_name or full_name
            self.buffer.add_node(full_name)
            if self.shadow:
                self.shadow.add_node(full_name)

    def remove_node(self, full_name):
        """Low-level method to remove a node from the structure."""
        assert not self.facts_about_node(full_name, True), \
               f"Remove facts using {full_name} before removing it."
        if full_name in self.nodes:
            self.nodes.remove(full_name)
            self.display_names.pop(full_name)
            self.buffer.remove_node(full_name)
            if self.shadow:
                self.shadow.remove_node(full_name)

    def add_fact(self, fact):
        """Low-level method to add a fact to the structure."""
        if self.lookup(*fact, read_direct=True):
            # The fact already exists in the structure.
            return
        assert all(map(self.has_node, fact)), \
               f"Add all nodes in {fact} before adding the fact."
        for key in self._iter_subfacts(fact):
            self.facts[key].append(fact)
        self.buffer.add_fact(fact)
        if self.shadow:
            self.shadow.add_fact(fact)

    def remove_fact(self, fact):
        """Remove a fact from the structure."""
        if not self.lookup(*fact, read_direct=True):
            # Fact was already removed, or never added.
            return
        for key in self._iter_subfacts(fact):
            self.facts[key].remove(fact)
        self.buffer.remove_fact(fact)
        if self.shadow:
            self.shadow.remove_fact(fact)

    def add_nodes(self, nodes):
        """Helper to add multiple nodes to the structure."""
        for node in nodes:
            self.add_node(node)

    def remove_nodes(self, nodes):
        """Helper to remove multiple nodes from the structure."""
        for node in nodes:
            self.remove_node(node)

    def add_facts(self, facts):
        """Helper to add multiple facts to the structure."""
        for fact in facts:
            self.add_fact(fact)

    def remove_facts(self, facts):
        """Helper to remove multiple facts from the structure."""
        for fact in facts:
            self.remove_fact(fact)

    def print_delta(self):
        """Helper context that prints changes to the structure on exit."""
        class DeltaPrinter:
            """Helper context manager for printing changes to a structure."""
            def __init__(self, ts):
                self.ts = ts
                self.frame = None

            def __enter__(self):
                self.frame = self.ts.freeze_frame()

            def __exit__(self, t, v, tb):
                print(self.ts.freeze_frame() - self.frame)
        return DeltaPrinter(self)

    @staticmethod
    def _iter_subfacts(fact):
        """Yields all keys of self.facts which should hold @fact.

        This method *MUST* be used any time ts.facts is modified. For examples,
        see ts.add_fact, ts.remove_fact.
        """
        for subset in range(2**3):
            yield tuple(arg if (subset & (0b1 << i)) else None
                        for i, arg in enumerate(fact))
        for argument in sorted(set(fact)):
            yield argument

    def _full_name(self, name):
        """Returns the full name of a node relative to the current scope."""
        if name.startswith("/"):
            return name
        return "{}{}".format(self.current_scope, name)

    def _force_clean(self):
        """Manually clears the buffer.

        NOTE: Code outside of this file should **NEVER** call _force_clean.
        """
        self.buffer = TSDelta(self)

    def __str__(self):
        """Returns a string representation of the Structure.

        WARNING: This representation basically prints all the facts; it can get
        quite long, especially with a lot of rules.
        """
        def _format_fact(fact):
            return str(tuple(map(self.display_names.get, fact)))
        return "TripletStructure ({id}):\n\t{facts}".format(
            id=id(self), facts="\n\t".join(
                map(_format_fact, self.lookup(None, None, None))))

class TSScope:
    """Represents a scope (node name prefix) in a particular structure.

    Often used indirectly as in with ts.scope("..."): ... to automatically
    prefix node names, but also has some useful methods for using directly (eg.
    listing all nodes with a certain prefix).
    """
    def __init__(self, structure, prefix, protect=False):
        """Initializes a new TSScope.

        This should usually only be called via ts.scope(...) or
        scope.scope(...).
        """
        self.structure = structure
        self.prefix = prefix
        # Keeps track of the prefix on the structure before __enter__ so we can
        # reset it upon __exit__.
        self.old_scope_stack = []
        self.protect = protect

    def __enter__(self):
        """Instructs the Structure to prefix nodes with self.prefix by default.

        Returns the TSScope instance for convenience.
        """
        assert not self.protect
        self.old_scope_stack.append(self.structure.current_scope)
        self.structure.current_scope = self.prefix
        return self

    def __exit__(self, type_, value, traceback):
        """Resets the Structure's default prefix.
        """
        assert not self.protect
        self.structure.current_scope = self.old_scope_stack.pop()

    def __getitem__(self, index):
        """Get node relative to self regardless of the structure's prefix.
        """
        if self.protect:
            return "{}{}".format(self.prefix, index)
        with self:
            return self.structure[index]

    def scope(self, scope):
        """Get a sub-scope relative to self regardless of structure's prefix.
        """
        with self:
            return self.structure.scope(scope, self.protect)

    def protected(self):
        """Returns a protected version of the scope.

        In a protected scope, doing scope[name] will return the full node name
        as a string instead of a NodeWrapper, and will *NOT* add the node if it
        does not exist.
        """
        return self.structure.scope(self.prefix, True)

    def __iter__(self):
        """Iterator for all nodes in the structure within the scope.
        """
        for member_name in self.structure.nodes:
            if member_name.startswith(self.prefix + ":"):
                yield self.structure[member_name]

    def __contains__(self, node):
        """True iff @node is a member of the scope.
        """
        if isinstance(node, NodeWrapper):
            assert node.structure == self.structure
            node = node.full_name
        return node.startswith(self.prefix + ":")

    def __len__(self):
        """Returns the number of nodes in the structure within the scope.
        """
        return sum(node.startswith(self.prefix + ":")
                   for node in self.structure.nodes)

class NodeWrapper:
    """Represents a single node in a given structure."""
    def __init__(self, structure, full_name):
        """Initialize the NodeWrapper."""
        self.structure = structure
        self.full_name = full_name

    def map(self, mappings):
        """Helper for adding facts to the structure.

        node.map({A: B, C: D}) adds (node, A, B) and (node, C, D).

        NOTE: Be wary of repeated keys!
        """
        def to_fact(value_node, key_node):
            return (self.full_name, value_node.full_name, key_node.full_name)
        facts = []
        for value, key in mappings.items():
            if isinstance(key, NodeWrapper):
                facts.append(to_fact(value, key))
            else:
                # Allow sets of keys
                facts.extend(to_fact(value, sub_key) for sub_key in key)

        # We sort here to ensure it's deterministic.
        self.structure.add_facts(sorted(facts))

    def scoped_name(self, scope):
        """Returns string @x such that @scope[@x] = @self.

        This is the "first name" where @scope is the "last name." Used, for
        example, by ts_utils to find rules that should be marked /= or /MAYBE=
        in rules based on their name.
        """
        if not self.full_name.startswith(scope.prefix):
            return self.full_name
        return self.full_name[len(scope.prefix):]

    def __sub__(self, scope):
        """Syntactic sugar for scoped_name(...)."""
        return self.scoped_name(scope)

    def remove_with_facts(self):
        """Removes the node and all associated facts from the structure."""
        self.structure.remove_facts(
            self.structure.facts_about_node(self.full_name))
        self.structure.remove_node(self.full_name)

    def remove(self):
        """Removes the node (without associated facts) from the structure.

        This is equivalent to assert not facts_about_node; remove_with_facts().
        It should be used when there is an invariant that no related facts
        should exist in the structure. See runtime/assignment.py for an
        example.
        """
        self.structure.remove_node(self.full_name)

    def display_name(self, set_to=None):
        """Gets or sets the display name of the node."""
        if set_to is not None:
            self.structure.display_names[self.full_name] = set_to
        return self.structure.display_names[self.full_name]

    def __eq__(self, other):
        """True iff @self and @other refer to the same node."""
        return ((self.structure, self.full_name) ==
                (other.structure, other.full_name))

    def __hash__(self):
        """Hash based on the structure and name of the node."""
        return hash((self.structure, self.full_name))

    def __lt__(self, other):
        """Lexicographical comparison for sorting."""
        return ((id(self.structure), self.full_name)
                < (id(other.structure), other.full_name))

    def __str__(self):
        """Returns the name of the node."""
        return self.full_name

class TSDelta:
    """Represents the change between two TripletStructures."""
    def __init__(self, ts):
        """Initialize a TSDelta."""
        self.ts = ts
        self.add_nodes, self.add_facts = set(), set()
        self.remove_nodes, self.remove_facts = set(), set()

    def apply(self):
        """Apply the TSDelta to self.ts."""
        assert self is not self.ts.buffer
        assert self.ts.is_clean()
        # NOTE: Sorted here is just for determinism.
        self.ts.add_nodes(sorted(self.add_nodes))
        self.ts.add_facts(sorted(self.add_facts))
        self.ts.remove_facts(sorted(self.remove_facts))
        self.ts.remove_nodes(sorted(self.remove_nodes))
        self.ts._force_clean()
        # TODO: maybe this should just wrap it?
        self.ts.path.append(self)

    def rollback(self):
        """Undo the TSDelta."""
        assert self is not self.ts.buffer
        # NOTE: Sorted here is just for determinism.
        self.ts.remove_facts(sorted(self.add_facts))
        self.ts.remove_nodes(sorted(self.add_nodes))
        self.ts.add_nodes(sorted(self.remove_nodes))
        self.ts.add_facts(sorted(self.remove_facts))
        # Maybe we should assert that this is at the end of the path and remove
        # it?

    def add_node(self, full_name):
        """Record the addition of a new node."""
        self.add_nodes.add(full_name)

    def add_fact(self, fact):
        """Record the addition of a new fact."""
        self.add_facts.add(fact)

    def remove_node(self, full_name):
        """Record the removal of an existing node."""
        try:
            self.add_nodes.remove(full_name)
        except KeyError:
            self.remove_nodes.add(full_name)

    def remove_fact(self, fact):
        """Record the removal of an existing fact."""
        try:
            self.add_facts.remove(fact)
        except KeyError:
            self.remove_facts.add(fact)

    def __bool__(self):
        """True iff the TSDelta is not a no-op."""
        return (bool(self.add_nodes) or bool(self.add_facts) or
                bool(self.remove_nodes) or bool(self.remove_facts))

    def __str__(self):
        """Human-readable format of the TSDelta. """
        def _format(list_):
            if list_ and isinstance(sorted(list_)[0], tuple):
                list_ = [tuple(map(lambda x: self.ts.display_names.get(x, x),
                                   map(str, el))) for el in sorted(list_)]
            return "\n\t\t" + "\n\t\t".join(map(str, list_))
        return ("TSDelta ({id}):"
                "\n\t- Nodes: {remove_nodes}" +
                "\n\t+ Nodes: {add_nodes}" +
                "\n\t- Facts: {remove_facts}" +
                "\n\t+ Facts: {add_facts}").format(
                    id=id(self),
                    remove_nodes=_format(self.remove_nodes),
                    add_nodes=_format(self.add_nodes),
                    remove_facts=_format(self.remove_facts),
                    add_facts=_format(self.add_facts))

class TSRecording:
    """Helper class representing all TSDeltas applied after some checkpoint."""
    def __init__(self, ts):
        """Initialize the TSRecording."""
        self.ts = ts
        self.start_path = self.ts.path.copy()
        assert self.ts.is_clean()

    def commits(self, rollback=False):
        """TSDeltas applied to the structure since @self was initialized.

        If rollback=True, it will also roll back the state of the structure to
        when @self was initialized.
        """
        assert (self.ts.path[:len(self.start_path)] == self.start_path
                and self.ts.is_clean())
        deltas = self.ts.path[len(self.start_path):]
        if rollback:
            self.rollback()
        return deltas

    def rollback(self):
        """Roll back the state of the structure to when @self was initialized.
        """
        assert (self.ts.path[:len(self.start_path)] == self.start_path
                and self.ts.is_clean())
        self.ts.rollback(len(self.start_path))

class TSFreezeFrame:
    """Represents a snapshot of a TripletStructure."""
    def __init__(self, ts):
        """Initialize the TSFreezeFrame."""
        self.ts = ts
        self.nodes = set(ts.nodes)
        self.facts = set(ts.lookup(None, None, None))

    def delta_to_reach(self, desired, nodes=True, facts=True):
        """Return a TSDelta, applying which transforms @self to @desired."""
        delta = TSDelta(self.ts)
        if nodes:
            delta.add_nodes = desired.nodes - self.nodes
            delta.remove_nodes = self.nodes - desired.nodes
        if facts:
            delta.add_facts = desired.facts - self.facts
            delta.remove_facts = self.facts - desired.facts
        return delta

    def __sub__(self, other):
        """Syntactic sugar for delta_to_reach."""
        return other.delta_to_reach(self)

    def __eq__(self, other):
        """True iff @self and @other represent the same structure state."""
        return (self.ts == other.ts
                and self.nodes == other.nodes
                and self.facts == other.facts)
