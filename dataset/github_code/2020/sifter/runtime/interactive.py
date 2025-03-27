"""An interactive front-end for Triplet Structures.

Call from runtime.py:interactive(...).
"""
# pylint: disable=import-error,no-name-in-module
from runtime.shadow_input import ShadowInput

class TSREPL:
    """An interactive interface for the TSRuntime.
    """
    def __init__(self, runtime):
        """Initializes a new TSREPL.
        """
        self.runtime = runtime
        self.ts = runtime.ts
        self.input = ShadowInput()

    def run(self):
        """Run the REPL.
        """
        while True:
            if self.iteration() == "EXIT":
                return

    def iteration(self):
        """Prompt the user for an action and execute it.
        """
        print("Starting a new iteration. Please pick an option:")
        option = self.select_one([
            "Manually apply individual rules.",
            "Automatically apply all rules matching pattern.",
            "Print structure.",
            "Begin recording command sequence.",
            "Save command sequence.",
            "Load command sequence.",
            "Set display name.",
            "Prefer nodes.",
            "Exit interactive session.",
        ])
        if option == 0:
            self.apply_manual()
        elif option == 1:
            self.to_fixedpoint()
        elif option == 2:
            print(self.ts)
        elif option == 3:
            self.input.begin_recording()
        elif option == 4:
            file_name = self.input("Save to file: ")
            self.input.scrub_last(2)
            self.input.save_recording(file_name)
        elif option == 5:
            file_name = self.input("Load from file: ")
            self.input.load_recording(file_name)
        elif option == 6:
            self.set_display_name()
        elif option == 7:
            self.prefer_nodes()
        elif option == 8:
            return "EXIT"
        else:
            raise NotImplementedError
        return None

    def to_fixedpoint(self):
        """Apply rule(s) until fixedpoint is reached.
        """
        rules = self.pick_rules()
        search_term = self.input("Search proposals for term: ").lower()
        n_matches = int(self.input("Apply if at least this many matches: "))
        fixedpoint = False
        while not fixedpoint:
            fixedpoint = True
            for _, delta in self.runtime.propose_all(rules):
                if str(delta).lower().count(search_term) >= n_matches:
                    print("Applying proposal:")
                    print(delta)
                    delta.apply()
                    fixedpoint = False
                    break

    def apply_manual(self):
        """Manually apply rule(s).
        """
        rules = self.pick_rules()
        search_term = self.input("Filter proposals for term: ").lower()
        n_matches = int(self.input("With at least this many matches: ") or "0")
        while True:
            for _, delta in self.runtime.propose_all(rules):
                if str(delta).lower().count(search_term) < n_matches:
                    continue
                print(delta)
                choice = self.input("Apply? (y/N/q): ").lower() or "n"
                if choice[0] == "y":
                    delta.apply()
                    break
                if choice[0] == "q":
                    return
            else:
                return

    def set_display_name(self):
        """Allows the user to set a display name for a particular node.
        """
        search_term = self.input("Existing display name: ")
        matching = [
            node for node, display_name in self.ts.display_names.items()
            if search_term in display_name]
        if matching:
            node_name = min(
                matching, key=lambda node: len(self.ts.display_names[node]))
            new_name = self.input("Please select a new display name: ")
            self.ts[node_name].display_name(new_name)
        else:
            print("No matching node.")

    def prefer_nodes(self):
        """Tell the runtime to prefer particular nodes in matches.
        """
        search_term = self.input("Prefer nodes containing string: ")
        self.runtime.affinity.prefer_nodes(lambda node: search_term in node)

    def pick_rules(self):
        """Helper to allow the user to select rules to apply.
        """
        names = [rule.name for rule in self.runtime.rules]
        print("Please select the rule(s):")
        indices = self.select_multiple(names)
        return [names[i] for i in indices]


    def select_one(self, options):
        """Helper method for selecting an option.
        """
        for i, option in enumerate(options):
            print("{}: {}".format(i, option))
        index = self.input("Selection: ").lower()
        try:
            option = options[int(index)]
            return int(index)
        except (ValueError, IndexError):
            try:
                # Treat it as a search term
                return next(i for i, option in enumerate(options)
                            if index in str(option).lower())
            except StopIteration:
                print("Invalid choice, try again.")
                return self.select_one(options)

    def select_multiple(self, options):
        """Helper method for selecting multiple options.
        """
        for i, option in enumerate(options):
            print("{}: {}".format(i, option))
        term = self.input("Selections: ").lower()
        try:
            if "-" in term:
                from_index, to_index = map(int, term.split("-")[:2])
            else:
                from_index, to_index = int(term), int(term)
            return list(range(from_index, to_index + 1))
        except ValueError:
            return [i for i, option in enumerate(options)
                    if term in str(option).lower()]
