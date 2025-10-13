import time
from collections import defaultdict, deque
from copy import deepcopy
from typing import List, Set, Tuple, Optional, Union, Dict

from packages.bnf.parser import BNFParser
from packages.utils.log import logger

# Type aliases
ShiftAction = Tuple[str, int]  # ('shift', target_state)
ReduceAction = Tuple[str, Tuple[str, ...]]  # ('reduce', (lhs, rhs))
AcceptAction = Tuple[str,]  # ('accept',)
Action = Union[ShiftAction, ReduceAction, AcceptAction]
Item = Tuple[str, Tuple[str, ...], int, str]  # (lhs, rhs, dot, lookahead)
ItemSet = Set[Item]


class GLRParser:
    """
    A GLR (Generalized LR) Parser that can parse input strings based on a grammar defined in BNF.

    Note: This grammar is not optimized for performance, just use for research purposes, and not guaranteed to work for all CFG grammars which is not the purpose of this parser.
    At the most time, the generated BNFs should be handled by the parser and this parser is enough to handle them, otherwise, the generated BNFs may be wrong, too complex, or too ambiguous.
    This parser is also used to check the validity of the BNFs with examples dataset and the parser can handle all of them.
    """

    def __init__(self, bnf_parser: BNFParser):
        self.bnf_parser = bnf_parser
        self.rules = deepcopy(bnf_parser.get_rules())  # Deep copy to avoid modifying the original
        self.start_symbol = self.bnf_parser.get_start_symbol()
        self.terminals = self.bnf_parser.get_terminals()
        self.non_terminals = self.bnf_parser.get_non_terminals()
        self.augmented_start = f"<{self.start_symbol}_aug>"  # S'
        self.rules[self.augmented_start] = [[self.start_symbol]]  # Add augmented start rule
        self.item_set_collection: List[ItemSet] = []  # List of sets of items
        self.ACTION_TABLE: Dict[int, Dict[str, List[Action]]] = defaultdict(lambda: defaultdict(list))
        self.GOTO_TABLE: Dict[int, Dict[str, int]] = defaultdict(dict)
        logger.info("Building parsing table...")
        self.build_parsing_table()
        logger.info("Parsing table built successfully.")

    def build_parsing_table(self):
        """
        Constructs the ACTION and GOTO tables for the GLR parser.
        """
        # Timeout after 60 minutes, this is to handle not-halting problem for the grammar this parser can not handle, and it is not reasonable for it to run more than 60 seconds.
        start_time = time.time()
        timeout_duration = 60  # secs

        # Initialize the canonical collection with the closure of the augmented start
        initial_item = (self.augmented_start, tuple(self.rules[self.augmented_start][0]), 0, '$')
        initial_closure = self.closure({initial_item})
        self.item_set_collection = [initial_closure]
        queue = deque([0])

        logger.info("Building canonical collection...")
        while queue:
            if time.time() - start_time > timeout_duration:
                raise TimeoutError("Timeout while building canonical collection.")
            state = queue.popleft()
            logger.info(f"Processing state {state}...")
            current_items = self.item_set_collection[state]
            symbols = self.get_symbols_after_dot(current_items)
            for symbol in symbols:
                goto_set = self.goto(current_items, symbol)
                if not goto_set:
                    continue
                existing_state = self.find_state(goto_set)
                if existing_state is None:
                    self.item_set_collection.append(goto_set)
                    target_state = len(self.item_set_collection) - 1
                    queue.append(target_state)
                else:
                    target_state = existing_state
                if symbol in self.terminals:
                    self.ACTION_TABLE[state][symbol].append(('shift', target_state))
                elif symbol in self.non_terminals:
                    self.GOTO_TABLE[state][symbol] = target_state

        logger.info("Building ACTION table...")
        for state, item_set in enumerate(self.item_set_collection):
            for item in item_set:
                lhs, rhs, dot, lookahead = item
                if dot == len(rhs):
                    if lhs == self.augmented_start:
                        self.ACTION_TABLE[state]['$'].append(('accept',))
                    else:
                        self.ACTION_TABLE[state][lookahead].append(('reduce', (lhs, rhs)))

    def find_state(self, goto_set: Set[Tuple[str, Tuple[str, ...], int, str]]) -> Optional[int]:
        """
        Checks if a given set of items already exists in the collection and returns its state number.
        """
        for idx, item_set in enumerate(self.item_set_collection):
            if item_set == goto_set:
                return idx
        return None

    def closure(self, items: ItemSet) -> ItemSet:
        """
        Computes the closure
        """
        # Time out after 60 seconds
        start_time = time.time()
        timeout_duration = 60
        # Deep copy to avoid modifying the original
        closure_set = deepcopy(items)
        added = True
        while added:
            if time.time() - start_time > timeout_duration:
                raise TimeoutError("Timeout while computing closure.")
            added = False
            new_items = set()
            for (lhs, rhs, dot, lookahead) in closure_set:
                if dot < len(rhs):
                    symbol = rhs[dot]
                    if symbol in self.non_terminals:
                        beta = rhs[dot + 1:]
                        beta_lookahead = self.first(list(beta) + [lookahead])
                        for alternative in self.rules[symbol]:
                            for la in beta_lookahead:
                                if la == '':
                                    la = '$'  # Replace empty string with end symbol
                                # Handle empty alternatives by representing them as empty tuples
                                if not alternative or (len(alternative) == 1 and alternative[0] == ''):
                                    item = (symbol, (), 0, la)
                                else:
                                    item = (symbol, tuple(alternative), 0, la)
                                if item not in closure_set and item not in new_items:
                                    new_items.add(item)
            if new_items:
                closure_set.update(new_items)
                added = True
        return closure_set

    def goto(self, items: ItemSet, symbol: str) -> Optional[ItemSet]:
        """
        Computes the GOTO set for a set of items and a grammar symbol.
        """
        goto_set = set()
        for (lhs, rhs, dot, lookahead) in items:
            if dot < len(rhs) and rhs[dot] == symbol:
                goto_set.add((lhs, rhs, dot + 1, lookahead))
        return self.closure(goto_set) if goto_set else None

    def get_symbols_after_dot(self, items: ItemSet) -> Set[str]:
        """
        Retrieves all grammar symbols that appear immediately after the dot in the given items.
        """
        symbols = set()
        for (_, rhs, dot, _) in items:
            if dot < len(rhs):
                symbols.add(rhs[dot])
        return symbols

    def first(self, symbols: List[str], visited: Optional[Set[str]] = None) -> Set[str]:
        """
        Computes the FIRST set for a sequence of grammar symbols.
        """
        if visited is None:
            visited = set()
        if not symbols:
            return {'$'}
        first_set = set()
        for symbol in symbols:
            if symbol in self.terminals:
                first_set.add(symbol)
                return first_set
            elif symbol in self.non_terminals:
                if symbol in visited:
                    continue  # Prevent infinite recursion
                visited.add(symbol)
                for prod in self.rules[symbol]:
                    if not prod or (len(prod) == 1 and prod[0] == ''):  # Empty production
                        first_set.add('')
                        continue
                    first_set.update(self.first([prod[0]], visited.copy()))
                if '' not in first_set:
                    break
                else:
                    first_set.remove('')
            else:
                pass
        else:
            first_set.add('$')
        return first_set

    def accepts_input(self, input_string: str,return_used_prs:bool=False) -> Union[bool,Tuple[bool,Set[Tuple[str, Tuple[str, ...]]]]]:
        """
        Parses the input string and returns True if the grammar can accept it, False otherwise.
        :param input_string: The input string to parse.
        :return: True if the input string is accepted by the grammar, False otherwise.
        """
        # Timeout after 60 seconds: this is to handle infinite loops that may happen for few grammars if the grammar is very ambiguous and can not be handled by this parser
        start_time = time.time()
        timeout_duration = 60
        # Tokenize the input string
        tokens = self.tokenize_input(input_string) + ['$']
        # Initialize with a single parser state: stack and pointer
        initial_parser_state = {
            'stack': [0],
            'pointer': 0,
            'productions': []
        }
        parser_states = [initial_parser_state]

        while parser_states:
            if time.time() - start_time > timeout_duration:
                raise TimeoutError("Timeout while parsing input.")
            current_states = parser_states
            parser_states = []
            for state in current_states:
                stack = state['stack']
                pointer = state['pointer']
                productions = state['productions']
                if pointer >= len(tokens):
                    continue  # No more tokens to process
                current_token = tokens[pointer]
                current_state = stack[-1]
                actions = self.ACTION_TABLE.get(current_state, {}).get(current_token, [])
                if not actions:
                    continue  # No action possible, discard this parser state
                for action in actions:
                    action_type = action[0]
                    if action_type == 'shift':
                        target_state = action[1]
                        new_stack = stack + [target_state]
                        new_pointer = pointer + 1
                        parser_states.append({
                            'stack': new_stack,
                            'pointer': new_pointer,
                            'productions': productions[:]
                        })
                    elif action_type == 'reduce':
                        lhs, rhs = action[1]
                        new_stack = stack.copy()
                        if rhs:
                            for _ in rhs:
                                if not new_stack:
                                    break
                                new_stack.pop()
                        if not new_stack:
                            continue  # Invalid reduction
                        goto_state = self.GOTO_TABLE[new_stack[-1]].get(lhs, None)
                        if goto_state is not None:
                            new_stack.append(goto_state)
                            new_productions = productions[:]
                            new_productions.append((lhs, rhs))
                            parser_states.append({
                                'stack': new_stack,
                                'pointer': pointer,
                                'productions': new_productions
                            })
                    elif action_type == 'accept':
                        if return_used_prs:
                            return True,set(productions)
                        return True
            # Remove duplicate parser states to optimize
            unique_parser_states = []
            seen = set()
            for ps in parser_states:
                key = (tuple(ps['stack']), ps['pointer'], tuple(ps['productions']))
                if key not in seen:
                    seen.add(key)
                    unique_parser_states.append(ps)
            parser_states = unique_parser_states
        if return_used_prs:
            return False,set()
        return False

    def tokenize_input(self, input_string: str) -> List[str]:
        """
        Tokenizes the input string based on the grammar's terminals (match the longest terminals first).
        :param input_string: The input string to tokenize.
        :return: A list of tokens.
        """
        tokens = []
        i = 0
        while i < len(input_string):
            match = None
            # Sort terminals by length in descending order to match the longest possible terminals first
            for terminal in sorted(self.terminals, key=lambda x: -len(x)):
                if terminal == '':
                    continue  # Skip empty terminals
                if input_string.startswith(terminal, i):
                    tokens.append(terminal)
                    i += len(terminal)
                    match = True
                    break
            if not match:
                # Do not skip spaces; treat them as separate tokens
                tokens.append(input_string[i])
                i += 1
        return tokens

    def display_tables(self):
        """
        Displays the ACTION and GOTO tables.
        """
        print("ACTION TABLE:")
        for state in sorted(self.ACTION_TABLE.keys()):
            actions = self.ACTION_TABLE[state]
            print(f"State {state}:")
            for symbol, action_list in actions.items():
                actions_str = ', '.join([str(action) for action in action_list])
                print(f"  {symbol}: {actions_str}")
        print("\nGOTO TABLE:")
        for state in sorted(self.GOTO_TABLE.keys()):
            gotos = self.GOTO_TABLE[state]
            print(f"State {state}:")
            for symbol, target in gotos.items():
                print(f"  {symbol}: {target}")

    def print_items(self):
        """
        Prints all the items (states) in the canonical collection.
        """
        for idx, item_set in enumerate(self.item_set_collection):
            print(f"State {idx}:")
            for item in item_set:
                lhs, rhs, dot, lookahead = item
                rhs_with_dot = list(rhs)
                if dot < len(rhs_with_dot):
                    rhs_with_dot.insert(dot, '•')
                else:
                    rhs_with_dot.append('•')
                rhs_str = ' '.join(rhs_with_dot)
                print(f"  {lhs} ::= {rhs_str} , {lookahead}")
            print()
