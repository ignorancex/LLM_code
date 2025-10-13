from typing import Set
from typing_extensions import deprecated

from automata.pda.npda import NPDA, NPDAStateT

from packages.bnf.parser import BNFParser


def _symbol_to_state(symbol: str) -> str:
    """
    Convert a symbol to its corresponding state name in NPDA
    :param symbol:
    :return:
    """
    return f"q_{symbol}"


@deprecated('Deprecated since NPDA is deprecated.')
def convert_bnf_to_npda(bnf: BNFParser, start_non_terminal: str) -> NPDA:
    """
    Given a BNF grammar and a non_terminal to start, convert it to an equivalent NPDA.
    :param bnf: BNFParser
    :param start_non_terminal: a non-terminal to start
    :return: NPDA
    """
    initial_state = "q_start"
    accept_state = "q_accept"
    states: Set[NPDAStateT] = set()
    initial_stack_symbol: str = '#'
    stack_symbols: Set[str] = set()
    input_symbols: Set[str] = bnf.get_terminals()
    transitions = dict()

    # Initial states
    states.add(initial_state)
    states.add(accept_state)
    for non_terminal in bnf.get_non_terminals():
        states.add(_symbol_to_state(non_terminal))

    # Initial stack symbol
    stack_symbols.add(initial_stack_symbol)
    stack_symbols.update(bnf.get_non_terminals())
    stack_symbols.update(bnf.get_terminals())

    # Define initial transition
    transitions[initial_state] = {
        '': {
            initial_stack_symbol: {(_symbol_to_state(start_non_terminal), (start_non_terminal, initial_stack_symbol))}
        }
    }

    # Define transitions for non-terminals
    for non_terminal, productions in bnf._rules.items():
        # Define transitions to accept state
        (transitions
         .setdefault(_symbol_to_state(non_terminal), {})
         .setdefault('', {})
         .setdefault(initial_stack_symbol, set()).
         add((accept_state, ''))
         )
        # Define transitions for each production
        for production in productions:
            # Define expand productions
            (transitions
             .setdefault(_symbol_to_state(non_terminal), {})
             .setdefault('', {})
             .setdefault(non_terminal, set())
             .add((_symbol_to_state(non_terminal), tuple(production))))
            # Define transitions for each symbol in production
            for symbol in production:
                if BNFParser.is_terminal(symbol): # Terminal: consume input
                    (transitions
                     .setdefault(_symbol_to_state(non_terminal), {})
                     .setdefault(symbol, {})
                     .setdefault(symbol, set())
                     .add((_symbol_to_state(non_terminal), '')))
                else: # Non-terminal: transition to this non-terminal symbol state
                    if non_terminal != symbol:
                        (transitions
                         .setdefault(_symbol_to_state(non_terminal), {})
                         .setdefault('', {})
                         .setdefault(symbol, set())
                         .add((_symbol_to_state(symbol), (symbol, initial_stack_symbol))))

                        # add epsilon transition to the original non-terminal state
                        (transitions
                         .setdefault(_symbol_to_state(symbol), {})
                         .setdefault('', {})
                         .setdefault(initial_stack_symbol, set())
                         .add((_symbol_to_state(non_terminal), '')))

    # Construct NPDA
    npda = NPDA(
        states=states,
        input_symbols=input_symbols,
        stack_symbols=stack_symbols,
        transitions=transitions,
        initial_state=initial_state,
        initial_stack_symbol=initial_stack_symbol,
        final_states={accept_state},
        acceptance_mode='empty_stack'
    )
    return npda
