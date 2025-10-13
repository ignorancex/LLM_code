import re
from copy import deepcopy
from typing import TypeAlias
from typing_extensions import deprecated

from packages.utils.string_tool import split_string_with_escapes, unescape_string, escape_string

Rules: TypeAlias = dict[str, list[list[str]]]


class BNFParser:
    """
    A parser for Backus-Naur Form (BNF) grammars.
    Note: This parser normally doesn't handle comments so comments should be removed first
    """

    def __init__(self, grammar_text: str):
        self._rules: Rules = {}
        self._grammar_text = grammar_text
        self.parse_grammar(grammar_text)

    def parse_grammar(self, grammar_text: str) -> None:
        """
        Parse a BNF grammar.
        :param grammar_text: BNF grammar text
        """
        lines = grammar_text.splitlines()
        current_lhs: str | None = None
        for l_idx, line in enumerate(lines):
            l_num = l_idx + 1  # line number should start from 1
            line = line.strip()
            # Skip empty lines and comments
            if not line:
                continue
            # Match lines starting with a non-terminal definition
            if re.match(r'<[^<>]+>\s*::=', line):
                lhs, rhs = self.split_rule(l_num, line)
                current_lhs = lhs
                self.add_rule(l_num, line, lhs, rhs)
            # Match lines starting with '|'
            elif line.startswith('|'):
                if current_lhs is None:
                    error_msg = (fr"""
Syntax error, "Alternation without a left-hand side", happened at line {l_num}: {line}.
This error is likely due to not satisfying: A rule MUST start with a non-terminal definition.
                    """).strip()
                    raise SyntaxError(error_msg)
                rhs = line.lstrip('|').strip()
                self.add_rule(l_num, line, current_lhs, rhs)
            else:
                error_msg = (fr"""
Syntax error happened at line {l_num}: "{line}".
This error is likely due to not satisfying one of the following requirements:
1. A rule MUST start with a non-terminal definition;
2. A non-terminal symbol MUST be in angle brackets, e.g. <non-terminal>;
3. A non-terminal definition must be followed by '::=' to indicate the start of the right-hand side;
                """.strip())
                raise SyntaxError(error_msg)

    @staticmethod
    def split_rule(l_num: int, line: str) -> tuple[str, str]:
        """
        Split a rule into left-hand side and right-hand side.
        Left-hand side is a non-terminal and right-hand side consists of alternatives.
        :param l_num: The line number.
        :param line: The line containing the rule.
        :return: A tuple containing the left-hand side and right-hand side.
        """
        match = re.match(r'(<[^<>]+>)\s*::=\s*(.+)', line)
        if not match:
            error_msg = fr"""
Syntax error happened at line {l_num}: {line}.
This error is likely due to the reason that the right-hand side is not defined after '::='.
            """.strip()
            raise SyntaxError(error_msg)
        lhs = match.group(1).strip()
        rhs = match.group(2).strip()
        return lhs, rhs

    def add_rule(self, l_num: int, line: str, lhs: str, rhs: str) -> None:
        """
        Add a rule to the rules' dictionary.
        :param l_num: The line number.
        :param line: The line containing the rule.
        :param lhs: Left-hand side of the rule.
        :param rhs: Right-hand side of the rule.
        """
        alternatives = self.split_alternatives(l_num, line, rhs)
        existing_alternatives = self._rules.setdefault(lhs, [])
        for alternative in alternatives:
            if alternative not in existing_alternatives:
                existing_alternatives.append(alternative)

    def split_alternatives(self, l_num: int, line: str, rhs: str) -> list[list[str]]:
        """
        Split a right-hand side into a list of alternatives.
        :param l_num: The line number.
        :param line: The line containing the rule.
        :param rhs: Right-hand side of the rule.
        :return: A list of alternatives, each alternative is a list of symbols/tokens.
        """
        alternatives = []
        current_alternative = ''
        in_quote = False
        in_angle = False
        i = 0
        while i < len(rhs):
            c = rhs[i]
            if c == '"':
                in_quote = not in_quote
                current_alternative += c
                i += 1
            elif c == '<':
                if not in_quote:
                    in_angle = True
                current_alternative += c
                i += 1
            elif c == '>':
                if not in_quote:
                    in_angle = False
                current_alternative += c
                i += 1
            elif c == '|' and not in_quote and not in_angle:
                # Split here
                alternatives.append(current_alternative.strip())
                current_alternative = ''
                i += 1
            else:
                current_alternative += c
                i += 1
        if current_alternative.strip():
            alternatives.append(current_alternative.strip())
        # Now tokenize each alternative
        alternatives = [self.tokenize(l_num, line, alternative) for alternative in alternatives]
        return alternatives

    @staticmethod
    def tokenize(l_num: int, line: str, expression: str) -> list[str]:
        """
        Tokenize an expression into terminals and non-terminals.
        :param l_num: The line number.
        :param line: The line containing the rule.
        :param expression: The expression to tokenize.
        :return: A list of tokens.
        """
        tokens = re.findall(r'<[^<>]+>|"[^"]+"|\S+', expression)
        processed_tokens = []
        for token in tokens:
            if token.startswith('"') and token.endswith('"'):
                # Remove quotes for terminals
                try:
                    symbol = token[1:-1]
                    if escape_string(symbol).startswith(r'\u'):  # do not escape other special unicode characters like: ε
                        processed_tokens.append(symbol)
                    else:  # unescape characters like '\n'
                        processed_tokens.append(unescape_string(symbol))
                except:
                    error_msg = (fr'''
Syntax error happened at line {l_num}: {line}.
This error is likely due to the invalid escape of quotes in the terminals which means whenever you want to introduce a terminal with a single or multiple quotes, you should not put slash before them.  
For example:
1. "\"" is invalid, but """ is valid.
2. "\"\"" is invalid, but """" is valid.
3. "\"\"\"" is invalid, but """"" is valid.
                    ''').strip()
                    raise SyntaxError(error_msg)
            elif token.startswith('<') and token.endswith('>'):
                # Non-terminals are valid as is
                processed_tokens.append(token)
            else:
                # If a token is not enclosed in quotes or angle brackets, raise an error
                error_msg = (fr'''
Syntax error happened at line {l_num}: {line}.
This error is due to the terminals not being enclosed in double quotes.
For example
1. "a" is a valid terminal, but a is invalid.
2. "a" is a valid terminal, but 'a' is invalid.
                '''
                             ).strip()
                raise SyntaxError(error_msg)
        return processed_tokens

    def get_rules(self) -> Rules:
        """
        Get the parsed rules.
        :return: A dictionary representing the grammar rules.
        """
        return self._rules

    def pretty_print(self):
        """
        Pretty print the grammar rules.
        """
        for lhs, alternatives in self._rules.items():
            print(f"{lhs} :")
            for alternative in alternatives:
                print(f"  {' '.join(alternative)}")

    def get_non_terminals(self) -> set[str]:
        """
        Get the non-terminals in the grammar.
        :return: A list of non-terminals.
        """
        return set(self._rules.keys())

    def get_non_terminals_for_a_rule(self, rule: str) -> set[str]:
        """
        Get the non-terminals in a rule.
        :return: A list of non-terminals.
        """
        non_terminals = set()
        for alternative in self._rules[rule]:
            for token in alternative:
                if token.startswith('<') and token.endswith('>'):
                    non_terminals.add(token)
        return non_terminals

    def get_terminals(self) -> set[str]:
        """
        Get the terminals in the grammar.
        :return: A list of terminals.
        """
        terminals = set()
        for alternatives in self._rules.values():
            for alternative in alternatives:
                for token in alternative:
                    if not (token.startswith('<') and token.endswith('>')):
                        terminals.add(token)
        return terminals

    def get_terminals_for_a_rule(self, rule: str) -> set[str]:
        """
        Get the terminals in a rule.
        :return: A list of terminals.
        """
        terminals = set()
        for alternative in self._rules[rule]:
            for token in alternative:
                if not (token.startswith('<') and token.endswith('>')):
                    terminals.add(token)
        return terminals

    def get_alternatives_for_a_rule(self, non_terminal: str) -> list[list[str]]:
        """
        Get the alternatives for a rule.
        :param non_terminal: The non_terminal to get the alternatives for.
        :return: A list of alternatives.
        """
        return self._rules[non_terminal]

    @staticmethod
    def is_terminal(token: str) -> bool:
        """
        Check if a token is a terminal.
        :param token: The token to check.
        :return: True if the token is a terminal, False otherwise.
        """
        return not (token.startswith('<') and token.endswith('>'))

    def get_left_side_terminals(self, non_terminal: str, max_depth: int) -> set[str]:
        """
        Get the left side terminals of a given non-terminal.
        """
        return self._get_left_side_terminals(non_terminal, max_depth)[0]

    def _get_left_side_terminals(self, non_terminal: str, max_depth: int) -> tuple[set[str], int]:
        """
        Get the left side terminals of a given non-terminal.
        """
        alternatives = self.get_alternatives_for_a_rule(non_terminal)
        left_side_terminals = set()
        for alternative in alternatives:
            left = alternative[0]
            if self.is_terminal(left):
                left_side_terminals.add(left)
            else:
                if max_depth == 0:
                    continue
                left_side_terminals.update(self._get_left_side_terminals(left, max_depth - 1)[0])
        return left_side_terminals, max_depth

    def is_correct(self) -> bool:
        """
        Check if the grammar is correct.
        :return: True if the grammar is correct, False otherwise.
        """
        try:
            self.check_syntax_correctness()
        except:
            return False

        return True

    def check_syntax_correctness(self):
        """
        Check if the grammar is correct.
        If the grammar is correct, it will return None, otherwise it will raise a SyntaxError.
        """
        # Try to parse the grammar, may raise
        self.parse_grammar(self._grammar_text)

        # Check if all non-terminals are defined
        undefined_non_terminals = set()
        non_terminals = self.get_non_terminals()
        rules = self.get_rules()
        for lhs, alternatives in rules.items():
            for alternative in alternatives:
                for token in alternative:
                    if not BNFParser.is_terminal(token):
                        if token not in non_terminals:
                            undefined_non_terminals.add(token)
        if len(undefined_non_terminals) != 0:
            undefined_non_terminals = ', '.join(undefined_non_terminals)
            error_msg = (fr'''
Syntax error, "Undefined non-terminals", happened for this BNF.
This error is likely due to the following non-terminals not being defined:
{undefined_non_terminals}
            ''').strip()
            raise SyntaxError(error_msg)
        else:
            return True

    @deprecated("This method should not be used any more since NPDA is not used in the project and deprecated.")
    def get_optimized_bnf_parser_for_npda(self) -> 'BNFParser':
        """
        A new BNF parser instance in which each terminal is replaced with a single char terminal excluding escape character.
        :return: A new BNF parser instance.
        """
        bnf = deepcopy(self)  # do not modify the original parser
        for lhs, alternatives in bnf._rules.items():
            for i, alternative in enumerate(alternatives):
                new_alternative = []
                for symbol in alternative:
                    if not BNFParser.is_terminal(symbol):
                        new_alternative.append(symbol)
                    else:
                        tokens = split_string_with_escapes(escape_string(symbol))
                        tokens = map(lambda token: unescape_string(token), tokens)
                        new_alternative.extend(tokens)
                alternatives[i] = new_alternative
        return bnf

    def get_start_symbol(self) -> str:
        """
        Get the start symbol of the grammar.
        Note: this assumes that the first rule is the start symbol.
        :return: The start symbol.
        """
        return list(self._rules.keys())[0]

    def is_left_recursive(self) -> bool:
        """
        Check if the grammar is left recursive.
        :return: True if the grammar is left recursive, False otherwise.
        """
        for non_terminal in self.get_non_terminals():
            visited = set()
            if self._is_left_recursive_from(non_terminal, non_terminal, visited):
                return True
        return False

    def _is_left_recursive_from(self, current_nt: str, target_nt: str, visited: set[str]) -> bool:
        """
        Recursive helper function to detect left recursion starting from current non-terminal and looking for target non-terminal.
        :param current_nt: The current non-terminal being processed.
        :param target_nt: The original non-terminal we are checking for left recursion.
        :param visited: A set of non-terminals that have already been visited to prevent infinite loops.
        :return: True if left recursion is detected, False otherwise.
        """
        visited.add(current_nt)
        alternatives = self.get_alternatives_for_a_rule(current_nt)
        for alternative in alternatives:
            if not alternative:
                continue  # Skip empty alternatives
            first_symbol = alternative[0]
            if first_symbol == target_nt:
                # Immediate left recursion detected
                return True
            elif not self.is_terminal(first_symbol) and first_symbol not in visited:
                # Recurse on the non-terminal
                if self._is_left_recursive_from(first_symbol, target_nt, visited):
                    return True
        return False

    def to_text(self) -> str:
        """
        Convert the BNF parser to a BNF text in a compact form where all alternatives of a non-terminal are on the same line.
        :return: A BNF text.
        """

        def token_to_str(token: str) -> str:
            if self.is_terminal(token):
                # Terminal: Enclose in double quotes, double any existing double quotes inside
                escaped_token = token.replace('"', '""')
                return f'"{escaped_token}"'
            else:
                # Non-terminal: Return as is
                return token
        lines = []
        for lhs in self._rules:
            alternatives = self._rules[lhs]
            rhs = ' | '.join(' '.join(token_to_str(token) for token in alternative) for alternative in alternatives)
            line = f"{lhs} ::= {rhs}"
            lines.append(line)
        return '\n'.join(lines)

    @staticmethod
    def create_parser_from_rules(rules: Rules) -> 'BNFParser':
        """
        Create a new BNF parser from a set of rules.
        :param rules: A dictionary of rules.
        :return: A new BNF parser.
        """
        bnf = BNFParser('')
        bnf._rules = deepcopy(rules)
        return bnf

    def count_production_rules(self) -> int:
        """
        Count the production rules in a grammar.

        :return: number of prs
        """
        count = 0
        rules = self.get_rules()
        for rule in rules.items():
            _, alternatives = rule
            count += len(alternatives)
        return count

# Example usage:
if __name__ == "__main__":
    bnf_text = r'''
    <term> ::= "12" "32" | "2" | "2"
    '''
    parser = BNFParser(bnf_text)
    parser.pretty_print()
    print(parser.to_text())
