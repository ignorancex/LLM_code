import unittest

from packages.bnf.parser import BNFParser


class TestBNFParser(unittest.TestCase):
    def test_simple_grammar(self):
        bnf_text = '''
        <start> ::= <expr>
        <expr> ::= <term> "+" <term>
                 | <term> "-" <term>
        <term> ::= "0" | "1"
        '''
        parser = BNFParser(bnf_text)
        expected_rules = {
            '<start>': [['<expr>']],
            '<expr>': [['<term>', '+', '<term>'], ['<term>', '-', '<term>']],
            '<term>': [['0'], ['1']]
        }
        self.assertEqual(parser.get_rules(), expected_rules)

    def test_terminal_with_pipe(self):
        bnf_text = '''
        <pipe_test> ::= "|" | "normal"
        '''
        parser = BNFParser(bnf_text)
        expected_rules = {
            '<pipe_test>': [['|'], ['normal']]
        }
        self.assertEqual(parser.get_rules(), expected_rules)

    def test_terminal_with_escape_character(self):
        bnf_text = r'''
        <escape_test> ::= "\t" | "\n"
        '''
        parser = BNFParser(bnf_text)
        expected_rules = {
            '<escape_test>': [["\t"], ['\n']]
        }
        self.assertEqual(parser.get_rules(), expected_rules)

    def test_complex_grammar(self):
        bnf_text = '''
        <expr> ::= <expr> "+" <term>
                 | <term>
        <term> ::= <term> "*" <factor>
                 | <factor>
        <factor> ::= "(" <expr> ")"
                   | <number>
        <number> ::= <digit> | <number> <digit>
        <digit> ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
        '''
        parser = BNFParser(bnf_text)
        expected_rules = {
            '<expr>': [['<expr>', '+', '<term>'], ['<term>']],
            '<term>': [['<term>', '*', '<factor>'], ['<factor>']],
            '<factor>': [['(', '<expr>', ')'], ['<number>']],
            '<number>': [['<digit>'], ['<number>', '<digit>']],
            '<digit>': [['0'], ['1'], ['2'], ['3'], ['4'], ['5'], ['6'], ['7'], ['8'], ['9']]
        }
        self.assertEqual(parser.get_rules(), expected_rules)

    def test_syntax_error_missing_lhs(self):
        bnf_text = '''
        ::= "no non-terminal alternative"
        '''
        with self.assertRaises(SyntaxError):
            BNFParser(bnf_text)

    def test_syntax_error_invalid_token(self):
        bnf_text = '''
        <start> ::= invalid_token
        '''
        with self.assertRaises(SyntaxError):
            BNFParser(bnf_text)

    def test_empty_grammar(self):
        bnf_text = ''
        parser = BNFParser(bnf_text)
        self.assertEqual(parser.get_rules(), {})

    def test_unquoted_terminal(self):
        bnf_text = '''
        <expr> ::= a
        '''
        with self.assertRaises(SyntaxError):
            BNFParser(bnf_text)

    def test_nested_rules(self):
        bnf_text = '''
        <A> ::= <B> | "a"
        <B> ::= <C> | "b"
        <C> ::= "c"
        '''
        parser = BNFParser(bnf_text)
        expected_rules = {
            '<A>': [['<B>'], ['a']],
            '<B>': [['<C>'], ['b']],
            '<C>': [['c']]
        }
        self.assertEqual(parser.get_rules(), expected_rules)

    def test_rule_with_multiple_pipes_in_terminal(self):
        bnf_text = '''
        <test> ::= "|" | "||" | "|||"
        '''
        parser = BNFParser(bnf_text)
        expected_rules = {
            '<test>': [['|'], ['||'], ['|||']]
        }
        self.assertEqual(parser.get_rules(), expected_rules)

    def test_long_terminal(self):
        bnf_text = '''
        <greeting> ::= "hello world"
        '''
        parser = BNFParser(bnf_text)
        expected_rules = {
            '<greeting>': [['hello world']]
        }
        self.assertEqual(parser.get_rules(), expected_rules)

    def test_alternatives(self):
        bnf_text = '''
        <start> ::= <A>
        <A> ::= "a"
               | "b"
               | "c"
        '''
        parser = BNFParser(bnf_text)
        expected_rules = {
            '<start>': [['<A>']],
            '<A>': [['a'], ['b'], ['c']]
        }
        self.assertEqual(parser.get_rules(), expected_rules)

    def test_left_recursion(self):
        bnf_text = '''
        <list> ::= <list> "," <item>
                 | <item>
        <item> ::= "item"
        '''
        parser = BNFParser(bnf_text)
        expected_rules = {
            '<list>': [['<list>', ',', '<item>'], ['<item>']],
            '<item>': [['item']]
        }
        self.assertEqual(parser.get_rules(), expected_rules)

    def test_multiple_rules_on_same_line(self):
        bnf_text = '''
        <A> ::= "a" | "aa" | "aaa"
        '''
        parser = BNFParser(bnf_text)
        expected_rules = {
            '<A>': [['a'], ['aa'], ['aaa']]
        }
        self.assertEqual(parser.get_rules(), expected_rules)

    def test_rule_with_parentheses_in_terminal(self):
        bnf_text = '''
        <paren_test> ::= "(" ")" | "(" <paren_test> ")"
        '''
        parser = BNFParser(bnf_text)
        expected_rules = {
            '<paren_test>': [['(', ')'], ['(', '<paren_test>', ')']]
        }
        self.assertEqual(parser.get_rules(), expected_rules)

    def test_rule_with_special_characters_in_terminal(self):
        bnf_text = '''
        <special> ::= "!" | "@" | "#" | "$" | "%"
        '''
        parser = BNFParser(bnf_text)
        expected_rules = {
            '<special>': [['!'], ['@'], ['#'], ['$'], ['%']]
        }
        self.assertEqual(parser.get_rules(), expected_rules)

    def test_get_non_terminals(self):
        bnf_text = '''
        <A> ::= "a" | "b"
        <B> ::= <A> | "c"
        <C> ::= "a" <A> "b" | <B>
        '''
        parser = BNFParser(bnf_text)
        self.assertEqual(parser.get_non_terminals(), {'<A>', '<B>', '<C>'})

    def test_get_non_terminals_for_a_rule(self):
        bnf_text = '''
        <A> ::= "a" | "b"
        <B> ::= <A> | "c"
        <C> ::= "a" <A> "b" | <B>
        '''
        parser = BNFParser(bnf_text)
        self.assertEqual(parser.get_non_terminals_for_a_rule('<A>'), set())
        self.assertEqual(parser.get_non_terminals_for_a_rule('<B>'), {"<A>"})
        self.assertEqual(parser.get_non_terminals_for_a_rule('<C>'), {'<A>', '<B>'})

    def test_get_terminals(self):
        bnf_text = '''
        <A> ::= "a" | "b" 
        <B> ::= <A> | "c"
        <C> ::= "a" <A> "b" | <B>
        '''
        parser = BNFParser(bnf_text)
        self.assertEqual(parser.get_terminals(), {'a', 'b', 'c'})

    def test_get_terminals_for_a_rule(self):
        bnf_text = '''
        <A> ::= "a" | "b"
        <B> ::= <A> | "c"
        <C> ::= "a" <A> "b" | <B>
        '''
        parser = BNFParser(bnf_text)
        self.assertEqual(parser.get_terminals_for_a_rule('<A>'), {'a', 'b'})
        self.assertEqual(parser.get_terminals_for_a_rule('<B>'), {'c'})
        self.assertEqual(parser.get_terminals_for_a_rule('<C>'), {"a", "b"})

    def test_get_left_side_terminals(self):
        bnf_text = '''
        <A> ::= "a" | "b"
        <B> ::= <A> | "c"
        <C> ::= "a" <A> "b" | <B>
        '''
        parser = BNFParser(bnf_text)
        self.assertEqual(parser.get_left_side_terminals("<A>", -1), {"a", "b"})
        self.assertEqual(parser.get_left_side_terminals("<B>", -1), {"a", "b", "c"})
        self.assertEqual(parser.get_left_side_terminals("<C>", -1), {"a", "b", "c"})
        self.assertEqual(parser.get_left_side_terminals("<A>", 0), {"a", "b"})
        self.assertEqual(parser.get_left_side_terminals("<B>", 0), {"c"})
        self.assertEqual(parser.get_left_side_terminals("<C>", 0), {"a"})

        bnf_text = '''
        <A> ::= "a" | "b"
        <B> ::= "c"
        <C> ::= <A> <B>
        '''
        parser = BNFParser(bnf_text)
        self.assertEqual(parser.get_left_side_terminals("<C>", -1), {"a", "b"})

        bnf_text = '''
        <A> ::= "a"
        <B> ::= <A>
        <C> ::= <B>
        '''
        parser = BNFParser(bnf_text)
        self.assertEqual(parser.get_left_side_terminals("<C>", -1), {"a"})
        self.assertEqual(parser.get_left_side_terminals("<C>", 0), set())
        self.assertEqual(parser.get_left_side_terminals("<C>", 1), set())
        self.assertEqual(parser.get_left_side_terminals("<C>", 2), {'a'})

    def test_is_correct(self):
        bnf_text = '''
        <A> ::= "a" | "b"
        <B> ::= <A> | "c"
        <C> ::= "a" <A> "b" | <B>
        '''
        parser = BNFParser(bnf_text)
        self.assertTrue(parser.is_correct())

        bnf_text = '''
        <A> ::= "a" | "b"
        <B> ::= <A> | "c"
        <C> ::= "a" <A> "b" | <B>
        <D> ::= <E>
        '''
        parser = BNFParser(bnf_text)
        self.assertFalse(parser.is_correct())

    def test_invalid_line(self):
        bnf_text = r"""
        <greeting> ::= "Hello" <first_name> " " <last_name> "!\n"
        <first_name> ::= "John" | "Jane"
        <invalid_line> = "invalid"
        """
        with self.assertRaises(SyntaxError):
            BNFParser(bnf_text)

    def test_invalid_line_starts_with_pipeline(self):
        bnf_text = r"""
        | "Yes"
        """
        with self.assertRaises(SyntaxError):
            BNFParser(bnf_text)

    def test_invalid_line_right_side_empty(self):
        bnf_text = r"""
        <greeting> ::=
        """
        with self.assertRaises(SyntaxError):
            BNFParser(bnf_text)

    def test_invalid_line_escape_error(self):
        bnf_text = r'''
        <term> ::= "\""
        '''
        with self.assertRaises(SyntaxError):
            BNFParser(bnf_text)

    def test_invalid_line_terminals_not_in_quotes(self):
        bnf_text = r'''
        <term> ::= a
        '''
        with self.assertRaises(SyntaxError):
            BNFParser(bnf_text)

    def test_incorrect_bnf_not_exhaustive_non_terminals(self):
        bnf_text = r"""
        <greeting> ::= <greeting_word> " " <first_name> " " <last_name> "!\n"
        <first_name> ::= "John" | "Jane"
        """
        parser = BNFParser(bnf_text)
        self.assertFalse(parser.is_correct())

    def test_is_left_recursive(self):
        bnf_text = r"""
        <list> ::= <list> "," <item>
                 | <item>
        <item> ::= "item"
        """
        parser = BNFParser(bnf_text)
        self.assertTrue(parser.is_left_recursive())

        bnf_text = r"""
        <list> ::= <item> "," <list>
                 | <item>
        <item> ::= "item"
        """
        parser = BNFParser(bnf_text)
        self.assertFalse(parser.is_left_recursive())

    def test_to_text(self):
        bnf_text = r"""
<list> ::= <list> "," <item> | <item>
<item> ::= "item"
        """.strip()
        parser = BNFParser(bnf_text)
        self.assertEqual(parser.to_text(), bnf_text)

        bnf_text = r"""
<expr> ::= <expr> "+" <term> | <term>
<term> ::= <term> "*" <factor> | <factor>
<factor> ::= "(" <expr> ")" | <number>
<number> ::= <digit> | <number> <digit>
<digit> ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
        """.strip()
        parser = BNFParser(bnf_text)
        self.assertEqual(parser.to_text(), bnf_text)

        bnf_text = r"""
<start> ::= <expr>
<expr> ::= <term> "+" <term> | <term> "-" <term>
<term> ::= "0" | "1"
        """.strip()
        parser = BNFParser(bnf_text)
        self.assertEqual(parser.to_text(), bnf_text)

    def test_create_parser_from_rules(self):
        rules = {
            '<start>': [['<expr>']],
            '<expr>': [['<term>', '+', '<term>'], ['<term>', '-', '<term>']],
            '<term>': [['0'], ['1']]
        }
        parser = BNFParser.create_parser_from_rules(rules)
        self.assertEqual(parser.get_rules(), rules)

    def test_empty_terminal(self):
        bnf_text = r"""
        <empty> ::= ""
        """
        parser = BNFParser(bnf_text)
        expected_rules = {
            '<empty>': [[""]]
        }
        self.assertEqual(parser.get_rules(), expected_rules)

    def test_special_terminals(self):
        bnf_text = r"""
        <special> ::= "ε" | "α" | "β" | "γ" | "δ"
        """
        parser = BNFParser(bnf_text)
        expected_rules = {
            '<special>': [["ε"], ["α"], ["β"], ["γ"], ["δ"]]
        }
        self.assertEqual(parser.get_rules(), expected_rules)

    def test_clear_redundant_alternatives(self):
        # remove redundant alternatives by default
        bnf_text = r"""
        <A> ::= "a" | "b" | "a"
        <B> ::= <A> | "c" | <A>
        <C> ::= "a" <A> "b" | <B> | "a" <A> "b"
        """
        parser = BNFParser(bnf_text)
        expected_rules = {
            '<A>': [['a'], ['b']],
            '<B>': [['<A>'], ['c']],
            '<C>': [['a', '<A>', 'b'], ['<B>']]
        }
        self.assertEqual(parser.get_rules(), expected_rules)

    def test_count_production_rules(self):
        # simple
        bnf_text = r"""
        <sentence> ::= <noun_phrase> <verb_phrase>
        <noun_phrase> ::= <article> <noun>
        <verb_phrase> ::= <verb> <noun_phrase> | <verb>
        <article> ::= "the" | "a" 
        """
        parser = BNFParser(bnf_text)
        self.assertEqual(parser.count_production_rules(),6)

        # redundant
        bnf_text = r"""
        <A> ::= "a" | "b" | "a"
        <B> ::= <A> | "c" | <A>
        <C> ::= "a" <A> "b" | <B> | "a" <A> "b" 
        """
        parser = BNFParser(bnf_text)
        self.assertEqual(parser.count_production_rules(),6)