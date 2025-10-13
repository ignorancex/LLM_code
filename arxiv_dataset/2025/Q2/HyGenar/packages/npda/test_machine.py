import unittest
from unittest import TestCase

from packages.bnf.parser import BNFParser
from packages.npda.machine import convert_bnf_to_npda

@unittest.skip('Skip NPDA tests since NPDA is deprecated.')
class TestConvertBNFToNPDA(TestCase):
    def test_convert_bnf_to_npda_complexity_0(self):
        grammar = r"""
        <expr> ::= "1" | "2"
        """
        bnf = BNFParser(grammar).get_optimized_bnf_parser_for_npda()
        npda = convert_bnf_to_npda(bnf, "<expr>")
        npda.validate()
        self.assertEqual(npda.accepts_input("1"), True)
        self.assertEqual(npda.accepts_input("2"), True)
        self.assertEqual(npda.accepts_input("3"), False)

    def test_convert_bnf_to_npda_complexity_1(self):
        grammar = r"""
        <expr> ::= <term> "+" <expr> | <term>
        <term> ::= "1"
        """
        bnf = BNFParser(grammar).get_optimized_bnf_parser_for_npda()
        npda = convert_bnf_to_npda(bnf, "<expr>")
        npda.validate()
        self.assertEqual(npda.accepts_input("1"), True)
        self.assertEqual(npda.accepts_input("1+1"), True)
        self.assertEqual(npda.accepts_input("1+1+1"), True)
        self.assertEqual(npda.accepts_input("2"), False)
        self.assertEqual(npda.accepts_input("1+"), False)
        self.assertEqual(npda.accepts_input("+1"), False)
        self.assertEqual(npda.accepts_input("1+1+"), False)

    def test_convert_bnf_to_npda_complexity_2(self):
        grammar = r"""
        <expr> ::= <term> "+" <expr> | <term>
        <term> ::= <factor> "*" <term> | <factor>
        <factor> ::= "1" | "2" | "3"
        """
        bnf = BNFParser(grammar).get_optimized_bnf_parser_for_npda()
        npda = convert_bnf_to_npda(bnf, "<expr>")
        npda.validate()
        self.assertEqual(npda.accepts_input("1"), True)
        self.assertEqual(npda.accepts_input("1+1"), True)
        self.assertEqual(npda.accepts_input("1+1+1"), True)
        self.assertEqual(npda.accepts_input("1*1"), True)
        self.assertEqual(npda.accepts_input("1*1*1"), True)
        self.assertEqual(npda.accepts_input("1+1*1"), True)
        self.assertEqual(npda.accepts_input("1*1+1"), True)

    def test_convert_bnf_to_npda_complexity_3(self):
        # Left recursion grammar
        grammar = r"""
        <expr> ::= <expr> "+" <term> | <term>
        <term> ::= <factor> "*" <term> | <factor>
        <factor> ::= "1" | "2" | "3"
        """
        bnf = BNFParser(grammar).get_optimized_bnf_parser_for_npda()
        npda = convert_bnf_to_npda(bnf, "<expr>")
        npda.validate()
        self.assertEqual(npda.accepts_input("1"), True)
        self.assertEqual(npda.accepts_input("1+1"), True)
        self.assertEqual(npda.accepts_input("1+1+1"), True)
        self.assertEqual(npda.accepts_input("1*1"), True)
        self.assertEqual(npda.accepts_input("1*1*1"), True)
        self.assertEqual(npda.accepts_input("1+1*1"), True)
        self.assertEqual(npda.accepts_input("1*1+1"), True)

        # Note: if the string is not right e.g. "1a", it will run forever, since currently do not handle this LR issue.


    def test_convert_bnf_to_npda_complexity_4(self):
        # Escape character grammar
        grammar = r"""
        <config> ::= <entry> | <entry> "\n" <config>
        <entry> ::= <key> "=" <value>
        <key> ::= <letter>
        <value> ::= <letter> | <digit> 
        <letter> ::= "a" | "b" | "c" | "n"
        <digit> ::= "0" | "1"
        """
        bnf = BNFParser(grammar).get_optimized_bnf_parser_for_npda()
        npda = convert_bnf_to_npda(bnf, "<config>")
        npda.validate()
        # npda.show_diagram().draw("npda.png")
        self.assertEqual(npda.accepts_input("a=1"), True)
        self.assertEqual(npda.accepts_input("a=1\na=0"), True)
        self.assertEqual(npda.accepts_input("a=1\nb=0\nc=1"), True)
        self.assertEqual(npda.accepts_input("n=1\nn=0\nn=1"), True)
        self.assertEqual(npda.accepts_input("a=1\nb=0\nc=1\n"), False)
        self.assertEqual(npda.accepts_input("\na=1\nb=0\nc=1"), False)

    def test_convert_bnf_to_npda_complexity_5(self):
        grammar = r"""
        <command> ::= <action> " " <object>
        <action> ::= "take" | "drop"
        <object> ::= <object_list>
        <object_list> ::= <item> " and " <object_list> | <item>
        <item> ::= "sword" | "shield"
        <and> ::= "and"
        """
        bnf = BNFParser(grammar).get_optimized_bnf_parser_for_npda()
        npda = convert_bnf_to_npda(bnf, "<command>")
        npda.validate()
        self.assertEqual(npda.accepts_input("take sword"), True)
        self.assertEqual(npda.accepts_input("take sword and shield"), True)
        self.assertEqual(npda.accepts_input("take sword and shield and sword"), True)

    def test_convert_bnf_to_npda_complexity_6(self):
        grammar = r"""
        <command> ::= <action> " " <object> | <action> " " <object> " " <modifier>
        <action> ::= "take" | "drop" | "use"
        <object> ::= <item> | <item> " with " <item>
        <item> ::= "key" | "sword" | "potion"
        <modifier> ::= <adverb> | <direction> | <location>
        <adverb> ::= "quickly" | "silently" | "carefully"
        <direction> ::= "north" | "south" | "east" | "west"
        <location> ::= "castle" | "forest" | "village"
        """
        bnf = BNFParser(grammar).get_optimized_bnf_parser_for_npda()
        npda = convert_bnf_to_npda(bnf, "<command>")
        npda.validate()
        self.assertEqual(npda.accepts_input("take key"), True)
        self.assertEqual(npda.accepts_input("take key with sword"), True)
        self.assertEqual(npda.accepts_input("take key with sword quickly"), True)

    def test_convert_bnf_to_npda_complexity_7(self):
        grammar = r"""
        <document> ::= <element> <document_tail>
        <document_tail> ::= <element> <document_tail> | ""
        <element> ::= <start_tag> <content> <end_tag>
        <start_tag> ::= "<" <tag_name> ">"
        <end_tag> ::= "</" <tag_name> ">"
        <tag_name> ::= "title" | "body" | "section"
        <content> ::= <text> | <element> <content_tail>
        <content_tail> ::= <element> <content_tail> | ""
        <text> ::= "a" | "b" | "c"
        """
        bnf = BNFParser(grammar).get_optimized_bnf_parser_for_npda()
        npda = convert_bnf_to_npda(bnf, "<document>")
        npda.validate()
        self.assertEqual(npda.accepts_input("<title>a</title>"), True)
        self.assertEqual(npda.accepts_input("<title>a</title><body>b</body>"), True)
        self.assertEqual(npda.accepts_input("<body><section>a</section></body>"), True)

    def test_convert_bnf_to_npda_complexity_8(self):
        grammar = r"""
        <query> ::= <select> <from> <where>
        <select> ::= "SELECT" " " <columns>
        <from> ::= " FROM" " " <table>
        <where> ::= " WHERE " <condition> | ""
        <columns> ::= "*" | <column_list>
        <column_list> ::= <column> | <column> "," <column_list>
        <column> ::= "name" | "age" | "salary"
        <table> ::= "employees" | "departments"
        <condition> ::= <column> "=" <value>
        <value> ::= "John" | "30" | "50000"
        """
        bnf = BNFParser(grammar).get_optimized_bnf_parser_for_npda()
        npda = convert_bnf_to_npda(bnf, "<query>")
        npda.validate()
        self.assertEqual(npda.accepts_input("SELECT * FROM employees"), True)
        self.assertEqual(npda.accepts_input("SELECT name,age FROM employees"), True)
        self.assertEqual(npda.accepts_input("SELECT name,age FROM employees WHERE name=John"), True)

    def test_convert_bnf_to_npda_complexity_9(self):
        # test some non-terminal symbols are not used in the grammar
        grammar = r"""
        <expression> ::= <expression> " OR " <term> | <term>
        <term> ::= <term> " AND " <factor> | <factor>
        <factor> ::= "NOT " <factor> | "(" <expression> ")" | <boolean>
        <boolean> ::= "TRUE" | "FALSE" | <variable>
        <variable> ::= "A" | "B" | "C"
        <operator_or> ::= "OR"
        <operator_and> ::= "AND"
        <operator_not> ::= "NOT"
        <parenthesis_open> ::= "("
        <parenthesis_close> ::= ")"
        """
        bnf = BNFParser(grammar).get_optimized_bnf_parser_for_npda()
        npda = convert_bnf_to_npda(bnf, "<expression>")
        npda.validate()
        self.assertEqual(npda.accepts_input("A"), True)
        self.assertEqual(npda.accepts_input("(A)"), True)
        self.assertEqual(npda.accepts_input("A OR B"), True)
        self.assertEqual(npda.accepts_input("(A OR B)"), True)
        self.assertEqual(npda.accepts_input("A OR B AND C"), True)
        self.assertEqual(npda.accepts_input("A OR (B AND C)"), True)
        self.assertEqual(npda.accepts_input("A OR B AND C OR A"), True)