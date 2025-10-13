from unittest import TestCase

from packages.bnf.glr import GLRParser
from packages.bnf.parser import BNFParser


class TestGLR(TestCase):
    """
    Write intensive tests for the GLR parser to ensure it can handle a variety of grammars and inputs.
    """

    def test_empty_terminal(self):
        grammar = r'''
        <term> ::= ""
        '''
        bnf_parser = BNFParser(grammar)
        glr = GLRParser(bnf_parser)
        self.assertTrue(glr.accepts_input(''))
        self.assertFalse(glr.accepts_input(' '))
        self.assertFalse(glr.accepts_input('  '))

        grammar = r'''
        <term> ::= "1" | ""
        '''
        bnf_parser = BNFParser(grammar)
        glr = GLRParser(bnf_parser)
        self.assertTrue(glr.accepts_input(''))
        self.assertTrue(glr.accepts_input('1'))
        self.assertFalse(glr.accepts_input(' '))

    def test_complexity_0(self):
        grammar = r'''
        <term> ::= "1" | "2" | " "
        '''
        bnf_parser = BNFParser(grammar)
        glr = GLRParser(bnf_parser)
        self.assertTrue(glr.accepts_input('1'))
        self.assertTrue(glr.accepts_input('2'))
        self.assertTrue(glr.accepts_input(' '))
        self.assertFalse(glr.accepts_input('3'))
        self.assertFalse(glr.accepts_input('a'))
        self.assertFalse(glr.accepts_input('&'))

    def test_complexity_1(self):
        grammar = r'''
        <term> ::= <factor> | <term> "*" <factor>
        <factor> ::= "id" | "(" <term> ")"
        '''
        bnf_parser = BNFParser(grammar)
        glr = GLRParser(bnf_parser)
        self.assertTrue(glr.accepts_input('id'))
        self.assertTrue(glr.accepts_input('id*id'))
        self.assertTrue(glr.accepts_input('(id)'))
        self.assertTrue(glr.accepts_input('(id)*id'))
        self.assertTrue(glr.accepts_input('id*(id)'))
        self.assertTrue(glr.accepts_input('(id)*(id)'))
        self.assertFalse(glr.accepts_input('id*'))
        self.assertFalse(glr.accepts_input('id*id*'))

    def test_complexity_2(self):
        grammar = r"""
        <expr> ::= <term> "+" <expr> | <term>
        <term> ::= <factor> "*" <term> | <factor>
        <factor> ::= "1" | "2" | "3"
        """
        bnf = BNFParser(grammar)
        glr = GLRParser(bnf)
        self.assertEqual(glr.accepts_input("1"), True)
        self.assertEqual(glr.accepts_input("1+1"), True)
        self.assertEqual(glr.accepts_input("1+1+1"), True)
        self.assertEqual(glr.accepts_input("1*1"), True)
        self.assertEqual(glr.accepts_input("1*1*1"), True)
        self.assertEqual(glr.accepts_input("1+1*1"), True)
        self.assertEqual(glr.accepts_input("1*1+1"), True)

    def test_complexity_3(self):
        # Left recursion grammar
        grammar = r"""
        <expr> ::= <expr> "+" <term> | <term>
        <term> ::= <factor> "*" <term> | <factor>
        <factor> ::= "1" | "2" | "3" | ""
        """
        bnf = BNFParser(grammar)
        glr = GLRParser(bnf)
        self.assertEqual(glr.accepts_input("1"), True)
        self.assertEqual(glr.accepts_input("1+1"), True)
        self.assertEqual(glr.accepts_input("1+1+1"), True)
        self.assertEqual(glr.accepts_input("1*1"), True)
        self.assertEqual(glr.accepts_input("1*1*1"), True)
        self.assertEqual(glr.accepts_input("1+1*1"), True)
        self.assertEqual(glr.accepts_input("1*1+1"), True)

    def test_complexity_4(self):
        # Escape character grammar
        grammar = r"""
        <config> ::= <entry> | <entry> "\n" <config>
        <entry> ::= <key> "=" <value>
        <key> ::= <letter>
        <value> ::= <letter> | <digit> 
        <letter> ::= "a" | "b" | "c" | "n"
        <digit> ::= "0" | "1"
        """
        bnf = BNFParser(grammar)
        glr = GLRParser(bnf)
        self.assertEqual(glr.accepts_input("a=1"), True)
        self.assertEqual(glr.accepts_input("a=1\na=0"), True)
        self.assertEqual(glr.accepts_input("a=1\nb=0\nc=1"), True)
        self.assertEqual(glr.accepts_input("n=1\nn=0\nn=1"), True)
        self.assertEqual(glr.accepts_input("a=1\nb=0\nc=1\n"), False)
        self.assertEqual(glr.accepts_input("\na=1\nb=0\nc=1"), False)

    def test_complexity_5(self):
        grammar = r"""
        <command> ::= <action> " " <object>
        <action> ::= "take" | "drop"
        <object> ::= <object_list>
        <object_list> ::= <item> " and " <object_list> | <item>
        <item> ::= "sword" | "shield"
        <and> ::= "and"
        """
        bnf = BNFParser(grammar)
        glr = GLRParser(bnf)
        self.assertEqual(glr.accepts_input("take sword"), True)
        self.assertEqual(glr.accepts_input("take sword and shield"), True)
        self.assertEqual(glr.accepts_input("take sword and shield and sword"), True)

    def test_complexity_6(self):
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
        bnf = BNFParser(grammar)
        glr = GLRParser(bnf)
        self.assertEqual(glr.accepts_input("take key"), True)
        self.assertEqual(glr.accepts_input("take key with sword"), True)
        self.assertEqual(glr.accepts_input("take key with sword quickly"), True)

    def test_complexity_7(self):
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
        bnf = BNFParser(grammar)
        glr = GLRParser(bnf)
        self.assertEqual(glr.accepts_input("<title>a</title>"), True)
        self.assertEqual(glr.accepts_input("<title>a</title><body>b</body>"), True)
        self.assertEqual(glr.accepts_input("<body><section>a</section></body>"), True)

    def test_complexity_8(self):
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
        bnf = BNFParser(grammar)
        glr = GLRParser(bnf)
        self.assertEqual(glr.accepts_input("SELECT * FROM employees"), True)
        self.assertEqual(glr.accepts_input("SELECT name,age FROM employees"), True)
        self.assertEqual(glr.accepts_input("SELECT name,age FROM employees WHERE name=John"), True)

    def test_complexity_9(self):
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
        bnf = BNFParser(grammar)
        glr = GLRParser(bnf)
        self.assertEqual(glr.accepts_input("A"), True)
        self.assertEqual(glr.accepts_input("(A)"), True)
        self.assertEqual(glr.accepts_input("(A OR B)"), True)
        self.assertEqual(glr.accepts_input("A OR B AND C"), True)
        self.assertEqual(glr.accepts_input("A OR (B AND C)"), True)
        self.assertEqual(glr.accepts_input("A OR B AND C OR A"), True)

    def test_complexity_10(self):
        # Ambiguous grammar which can not be handled by LR(1) parser
        grammar = r"""
        <command> ::= <action_phrase> " " <object>
        <action_phrase> ::= <action> | <action> " " <adverb>
        <action> ::= "move" | "take" | "drop"
        <adverb> ::= "quickly" | "slowly"
        <object> ::= <item> | <direction>
        <item> ::= "key" | "coin" | "sword"
        <direction> ::= "north" | "south" | "east" | "west"
        """
        bnf = BNFParser(grammar)
        glr = GLRParser(bnf)
        self.assertTrue(glr.accepts_input('move key'))

    def test_complexity_11(self):
        grammar = r"""
        <Spaceship> ::= <ShipName> "is a" <ShipType> "equipped with" <Features>
        <ShipName> ::= "USS Enterprise" | "Millennium Falcon" | "Serenity" | "Galactica"
        <ShipType> ::= "explorer" | "fighter" | "cargo vessel" | "research ship"
        <Features> ::= <Feature> | <Feature> ", " <Features>
        <Feature> ::= <EngineType> | <WeaponSystem> | <ShieldType> | <CrewCapacity>
        <EngineType> ::= "warp drive" | "hyperdrive" | "ion engines"
        <WeaponSystem> ::= "laser cannons" | "photon torpedoes" | "plasma rifles"
        <ShieldType> ::= "energy shields" | "deflector shields"
        <CrewCapacity> ::= "a crew of 50" | "a crew of 5" | "unmanned"
        """
        bnf = BNFParser(grammar)
        glr = GLRParser(bnf)
        self.assertTrue(glr.accepts_input('Galacticais afighterequipped witha crew of 5'))

    def test_complexity_12(self):
        grammar = r"""
        <email> ::= <local_part> "@" <domain>
        <local_part> ::= <word> <local_part_tail>
        <local_part_tail> ::= "." <word> <local_part_tail> | ""
        <domain> ::= <subdomain> "." <top_level_domain>
        <subdomain> ::= <word>
        <top_level_domain> ::= "com" | "org" | "net"
        <word> ::= <letter> <word_tail>
        <word_tail> ::= <letter> <word_tail> | ""
        <letter> ::= "a" | "b" | "c" | "d" | "e" | "f" | "g" | "h" | "i" | "j" | "k" | "l" | "m" | "n" | "o" | "p" | "q" | "r" | "s" | "t" | "u" | "v" | "w" | "x" | "y" | "z" | "_"
        """
        bnf = BNFParser(grammar)
        glr = GLRParser(bnf)
        self.assertTrue(glr.accepts_input('john_doe.doe@example.com'))
        self.assertTrue(glr.accepts_input('ruta_handsome.hahaha@example.com'))
        self.assertTrue(glr.accepts_input('charlie.coder@development.net'))

    def test_grammar_with_epsilon_production(self):
        """
        Test parsing with a grammar that includes epsilon (empty) productions.
        """
        grammar = r"""
        <S> ::= <A> "a" | "b"
        <A> ::= "" | "c"
        """
        bnf = BNFParser(grammar)
        glr = GLRParser(bnf)
        self.assertTrue(glr.accepts_input("a"))
        self.assertTrue(glr.accepts_input("ca"))
        self.assertTrue(glr.accepts_input("b"))
        self.assertFalse(glr.accepts_input("c"))
        self.assertFalse(glr.accepts_input("caa"))

    def test_ambiguous_grammar(self):
        """
        Test parsing with an ambiguous grammar to ensure the parser handles multiple parses.
        """
        grammar = r"""
        <E> ::= <E> "+" <E> | <E> "*" <E> | "id"
        """
        bnf = BNFParser(grammar)
        glr = GLRParser(bnf)
        self.assertTrue(glr.accepts_input("id+id*id"))
        self.assertTrue(glr.accepts_input("id*id+id"))
        self.assertTrue(glr.accepts_input("id+id+id"))
        self.assertTrue(glr.accepts_input("id"))
        self.assertFalse(glr.accepts_input("id+"))
        self.assertFalse(glr.accepts_input("+id"))

    def test_left_recursive_grammar(self):
        """
        Test parsing with a left-recursive grammar to ensure the parser handles it correctly.
        """
        grammar = r"""
        <Expr> ::= <Expr> "+" <Term> | <Term>
        <Term> ::= <Term> "*" <Factor> | <Factor>
        <Factor> ::= "(" <Expr> ")" | "id"
        """
        bnf = BNFParser(grammar)
        glr = GLRParser(bnf)
        self.assertTrue(glr.accepts_input("id+id"))
        self.assertTrue(glr.accepts_input("id+id*id"))
        self.assertTrue(glr.accepts_input("(id+id)*id"))
        self.assertTrue(glr.accepts_input("id"))
        self.assertTrue(glr.accepts_input("(id)"))
        self.assertFalse(glr.accepts_input("+id"))
        self.assertFalse(glr.accepts_input("id*"))
        self.assertFalse(glr.accepts_input("(id+id"))
        self.assertFalse(glr.accepts_input("id id"))

    def test_unreachable_symbols(self):
        """
        Test parsing with a grammar that includes unreachable symbols to ensure they are ignored.
        """
        grammar = r"""
        <S> ::= <A>
        <A> ::= "a"
        <B> ::= "b"
        """
        bnf = BNFParser(grammar)
        glr = GLRParser(bnf)
        self.assertTrue(glr.accepts_input("a"))
        self.assertFalse(glr.accepts_input("b"))
        self.assertFalse(glr.accepts_input("ab"))

    def test_reduce_reduce_conflict(self):
        """
        Test parsing with a grammar that causes reduce-reduce conflicts to ensure the parser handles them.
        """
        grammar = r"""
        <S> ::= <A> | <B>
        <A> ::= "a"
        <B> ::= "a"
        """
        bnf = BNFParser(grammar)
        glr = GLRParser(bnf)
        self.assertTrue(glr.accepts_input("a"))
        self.assertFalse(glr.accepts_input(""))

    def test_long_input_string(self):
        """
        Test parsing with a very long input string to ensure the parser can handle it.
        """
        grammar = r"""
        <S> ::= <S> "a" | "a"
        """
        bnf = BNFParser(grammar)
        glr = GLRParser(bnf)
        long_string = "a" * 1000
        self.assertTrue(glr.accepts_input(long_string))
        self.assertTrue(glr.accepts_input(long_string + "a"))
        self.assertFalse(glr.accepts_input(long_string + "b"))

    def test_invalid_input(self):
        """
        Test parsing with inputs that are not in the language to ensure they are correctly rejected.
        """
        grammar = r"""
        <S> ::= "a" <S> "b" | ""
        """
        bnf = BNFParser(grammar)
        glr = GLRParser(bnf)
        self.assertTrue(glr.accepts_input(""))
        self.assertTrue(glr.accepts_input("ab"))
        self.assertTrue(glr.accepts_input("aabb"))
        self.assertTrue(glr.accepts_input("aaabbb"))
        self.assertFalse(glr.accepts_input("aab"))
        self.assertFalse(glr.accepts_input("abb"))
        self.assertFalse(glr.accepts_input("aabbb"))
        self.assertFalse(glr.accepts_input("ba"))
        self.assertFalse(glr.accepts_input("aaba"))
        self.assertFalse(glr.accepts_input("abc"))

    def test_unbounded_recursion(self):
        """
        Test parsing with a grammar that can cause unbounded recursion to ensure the parser handles it.
        """
        grammar = r"""
        <S> ::= <S> <S> | "a"
        """
        bnf = BNFParser(grammar)
        glr = GLRParser(bnf)
        self.assertTrue(glr.accepts_input("a"))
        self.assertTrue(glr.accepts_input("aa"))
        self.assertTrue(glr.accepts_input("aaa"))
        self.assertTrue(glr.accepts_input("aaaa"))
        self.assertFalse(glr.accepts_input(""))
        self.assertFalse(glr.accepts_input("b"))
        self.assertFalse(glr.accepts_input("aab"))
        self.assertFalse(glr.accepts_input("aba"))

    def test_multiple_epsilon_productions(self):
        """
        Test parsing with multiple epsilon productions to ensure correct handling.
        """
        grammar = r"""
        <S> ::= <A> <B>
        <A> ::= "a" | ""
        <B> ::= "b" | ""
        """
        bnf = BNFParser(grammar)
        glr = GLRParser(bnf)
        self.assertTrue(glr.accepts_input("ab"))
        self.assertTrue(glr.accepts_input("a"))
        self.assertTrue(glr.accepts_input("b"))
        self.assertTrue(glr.accepts_input(""))
        self.assertFalse(glr.accepts_input("aa"))
        self.assertFalse(glr.accepts_input("bb"))
        self.assertFalse(glr.accepts_input("ba"))
        self.assertFalse(glr.accepts_input("abc"))

    def test_right_recursive_grammar(self):
        """
        Test parsing with a right-recursive grammar to ensure the parser handles it correctly.
        """
        grammar = r"""
        <S> ::= "a" <S> | "a"
        """
        bnf = BNFParser(grammar)
        glr = GLRParser(bnf)
        self.assertTrue(glr.accepts_input("a"))
        self.assertTrue(glr.accepts_input("aa"))
        self.assertTrue(glr.accepts_input("aaa"))
        self.assertTrue(glr.accepts_input("aaaa"))
        self.assertFalse(glr.accepts_input(""))
        self.assertFalse(glr.accepts_input("b"))
        self.assertFalse(glr.accepts_input("aab"))
        self.assertFalse(glr.accepts_input("aba"))

    def test_grammar_with_cycles(self):
        """
        Test parsing with a cyclic grammar to ensure the parser handles it correctly.
        """
        grammar = r"""
        <S> ::= <A>
        <A> ::= <B>
        <B> ::= <C>
        <C> ::= <A> | "c"
        """
        bnf = BNFParser(grammar)
        glr = GLRParser(bnf)
        self.assertTrue(glr.accepts_input("c"))
        self.assertFalse(glr.accepts_input(""))
        self.assertFalse(glr.accepts_input("a"))
        self.assertFalse(glr.accepts_input("cc"))
        self.assertFalse(glr.accepts_input("cabc"))

    def test_mutually_recursive_grammar(self):
        """
        Test parsing with a mutually recursive grammar to ensure the parser handles it correctly.
        """
        grammar = r"""
        <S> ::= <A>
        <A> ::= <B> | "a"
        <B> ::= <A> | "b"
        """
        bnf = BNFParser(grammar)
        glr = GLRParser(bnf)
        self.assertTrue(glr.accepts_input("a"))
        self.assertTrue(glr.accepts_input("b"))
        self.assertFalse(glr.accepts_input("ab"))
        self.assertFalse(glr.accepts_input("abc"))
        self.assertFalse(glr.accepts_input("bac"))

    def test_used_production_rules(self):
        """
        Test parsing with a grammar and for a given input string, check the used production rules.
        """
        grammar = r"""
        <Expr> ::= <Expr> "+" <Term> | <Term>
        <Term> ::= <Term> "*" <Factor> | <Factor>
        <Factor> ::= "(" <Expr> ")" | "id"
        """
        bnf = BNFParser(grammar)
        glr = GLRParser(bnf)
        r,prs = glr.accepts_input("id+id",True)
        self.assertTrue(r)
        self.assertEqual(len(prs),4)