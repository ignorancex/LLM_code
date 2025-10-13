from unittest import TestCase

from packages.bnf.glr import GLRParser
from packages.bnf.parser import BNFParser
from packages.bnf_generation.genetic import bnf_generation, fitness, crossover, mutate, mutate_alternatives
from packages.llms.openai_model import OpenAIModel
from packages.utils.string_tool import extract_code_block
from packages.utils.log import logger


class Test(TestCase):
    def test_bnf_generation(self):
        # disable log
        logger.disabled = True

        llm = OpenAIModel()
        llm.reconfig({
            "model": "gpt-4o",
            "max_tokens": 500,
            "temperature": 0.7,
        })
        positive_examples = [
            'SELECT column_name FROM table_name WHERE column_name=value',
            'SELECT * FROM table_name WHERE column_name=value',
            'SELECT column_name,column_name FROM table_name WHERE column_name=value'
        ]
        negative_examples = [
            'SELECT column_name WHERE condition',
            'SELECT FROM table_name WHERE column_name = value',
            'SELECT * table_name WHERE column_name = value'
        ]
        bnf = bnf_generation(positive_examples=positive_examples, negative_examples=negative_examples, llm=llm)
        bnf = extract_code_block(bnf)
        bnf_parser = BNFParser(grammar_text=bnf)
        self.assertTrue(bnf_parser.is_correct())
        glr = GLRParser(bnf_parser)
        for example in positive_examples:
            self.assertTrue(glr.accepts_input(example))
        for example in negative_examples:
            self.assertFalse(glr.accepts_input(example))

    def test_fitness(self):
        grammar = r"""
        <entry> ::= <positive>
        <positive> ::= "0" | "5" | "9"
        """
        pos = ["0", "5", "9"]
        neg = ["1", "2", "3"]
        score = fitness(grammar, pos, neg)
        self.assertEqual(score, 6)

    def test_crossover(self):
        grammar1= r"""
        <entry> ::= <positive>
        <positive> ::= "0" | "5" | "9"
        """
        grammar2 = r"""
        <entry> ::= <positive>
        <positive> ::= "1" | "5" | "9"
        """
        grammar = crossover(grammar1, grammar2,crossover_rate=1)
        bnf_parser = BNFParser(grammar_text=grammar)
        self.assertTrue(bnf_parser.is_correct())
        self.assertTrue(bnf_parser.to_text(), grammar2)