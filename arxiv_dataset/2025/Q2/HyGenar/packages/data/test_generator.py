from unittest import TestCase

from PIL.IcnsImagePlugin import read_png_or_jpeg2000

from packages.data.generator import generate_bnfs, generate_examples_based_on_bnf
from packages.llms.openai_model import OpenAIModel


class Test(TestCase):
    def test_generate_bnfs(self):
        llm = OpenAIModel()
        llm.reconfig({
            "model": "gpt-4o",
            "max_tokens": 200,
            "temperature": 0,
        })
        n_of_production_rules = 2
        n_of_bnfs = 5
        bnfs = generate_bnfs(n_of_production_rules, n_of_bnfs, llm)
        self.assertEqual(len(bnfs), n_of_bnfs)

    def test_generate_examples_based_on_bnf_complexity_0(self):
        llm = OpenAIModel()
        llm.reconfig({
            "model": "gpt-4o",
            "max_tokens": 500,
            "temperature": 0,
        })
        bnf = r"""
    <term> ::=  "0" | "1" | "2"
        """
        examples = generate_examples_based_on_bnf(bnf, 5, llm)
        positive_examples = examples[0]
        negative_examples = examples[1]
        self.assertEqual(len(positive_examples), 5)
        self.assertEqual(len(negative_examples), 5)

    def test_generate_examples_based_on_bnf_complexity_1(self):
        llm = OpenAIModel()
        llm.reconfig({
            "model": "gpt-4o",
            "max_tokens": 500,
            "temperature": 0,
        })
        bnf = r"""
        <expr> ::= <term> "+" <expr> | <term>
        <term> ::= <factor> "*" <term> | <factor>
        <factor> ::= "(" <expr> ")" | <number>
        <number> ::= <digit> <number> | <digit>
        <digit> ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
        """
        examples = generate_examples_based_on_bnf(bnf, 5, llm)
        positive_examples = examples[0]
        negative_examples = examples[1]
        self.assertEqual(len(positive_examples), 5)
        self.assertEqual(len(negative_examples), 5)


