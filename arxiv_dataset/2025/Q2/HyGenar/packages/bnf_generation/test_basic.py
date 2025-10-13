from unittest import TestCase

from packages.bnf.glr import GLRParser
from packages.bnf.parser import BNFParser
from packages.bnf_generation.basic import bnf_generation
from packages.llms.openai_model import OpenAIModel
from packages.utils.string_tool import extract_code_block


class Test(TestCase):
    def test_bnf_generation(self):
        llm = OpenAIModel()
        llm.reconfig({
            "model":"gpt-4o",
            "max_tokens": 500,
            "temperature": 0,
        })
        positive_examples = [
            "a = 1",
            "a = 2",
            "b = 2"
        ]
        negative_examples = [
            "a = y",
            "b=3",
            "c=9"
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

