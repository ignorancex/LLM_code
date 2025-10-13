from unittest import TestCase

from packages.bnf.glr import GLRParser
from packages.bnf.parser import BNFParser
from packages.bnf_generation.reflexion import bnf_generation
from packages.llms.openai_model import OpenAIModel
from packages.utils.string_tool import extract_code_block


class Test(TestCase):
    def test_bnf_generation(self):
        llm = OpenAIModel()
        llm.reconfig({
            "model": "gpt-4o",
            "max_tokens": 2000,
            "temperature": 0.3,
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
        bnf, max_feedback_turns = bnf_generation(positive_examples=positive_examples, negative_examples=negative_examples,
                                                 max_turns=10,
                                                 llm=llm)
        try:
            bnf = extract_code_block(bnf)
        except:
            pass
        bnf_parser = BNFParser(grammar_text=bnf)
        glr = GLRParser(bnf_parser)
        self.assertTrue(bnf_parser.is_correct())
        for example in positive_examples:
            self.assertTrue(glr.accepts_input(example))
        for example in negative_examples:
            self.assertFalse(glr.accepts_input(example))

