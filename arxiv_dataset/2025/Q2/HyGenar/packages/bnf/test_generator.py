from unittest import TestCase

from packages.bnf.generator import prompt_template, GenerateBNFFromExamples
from packages.bnf.parser import BNFParser
from packages.llms.openai_model import OpenAIModel


class TestPromptTemplate(TestCase):
    def test_prompt_template(self):
        pos = [
            "1+1",
            "2+2"
        ]
        neg = [
            "1*2",
            "2-1",
            "3/1"
        ]
        expected = """\
Given the following positive and negative examples, generate a BNF grammar: 

======Positive Examples======
Positive Example 0:
1+1

Positive Example 1:
2+2


======Negative Examples======
Negative Example 0:
1*2

Negative Example 1:
2-1

Negative Example 2:
3/1
        """.strip()
        prompt = prompt_template(pos, neg)
        self.assertEqual(expected, prompt)


class TestGenerateBNFFromExamples(TestCase):
    def test_generate(self):
        pos = [
            "1+1",
            "2+2"
        ]
        neg = [
            "1*2",
            "2-1",
            "3/1"
        ]
        expected = """\
<expression> ::= <number> "+" <number>

<number> ::= "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"
        """.strip()
        llm = OpenAIModel()
        llm.reconfig({
            "max_tokens": 100,
            "temperature": 0,
        })
        parser = GenerateBNFFromExamples(llm, (pos, neg))
        bnf = parser.generate().strip()
        self.assertEqual(expected, bnf)