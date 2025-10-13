from distutils.fancy_getopt import neg_alias_re
from typing import TypeAlias

from packages.llms.base_model import BaseModel

# Examples: first is a list of positive examples, second is a list of negative examples
Examples: TypeAlias = (list[str], list[str])


class GenerateBNFFromExamples:
    """
    Given a language model and a set of examples, generate a BNF grammar
    """

    def __init__(self, llm: BaseModel, examples: Examples):
        self.llm = llm
        self.examples = examples

    def generate(self) -> str:
        """
        Generate a BNF grammar from the examples
        :return: bnf
        """
        prompt = prompt_template(self.examples[0], self.examples[1])
        bnf = self.llm.chat(prompt)
        return bnf


def prompt_template(pos: list[str], neg: list[str]) -> str:
    """
    Given a list of positive and negative examples, generate a prompt for the LLM to generate a BNF grammar
    :param pos: positive examples
    :param neg: negative examples
    :return: prompt
    """
    pos_examples = []
    for idx, pos_example in enumerate(pos):
        example = f"Positive Example {idx}:\n" \
                  f"{pos_example}\n"
        pos_examples.append(
            example
        )
    neg_examples = []
    for idx, neg_example in enumerate(neg):
        example = f"Negative Example {idx}:\n" \
                  f"{neg_example}\n"
        neg_examples.append(
            example
        )

    return "Given the following positive and negative examples, generate a BNF grammar: \n\n" \
           "======Positive Examples======\n" \
           "{positive_examples}\n\n" \
           "======Negative Examples======\n" \
           "{negative_examples}\n\n".format(
        positive_examples="\n".join(pos_examples),
        negative_examples="\n".join(neg_examples)
    ).strip()
