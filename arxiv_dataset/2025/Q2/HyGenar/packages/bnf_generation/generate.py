from typing import Any

from packages.bnf.parser import BNFParser
from packages.bnf_generation import basic, reflexion, genetic
from packages.llms.base_model import BaseModel
from packages.utils.log import logger
from packages.utils.string_tool import extract_code_block


def generate(example: tuple[list[str], list[str]], method: str, llm: BaseModel) -> (
        str, Any):
    """
    Generate BNF using the given method on the given examples and LLM
    :param example: A tuple of positive and negative examples
    :param method: The method to be used for BNF generation
    :param llm: The LLM to be evaluated
    :return: Generated BNF and additional information
    """
    positive_examples, negative_examples = example
    additional_info = None
    if method == "basic":
        bnf = basic.bnf_generation(positive_examples=positive_examples, negative_examples=negative_examples, llm=llm)
    elif method == "reflexion":
        bnf, used_turns = reflexion.bnf_generation(positive_examples=positive_examples,
                                                   negative_examples=negative_examples, llm=llm, max_turns=5)
        additional_info = {
            "max_feedback_turns": used_turns
        }
    elif method == "genetic":
        bnf = genetic.bnf_generation(positive_examples=positive_examples, negative_examples=negative_examples, llm=llm)
    else:
        raise ValueError("Invalid method for BNF generation.")
    try:
        bnf = extract_code_block(bnf)
    except Exception as e:
        logger.error(e, exc_info=True)
        pass
    return bnf, additional_info
