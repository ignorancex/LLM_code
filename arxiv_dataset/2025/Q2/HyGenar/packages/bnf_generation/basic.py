from packages.llms.base_model import BaseModel


def bnf_generation(positive_examples: list[str], negative_examples: list[str], llm: BaseModel) -> str:
    """
    Prompt an LLM to generate BNF when given a set of positive and negative examples.
    :param positive_examples: a list of positive examples
    :param negative_examples: a list of negative examples
    :param llm: the LLM model
    :return:
    """
    positive_examples = '\n'.join([f"Example {i}:\n{example}" for i, example in enumerate(positive_examples)])
    negative_examples = '\n'.join([f"Example {i}:\n{example}" for i, example in enumerate(negative_examples)])
    prompt = r"""
Given a set of positive and negative examples, generate the Backus–Naur Form (BNF) grammar that accepts all positive examples and rejects all negative examples.
1. Only generate the standard BNF grammar;
2. The generated BNF grammar MUST accept all positive examples and reject all negative examples;
3. Each terminal symbol MUST be quoted with double quotes and MUST NOT escape double quotes or pipeline in terminal symbols;
4. Pay special attention to whether spaces, line breaks, or other special symbols are required between each symbol, and if so, these need to be explicitly specified, e.g. <term> ::= "1" "+" "2" can handle "1+2" but not "1 + 2" while <term> ::= "1" " " "+" " " "2" can handle "1 + 2" but not "1+2";
5. The entry point of the generated BNF grammar MUST be the non-terminal symbol in the first production rule;
6. Only the generated BNF should be wrapped in a pair of triple backtick;
7. Do NOT output any additional texts, comments, or explanations.

===Positive Examples===
{}
===Negative Examples===
{}
    """.format(positive_examples, negative_examples).strip()
    llm.reconfig({
        "max_tokens": 2000,
        "temperature": 0,
    })
    try:
        bnf = llm.chat(prompt)
    except Exception:
        bnf = ""
    return bnf
