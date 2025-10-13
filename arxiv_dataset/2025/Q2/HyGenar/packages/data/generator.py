from packages.llms.base_model import BaseModel


def generate_bnfs(n_of_lines: int, n_of_bnfs: int, llm: BaseModel) -> list[str]:
    """
    Generate a list of random standard Backus-Naur Form (BNF) grammar with given number of lines.
    :param n_of_lines: number of lines
    :param n_of_bnfs: number of BNFs to generate
    :param llm: language model
    :return: generated BNF
    """
    prompt = r"""
Generate a list of random standard Backus-Naur Form (BNF) grammar with the following constraints:
1. Each generated BNF grammar MUST be SELF-CONTAINED and VALID, which means it should be able to recognize a valid string;
2. Each generated BNF grammar MUST have exactly {} lines;
3. Each generated BNF grammar MUST be unique;
4. Each generated BNF grammar MUST be separated by a newline in addition to the linebreak;
5. For each generated BNF grammar, the entry point MUST be at the first line;
6. Only generate {} BNF grammars;
7. Only output BNF grammars WITHOUT any additional text or code block, like "```".
    """.format(n_of_lines, n_of_bnfs).strip()
    bnfs = llm.chat(prompt)
    bnfs = bnfs.strip().split("\n\n")
    return bnfs


def generate_examples_based_on_bnf(bnf: str, n_of_examples: int, llm: BaseModel) -> tuple[list[str], list[str]]:
    """
    Generate a list of positive and negative examples from the given BNF grammar.
    :param bnf: BNF grammar
    :param n_of_examples: number of examples to generate
    :param llm: language model
    :return: generated examples
    """

    # Generate positive examples
    pos_prompt = r"""
Generate a list of positive examples with the following constraints:
1. Each example MUST be separated by a newline in addition to the linebreak;
2. Only output examples WITHOUT any additional text or code block, like "```";
3. Only output {} examples;
4. Each example MUST be generated based on the given BNF grammar;
5. Pay attention to whether the whitespaces are allowed between symbols.

For example, given the following BNF grammar:
<term> ::=  "0" | "1" | "2"
you should output positive examples like:
0

1

2

Then, the given BNF grammar is:
{}
    """.strip().format(n_of_examples, bnf).strip()
    pos_examples = llm.chat(pos_prompt)
    pos_examples = pos_examples.strip().split("\n\n")

    # Generate negative examples
    neg_prompt = r"""
Generate a list of negative examples with the following constraints:
1. Each example MUST be separated by a newline in addition to the linebreak;
2. Only output examples WITHOUT any additional text or code block, like "```";
3. Only output {} examples;
4. Each example MUST be generated based on the given BNF grammar;
5. Each example should be greatly related to the given BNF grammar, but ensure it is NOT a valid string for the given BNF grammar.

For example, given the following BNF grammar:
<term> ::=  "0" | "1" | "2"
you should output negative examples like:
6

*

9

Then, the given BNF grammar is:
{}
    """.strip().format(n_of_examples, bnf).strip()
    neg_examples = llm.chat(neg_prompt)
    neg_examples = neg_examples.strip().split("\n\n")

    return pos_examples, neg_examples
