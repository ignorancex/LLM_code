from packages.bnf.parser import BNFParser
from packages.llms.base_model import BaseModel
from packages.utils.string_tool import extract_code_block


def bnf_generation(positive_examples: list[str], negative_examples: list[str], max_turns:int, llm: BaseModel) -> (str,int):
    """
    (Integrated Reflexion to enhance BNF generation) Prompt an LLM to generate BNF when given a set of positive and negative examples.
    :param positive_examples: a list of positive examples
    :param negative_examples: a list of negative examples
    :param max_turns: maximum number of turns to reflect
    :param llm: the LLM model
    :return: a tuple of the generated BNF and the number of turns used for feedback
    """
    positive_examples = '\n'.join([f"Example {i}:\n{example}" for i, example in enumerate(positive_examples)])
    negative_examples = '\n'.join([f"Example {i}:\n{example}" for i, example in enumerate(negative_examples)])
    init_prompt = r"""
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
    # with feedback from BNF parser
    feedback_prompt = r'''
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

===Generated BNF===
{}

===Feedback===
The generated BNF grammar has incorrect syntax and please consider fixing it by referring to the feedback.
Here is the feedback from the BNF parser:
{}
    '''.strip()
    # Generate initial BNF
    llm.reconfig({
        "max_tokens": 2000,
        "temperature": 0.3,
    })
    try:
        bnf = llm.chat(init_prompt)
    except Exception:
        bnf = ""
    used_feedback_turns = 0
    # Loop to get feedback
    for i in range(max_turns):
        # check whether needing to extract code block
        try:
            bnf = extract_code_block(bnf)
        except Exception:
            pass
        # feedback from parser
        try:
            BNFParser(bnf).is_correct()
        except SyntaxError as e:
            feedback = feedback_prompt.format(positive_examples,negative_examples,bnf,e)
            try:
                bnf = llm.chat(feedback)
            except Exception:
                bnf = ""
            used_feedback_turns += 1
            continue
        break
    return bnf,used_feedback_turns
