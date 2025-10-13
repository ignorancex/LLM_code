import random
from copy import deepcopy
from typing import List, Union, Final

from packages.bnf.glr import GLRParser
from packages.bnf.parser import BNFParser
from packages.llms.base_model import BaseModel
from packages.utils.string_tool import extract_code_block
from packages.utils.log import logger


def prepare_examples(examples:list[str]) -> str:
    """
    Prepare examples for prompting.
    :param examples: a list of positive/negative examples
    :return:
    """
    return '\n'.join([f"Example {i}:\n{example}" for i, example in enumerate(examples)])


def generate_parser_from_bnf(bnf_grammar: str) -> Union[BNFParser, None]:
    """
    Generate parser from BNF.
    :param bnf_grammar: the BNF grammar
    :return: the BNF parser or None
    """
    try:
        parser = BNFParser(bnf_grammar)
        if not parser.is_correct():  # should check if the BNF is correct
            return None
        return parser
    except Exception as e:
        logger.error(e, exc_info=True)
        return None


def fitness(bnf_grammar: str, positive_examples: list[str], negative_examples: list[str]) -> int:
    """
    Calculate the fitness score of a BNF grammar based on the given positive and negative examples.
    Fitness Score = Number of positive examples accepted + Number of negative examples rejected
    :param bnf_grammar: the BNF grammar
    :param positive_examples: a list of positive examples
    :param negative_examples: a list of negative examples
    :return: fitness score
    """
    score = 0
    try:
        bnf_grammar = extract_code_block(bnf_grammar)
    except Exception as e:
        logger.error(e, exc_info=True)
        pass
    parser = generate_parser_from_bnf(bnf_grammar)
    if parser is None:
        logger.error("Invalid grammar.")
        return -1  # Invalid grammar
    logger.info(f"Generated parser: {parser.to_text()}")
    try:
        logger.info("Creating GLR parser.")
        glr = GLRParser(parser)
        logger.info("Created GLR parser.")
    except Exception as e:
        logger.error(e, exc_info=True)
        return -1
    logger.info(f"Checking positive examples: {positive_examples}")
    for example in positive_examples:
        try:
            logger.info(f"Checking example: {example}")
            result = glr.accepts_input(str(example))
            logger.info(f"Result: {result}")
        except Exception as e:
            logger.error(e, exc_info=True)
            result = False
        if result:
            score += 1
    logger.info(f"Checking negative examples: {negative_examples}")
    for example in negative_examples:
        try:
            logger.info(f"Checking example: {example}")
            result = glr.accepts_input(str(example))
            logger.info(f"Result: {not result}")
        except Exception as e:
            logger.error(e, exc_info=True)
            result = False
        if not result:
            score += 1
    logger.info(f"Fitness score: {score}")
    return score


def mutate(bnf_grammar: str, llm: BaseModel, positive_examples: str, negative_examples: str) -> str:
    """
    Mutate the BNF grammar by changing a line or using LLM to modify the grammar.
    :param bnf_grammar: the BNF grammar
    :param llm: the LLM model
    :param positive_examples: the positive examples
    :param negative_examples: the negative examples
    :return: the mutated BNF grammar
    """
    try:
        bnf_grammar = extract_code_block(bnf_grammar)
    except Exception as e:
        logger.error(e, exc_info=True)
        pass
    bnf_parser = generate_parser_from_bnf(bnf_grammar)
    if random.random() < 0.5 and bnf_parser is not None:  # only mutate the BNF grammar if it is valid
        logger.info("Mutate BNF grammar by changing a line.")
        # change a line in the BNF grammar
        rules = bnf_parser.get_rules()
        if len(rules) > 0:
            key = random.choice(list(rules.keys()))
            rules[key] = mutate_alternatives(rules[key])
            bnf_parser = BNFParser.create_parser_from_rules(rules)
            return bnf_parser.to_text()
    else:
        logger.info("Mutate BNF grammar using LLM.")
        # Use LLM to modify the grammar
        mutation_prompt = f"""
Modify the following BNF grammar slightly to improve its acceptance of the positive examples and rejection of the negative examples.

===BNF Grammar===
{bnf_grammar}

===Positive Examples===
{positive_examples}
===Negative Examples===
{negative_examples}

Only output the modified BNF grammar wrapped in triple backticks.
"""
        try:
            mutated_bnf = llm.chat(mutation_prompt)
            try:
                mutated_bnf = extract_code_block(mutated_bnf)
            except Exception as e:
                logger.error(e, exc_info=True)
                pass
        except Exception as e:
            logger.error(e, exc_info=True)
            return bnf_grammar
        return mutated_bnf
    return bnf_grammar


def mutate_alternatives(alternatives: list[list[str]]) -> list[list[str]]:
    """
    Mutate a line in the BNF grammar: swap symbols (terminals and non-terminals) or add " " between each symbol.
    :param alternatives: the list of alternatives in the line
    :return: the mutated alternatives
    """
    new_alternatives = deepcopy(alternatives)
    for alternative in new_alternatives:
        random.shuffle(alternative)
        if random.random() < 0.1:
            inject_indices = random.sample(range(len(alternative)), random.randint(0, len(alternative)))
            for index in inject_indices:
                alternative.insert(index, ' ')
    return new_alternatives


def crossover(bnf1: str, bnf2: str, crossover_rate: float) -> str:
    """
    Perform crossover between two BNF grammars: choose a random point to separate given two BNFs and combine the two BNFs.
    :param bnf1: the first BNF grammar
    :param bnf2: the second BNF grammar
    :param crossover_rate: the crossover rate
    :return: the combined BNF grammar
    """
    # if the BNF is not valid, return the other one
    try:
        bnf1 = extract_code_block(bnf1)
    except Exception as e:
        logger.error(e, exc_info=True)
        pass
    try:
        bnf2 = extract_code_block(bnf2)
    except Exception as e:
        logger.error(e, exc_info=True)
        pass
    bnf1_parser = generate_parser_from_bnf(bnf1)
    bnf2_parser = generate_parser_from_bnf(bnf2)
    if bnf1_parser is None and bnf2_parser is None:
        return bnf1 if random.random() < 0.5 else bnf2
    if bnf1_parser is None:
        return bnf2
    if bnf2_parser is None:
        return bnf1
    # if the two BNFs are both valid, perform crossover
    bnf1 = bnf1_parser.to_text()
    bnf2 = bnf2_parser.to_text()
    if random.random() > crossover_rate:  # do not perform crossover
        return bnf1
    lines1 = bnf1.strip().split('\n')
    lines2 = bnf2.strip().split('\n')
    min_length = min(len(lines1), len(lines2))
    if min_length < 2:
        return bnf1 if random.random() < 0.5 else bnf2
    crossover_point = random.randint(1, min_length - 1)
    new_lines = lines1[:crossover_point] + lines2[crossover_point:]
    return '\n'.join(new_lines)


def bnf_generation(positive_examples: List[str], negative_examples: List[str], llm: BaseModel) -> str:
    """
    Use a genetic algorithm to generate BNF when given a set of positive and negative examples.
    :param positive_examples: a list of positive examples
    :param negative_examples: a list of negative examples
    :param llm: the LLM model
    :return: the optimized BNF grammar
    """

    # Parameters for the genetic algorithm
    POPULATION_SIZE: Final[int] = 10
    GENERATIONS: Final[int] = 5
    MUTATION_RATE: Final[float] = 0.3
    CROSSOVER_RATE: Final[float] = 0.7

    # LLM
    llm.reconfig({
        "max_tokens": 2000,
        "temperature": 0.7,
    })

    pos_examples_str = prepare_examples(positive_examples)
    neg_examples_str = prepare_examples(negative_examples)

    # Initial prompt template
    prompt_template = r"""
Given a set of positive and negative examples, generate a Backus–Naur Form (BNF) grammar that accepts all positive examples and rejects all negative examples.
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
    """.strip()

    # Initialize population (use LLMs)
    population = []

    for _ in range(POPULATION_SIZE):
        prompt = prompt_template.format(pos_examples_str, neg_examples_str)
        try:
            bnf = llm.chat(prompt)
            try:
                bnf = extract_code_block(bnf)
            except Exception as e:
                logger.error(e, exc_info=True)
                pass
        except Exception as e:
            logger.error(e, exc_info=True)
            bnf = ""
        # Early stopping if the BNF is perfect
        score = fitness(bnf, positive_examples=positive_examples, negative_examples=negative_examples)
        if score == len(positive_examples) + len(negative_examples):
            return f"```{bnf}```"
        population.append(bnf)
    logger.info(f"Initial population: {population}")

    # Evolutionary loop
    best_bnf = None
    best_fitness = -1

    for generation in range(GENERATIONS):
        logger.info(f"Generation {generation + 1}/{GENERATIONS}")
        # Evaluate fitness
        fitness_scores = []
        for bnf in population:
            score = fitness(bnf, positive_examples=positive_examples, negative_examples=negative_examples)
            fitness_scores.append((score, bnf))
            if score > best_fitness:
                best_fitness = score
                best_bnf = bnf
        logger.info(f"Evaluation: {fitness_scores}")
        # Check termination condition
        if best_fitness == len(positive_examples) + len(negative_examples):
            logger.info("Found perfect.")
            break  # Found perfect grammar

        # Selection: keep the top half of the population
        fitness_scores.sort(reverse=True)
        selected = [bnf for _, bnf in fitness_scores[:POPULATION_SIZE // 2]]

        # Generate new population
        new_population = []
        while len(new_population) < POPULATION_SIZE:
            parent1 = random.choice(selected)
            parent2 = random.choice(selected)
            child = crossover(parent1, parent2, crossover_rate=CROSSOVER_RATE)
            if random.random() < MUTATION_RATE:
                child = mutate(child, llm=llm, positive_examples=pos_examples_str, negative_examples=neg_examples_str)
            new_population.append(child)
            logger.info(f"New child: {child}")
        population = new_population

    logger.info(f"Best fitness: {best_fitness}")
    return f"```{best_bnf}```"
