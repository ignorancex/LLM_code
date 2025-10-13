import json
import os.path

import click
import typer
from tqdm import tqdm

from packages.bnf.glr import GLRParser
from packages.bnf.parser import BNFParser
from packages.data.generator import generate_examples_based_on_bnf
from packages.llms.openai_model import OpenAIModel
from packages.utils.file import DATA_RAW_DIR, DATA_PROCESSED_DIR
from packages.data import generator

data = typer.Typer(help="Data utils: generation and checking.")


@data.command()
def generate_bnfs():
    """
       Generate 10 raw BNFs for each 1-9 number of lines.
       :return:
       """
    # Number of BNFs to generate
    number_of_bnfs = 10
    range_of_lines = range(1, 10)

    # Check if BNFs file already generated
    bnfs_file_path = os.path.join(DATA_RAW_DIR, 'bnfs.json')
    if os.path.exists(bnfs_file_path):
        click.echo(f'BNFs already generated at {bnfs_file_path}')
        return

    # Generate BNFs
    json_data = {}
    llm = OpenAIModel()
    llm.reconfig({
        "model": "gpt-4o",
        "max_tokens": 2000,
        "temperature": 0.1,
    })
    for number_of_lines in tqdm(range_of_lines):
        bnfs = generator.generate_bnfs(n_of_bnfs=number_of_bnfs, n_of_lines=number_of_lines,
                                       llm=llm)
        key = f'{number_of_lines}'
        json_data[key] = bnfs
    typer.echo(
        f'Generated {number_of_bnfs} BNFs for each number of lines in range {range_of_lines}')

    # Save to file
    with open(bnfs_file_path, 'w') as f:
        json.dump(json_data, f, indent=4)
    typer.echo(f'Saving to {bnfs_file_path}')


@data.command()
def check_bnfs(expected_n: int = 10):
    """
    Check the syntax correctness of processed BNFs.
    :param expected_n: expected number of bnfs for each number of lines
    :return:
    """
    incorrect_bnfs = []
    incorrect_n_of_bnfs = []
    duplicate_bnfs = []
    bnfs_file_path = os.path.join(DATA_PROCESSED_DIR, 'bnfs.json')
    with open(bnfs_file_path, 'r') as f:
        data = json.load(f)
        for key, bnfs in data.items():
            if len(bnfs) != len(set(bnfs)):
                duplicate_bnfs.append(f'Duplicate BNFs for {key} lines')
            if len(bnfs) != int(expected_n):
                incorrect_n_of_bnfs.append(
                    f'Error in the number of BNFs for {key} lines: expected {expected_n} but got {len(bnfs)}')
            for i, bnf in enumerate(bnfs):
                try:
                    bnf = BNFParser(grammar_text=bnf)
                    correctness = bnf.is_correct()
                    is_correct = correctness
                    if len(bnf.get_rules()) != int(key):
                        is_correct = False
                except Exception as e:
                    is_correct = False

                if not is_correct:
                    message = f'Error in BNF {i} with {key} lines: invalid BNF'
                    incorrect_bnfs.append(message)
                    typer.echo(message)
            # typer.echo()

    typer.echo("===Accuracy Summary===")
    if incorrect_bnfs or incorrect_n_of_bnfs:
        if incorrect_n_of_bnfs:
            typer.echo("Some BNFs have incorrect number of BNFs:")
            for incorrect_n_of_bnf in incorrect_n_of_bnfs:
                typer.echo(incorrect_n_of_bnf)
            typer.echo()
        if incorrect_bnfs:
            typer.echo("Some BNFs are incorrect")
            typer.echo("Please check the following BNFs:")
            for incorrect_bnf in incorrect_bnfs:
                typer.echo(incorrect_bnf)
    else:
        typer.echo('All BNFs are correct.')

    typer.echo()
    typer.echo("===Duplication Summary===")
    if duplicate_bnfs:
        typer.echo("Some BNFs are duplicated:")
        for duplicate_bnf in duplicate_bnfs:
            typer.echo(duplicate_bnf)
    else:
        typer.echo("No duplicated BNFs.")


@data.command()
def pretty_print_bnf(bnf_idx: int, n_of_lines: str, raw: bool = True):
    """
    Pretty print the BNF in the processed BNFs file
    :param bnf_idx: bnf index
    :param n_of_lines: number of lines
    :return:
    """
    bnfs_file_path = os.path.join(DATA_PROCESSED_DIR, 'bnfs.json')
    with open(bnfs_file_path, 'r') as f:
        bnfs = json.load(f)
        bnf = bnfs[str(n_of_lines)][int(bnf_idx)]
        if raw:
            typer.echo(bnf)
            return
        bnf = BNFParser(grammar_text=bnf)
        bnf.pretty_print()


@data.command()
def generate_examples():
    """
    Generate 6 examples for each BFN with each example having 3 positive and 3 negative examples for target BNF from processed BNFs.
    :return:
    """
    n_of_examples_each_bnf = 6
    n_of_pnexamples_each_example = 3

    # raw bnfs_with_examples.json path
    bnfs_with_examples_file_path = os.path.join(DATA_RAW_DIR, 'bnfs_with_examples.json')
    if os.path.exists(bnfs_with_examples_file_path):
        typer.echo(f'Examples already generated at {bnfs_with_examples_file_path}')
        return

    # processed bnfs.json path
    bnfs_file_path = os.path.join(DATA_PROCESSED_DIR, 'bnfs.json')
    # load json
    with open(bnfs_file_path, 'r') as f:
        bnfs = json.load(f)
    bnfs_with_examples = {}
    for n_of_lines, bnfs_list in bnfs.items():
        typer.echo(f'===Generating examples for BNFs with {n_of_lines} lines===')
        value = bnfs_with_examples.setdefault(n_of_lines, [])
        for bnf in tqdm(bnfs_list):
            bnf_info = {}
            bnf_info['ref_bnf'] = bnf
            bnf_info['examples'] = []
            for _ in range(n_of_examples_each_bnf):
                llm = OpenAIModel()
                llm.reconfig({
                    "model": "gpt-4o",
                    "max_tokens": 2000,
                    "temperature": 1,
                })
                positive_examples, negative_examples = generate_examples_based_on_bnf(bnf=bnf,
                                                                                      n_of_examples=n_of_pnexamples_each_example,
                                                                                      llm=llm)
                example = {
                    'positive_examples': positive_examples,
                    'negative_examples': negative_examples
                }
                bnf_info['examples'].append(example)
            value.append(bnf_info)

    # save to file
    with open(bnfs_with_examples_file_path, 'w') as f:
        json.dump(bnfs_with_examples, f, indent=4)

    typer.echo(
        f"Generated {n_of_examples_each_bnf} examples for each BFN with each example having {n_of_pnexamples_each_example} positive and {n_of_pnexamples_each_example} negative examples for the target BNF.")
    typer.echo(f'Saving to {bnfs_with_examples_file_path}')


@data.command()
def check_examples():
    """
    Check whether processed examples correctly.
    :return:
    """
    n_of_pnexamples_each_example = 3
    # processed bnfs_with_examples.json path
    bnfs_with_examples_file_path = os.path.join(DATA_PROCESSED_DIR, 'bnfs_with_examples.json')
    with open(bnfs_with_examples_file_path, 'r') as f:
        data = json.load(f)
    for n_of_lines, bnfs_list in data.items():
        typer.echo(f'===Checking examples for BNFs with {n_of_lines} lines===')
        for idx, bnf_info in enumerate(bnfs_list):
            bnf = bnf_info['ref_bnf']
            bnf_parser = BNFParser(grammar_text=bnf)
            glr = GLRParser(bnf_parser)
            examples = bnf_info['examples']
            for jdx, example in enumerate(examples):
                positive_examples = example['positive_examples']
                negative_examples = example['negative_examples']
                # Check the number of positive and negative examples
                if len(positive_examples) != n_of_pnexamples_each_example or len(
                        negative_examples) != n_of_pnexamples_each_example:
                    typer.echo(
                        f'For BNF at index {idx} and Example at index {jdx}: Error in the number of positive examples (n={len(positive_examples)}) or negative examples (n={len(negative_examples)}), but expect n={n_of_pnexamples_each_example}.')
                    exit(0)
                # Check the correctness of positive examples
                for i, positive_example in enumerate(positive_examples):
                    # typer.echo(f'p:{idx,jdx,i}')
                    if not glr.accepts_input(positive_example):
                        typer.echo(
                            f'For BNF at index {idx} and Example at index {jdx}: Error in the positive example {i}')
                        exit(0)
                # Check the correctness of negative examples
                for i, negative_example in enumerate(negative_examples):
                    # typer.echo(f'n:{idx,jdx,i}')
                    if glr.accepts_input(negative_example):
                        typer.echo(
                            f'For BNF at index {idx} and Example at index {jdx}: Error in the negative example {i}')
                        exit(0)

    typer.echo('All examples are correct.')


@data.command()
def pretty_print_bnf_in_examples_file(bnf_idx: int, n_of_lines: str, raw: bool = True):
    """
    Pretty print the BNF in the processed BNFs file
    :param bnf_idx: bnf index
    :param n_of_lines: number of lines
    :return:
    """
    bnfs_file_path = os.path.join(DATA_PROCESSED_DIR, 'bnfs_with_examples.json')
    with open(bnfs_file_path, 'r') as f:
        bnfs = json.load(f)
        bnf = bnfs[str(n_of_lines)][int(bnf_idx)]['ref_bnf']
        if raw:
            typer.echo(bnf)
            return
        bnf = BNFParser(grammar_text=bnf)
        bnf.pretty_print()