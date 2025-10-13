import json
import os.path
from concurrent.futures.thread import ThreadPoolExecutor
from concurrent.futures import as_completed, Future
from typing import List, Tuple, Dict
from rich.progress import Progress, TaskID, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

import typer
from packages.utils.log import logger

from packages.bnf_generation.generate import generate
from packages.llms.choose import choose_model
from packages.utils.file import DATA_PROCESSED_DIR, RESULTS_FILE_PATH_TEMPLATE

bnfs = typer.Typer(help="Generate BNFs based on positive and negative examples.")


def check_existing_results(methods: List[str], llms: List[str]) -> List[Tuple[str, str]]:
    """
    Get the list of existing results files.
    :param methods: a list of methods
    :param llms: a list of llms
    :return: a list of tuples of method and llm, e.g [('basic', 'gpt-4o'), ('reflexion', 'gpt-4o')]
    """
    exists_results = []
    for method in methods:
        for llm in llms:
            results_file_path = RESULTS_FILE_PATH_TEMPLATE.format(method=method, llm=llm)
            if os.path.exists(results_file_path):
                print(f"Results file exists: {results_file_path}")
                exists_results.append((method, llm))
    return exists_results


def load_data(data_file_path: str) -> dict:
    with open(data_file_path, 'r') as f:
        data = json.load(f)
    return data


def count_examples(data: dict) -> int:
    """
    Count the number of examples/challenges in the data.
    :param data: a dictionary of data
    :return: the number of examples
    """
    len_of_examples = 0
    for n_of_production_rules, bnfs_with_examples in data.items():
        for bnf_with_examples in bnfs_with_examples:
            examples = bnf_with_examples['examples']
            len_of_examples += len(examples)
    return len_of_examples


def generate_bnf_with_one_method_one_llm_model(method: str, llm: str, data_file_path: str, progress: Progress,
                                               task_id: TaskID) -> None:
    """
    Generate BNFs with one method and one LLM model and save the results.(This function is used for parallel generating BNFs)
    :param method: a method for generating BNFs
    :param llm: a LLM model for generating BNFs
    :param data_file_path: a data file path
    :param progress: a progress handler
    :param task_id: a task id for progress
    :return:
    """
    data = load_data(data_file_path)
    # choose model
    model = choose_model(model_name=llm)
    # start generating
    for n_of_production_rules, bnfs_with_examples in data.items():
        for bnf_with_examples in bnfs_with_examples:
            examples = bnf_with_examples['examples']
            bnf_with_examples.setdefault('generation', [])
            for example in examples:
                positive_examples = example['positive_examples']
                negative_examples = example['negative_examples']
                logger.info(f"""
                Generating BNFs with example:
                Positive Examples: 
                {positive_examples}
                Negative Examples: 
                {negative_examples}
                """.strip())
                bnf, additional_info = generate(
                    example=(positive_examples, negative_examples),
                    method=method,
                    llm=model,
                )
                bnf_with_examples['generation'].append(
                    {
                        'bnf': bnf,
                        'additional_info': additional_info
                    }
                )
                progress.update(task_id=task_id, advance=1)
    # save results
    results_file_path = RESULTS_FILE_PATH_TEMPLATE.format(method=method, llm=llm)
    with open(results_file_path, 'w') as f:
        json.dump(data, f, indent=4)


@bnfs.command()
def generate_bnfs():
    """
    Generate BNFs for different methods and LLMs.
    """
    # methods: a list of methods for generating BNFs
    methods = [
        'basic',
        'reflexion',
        'genetic'
    ]

    # llms: a list of llms for BNFs generation
    llms = [
        # OpenAI
        'gpt-3.5-turbo',
        'gpt-4o',

        # Open source
        'llama3:70b-instruct',
        'qwen:72b-chat',
        'gemma2:27b-instruct-fp16',
        'starcoder2:instruct',
        'codestral',
        'mistral:7b-instruct',
    ]

    # check if the results files exists
    exists_results = check_existing_results(methods=methods, llms=llms)
    if exists_results:
        response = typer.prompt(
            f"Some results files already exist, do you want to continue and skip generating existed results? [Y/n]")
        if response.lower() != 'y':
            return

    # generate
    data_file_path = os.path.join(DATA_PROCESSED_DIR, "bnfs_with_examples.json")  # data file path
    length_examples = count_examples(load_data(data_file_path))
    for llm in llms:  # check each llm
        typer.echo(f"")
        typer.echo(f"===Start {llm}===")
        remaining_methods = []
        for method in methods:  # check each method
            # check whether to skip
            if (method, llm) in exists_results:
                typer.echo(
                    f"Skip generating BNFs based on \"{method}\" Method for \"{llm}\" LLM since it is already exists.")
                continue
            else:
                remaining_methods.append(method)
        with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
        ) as progress:  # progress
            tasks = {}  # tasks
            for method in remaining_methods:  # add tasks as progress bars
                tasks[method] = progress.add_task(f"Generating BNFs based on ({method},{llm})", total=length_examples,
                                                  completed=True)

            # generate BNFs in parallel for a single LLM model with multiple methods
            with ThreadPoolExecutor(max_workers=max(3, len(remaining_methods))) as executor:
                future_to_task: Dict[Future, Tuple[str, str]] = {}
                # submit async tasks
                for method in remaining_methods:
                    future = executor.submit(generate_bnf_with_one_method_one_llm_model, method=method, llm=llm,
                                             data_file_path=data_file_path, progress=progress, task_id=tasks[method])
                    future_to_task[future] = (method, llm)
                # wait for the results
                for future in as_completed(future_to_task.keys()):
                    method, llm = future_to_task[future]
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Failed to generate BNFs based on \"{method}\" Method for \"{llm}\" LLM.")
                        logger.error(f"Error: {e}", exc_info=True)
        typer.echo(f"===End {llm}===")
