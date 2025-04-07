from typing import List, Dict, Optional
import logging

from datasets import Dataset
from promptsource.templates import Template, DatasetTemplates
from transformers import PreTrainedTokenizer
import yaml
from pathlib import Path
from omegaconf import DictConfig
from copy import deepcopy
import re

from src.common import sanitize_name

logger = logging.getLogger(__name__)

DEFAULT_PROMPT_GROUP = "PROMPTSOURCE"


class PromptMapper:
    """
    Class to map a prompt to a dataset
    """

    def __init__(
            self,
            prompt: Template,
            tokenizer: PreTrainedTokenizer,
            num_proc=1,
            remove_columns=None,
            batch_size=None,
    ):
        self.prompt = prompt
        self.tokenizer = tokenizer
        self.num_proc = num_proc
        self.remove_columns = remove_columns or []
        self.batch_size = batch_size or 1

    def _preprocess_dataset(self, dataset: Dataset) -> Dataset:
        raise NotImplementedError()

    def __call__(self, task: str, dataset: Dataset) -> Dataset:
        logging.info(f"Preprocessing '{task}' with prompt {self.prompt.name}")

        # Try to preprocess the dataset with a subclass implementation, if that
        # does not work just use the original dataset.
        try:
            preprocessed = self._preprocess_dataset(dataset)
        except NotImplementedError:
            logging.debug("Reverting to default behavior for dataset")
            preprocessed = dataset

        # Map the prompt to the dataset and remove columns as needed.
        if self.batch_size > 1:
            return preprocessed.map(
                lambda b, idx: self.apply_prompt_to_batch(
                    self.prompt, b, idx
                ),
                num_proc=self.num_proc,
                remove_columns=self.remove_columns,
                batched=True,
                batch_size=self.batch_size,
                with_indices=True
            )

        return preprocessed.map(
            lambda b, idx: self.apply_prompt_to_example(
                self.prompt, b, idx
            ),
            num_proc=self.num_proc,
            remove_columns=self.remove_columns,
            with_indices=True
        )

    @staticmethod
    def apply_prompt_to_batch(prompt, batch, batch_idx):
        out = {"prompt": [], "output": [], "choices": [], 'idx': batch_idx}
        example_num = 0
        keys = list(batch.keys())
        while True:
            failed = False
            example = {}
            for k in keys:
                try:
                    example[k] = batch[k][example_num]
                except IndexError:
                    failed = True,
                    break

            if failed: break

            prompt_str, output_str = prompt.apply(example)
            choices = prompt.get_answer_choices_list(example) or []

            out['prompt'].append(prompt_str)
            out['output'].append(output_str)
            out['choices'].append(choices)
            example_num += 1

        return out

    @staticmethod
    def apply_prompt_to_example(prompt, example, idx):
        prompt_str, output_str = prompt.apply(example)
        choices = prompt.get_answer_choices_list(example) or []

        return {
            "prompt" : prompt_str,
            "output" : output_str,
            "choices": choices,
            "idx"    : idx
        }


def load_prompts(
        prompt_task: str,
        categories: List,
        prompt_filter_kwargs: Optional[Dict] = None,
        blacklist: Optional[List[str]] = None,
        default_choices: Optional[List[str]] = None,
        target_key: Optional[str] = None
):
    if (
            (default_choices is not None and target_key is None)
            or (default_choices is None and target_key is not None)
    ):
        raise ValueError("Must have default choices AND target_key")

    task_prompt_templates = DatasetTemplates(prompt_task)

    prompt_filter_kwargs = prompt_filter_kwargs or {"name_list": [], "choice_list": []}
    logger.info(f"Template Names for {prompt_task}: {task_prompt_templates.all_template_names}")

    prompts_to_use = []
    for prompt in filter_prompts(task_prompt_templates.templates, **prompt_filter_kwargs):
        if blacklist and prompt.name in blacklist:
            logger.info(f"Skipping {prompt.name}, in blacklist")
            continue

        if prompt.get_fixed_answer_choices_list() is None:
            if default_choices is None or prompt.answer_choices is None or target_key is None:
                logger.info(f"Skipping {prompt.name}, does not have fixed choices")
                continue

            prompt.answer_choices = " ||| ".join(default_choices)
            prompt_str, _ = prompt.jinja.split("|||")
            target_str = "{{ " + target_key + " }}"
            prompt.jinja = f"{prompt_str} ||| {target_str}"

        # Add the group name and the prompt name to the list of prompts to use.
        prompts_to_use.append(
            (DEFAULT_PROMPT_GROUP, prompt, {
                "name"         : prompt.name, "category": categories[0],
                "original_task": True, "prompt_task": f"Original {prompt_task}"
            })
        )

    return prompts_to_use


def filter_prompts(
        prompt_templates: Dict,
        name_list: Optional[List],
        choice_list: Optional[List],
        choices_in_prompt: Optional[bool] = None,
        original_task: Optional[bool] = None
) -> List:
    if not name_list:
        name_list = None
    if not choice_list:
        choice_list = None

    # Thought this would be easier to implement and expand on but I do not
    # think so anymore.
    filter_fn = [
        ("name_list", name_list, lambda p: name_list is None or p.name in name_list),
        ("choice_list", choice_list,
         lambda p: (
                 choice_list is None
                 or (choice_list == ['null'] and p.answer_choices is None)
                 or (p.answer_choices in choice_list))
         ),
        ("choices_in_prompt", choices_in_prompt,
         lambda p: choices_in_prompt is None or p.metadata.choices_in_prompt == choices_in_prompt
         ),
        ("original_task", original_task,
         lambda p: original_task is None or p.metadata.original_task == original_task
         ),
    ]

    # Go through the prompts and filter with each filter function. If there are
    # no more prompts left raise an error.
    out = list(prompt_templates.values())
    for name, filter_value, fn in filter_fn:
        logger.debug(f"Applying filter '{name}' with values {filter_value}")
        out = list(filter(fn, out))
        if len(out) == 0:
            raise ValueError(f"No prompts found for filter named '{name}' in "
                             f"{filter_value}")

        logger.info(f"{len(out)} prompts after filtering with '{name}'={filter_value}")

    return out


def load_answer_choice_experiment_prompts(
        prompt_dir: Path,
        prompt_cfg: DictConfig,
        category_filter: Optional[List] = None,
        answer_filter: Optional[List] = None,
        prompt_filter_kwargs: Dict = None
) -> List:
    prompt_path = prompt_dir.joinpath(prompt_cfg['file_name'])
    prompt_metadata = prompt_cfg['prompt_metadata']

    if not prompt_path.exists():
        raise FileNotFoundError(f"Could not find general prompt file {prompt_path}")

    prompt_templates = yaml.load(prompt_path.open('r'), yaml.Loader)['templates']
    logger.info(f"Found {len(prompt_templates)} prompts at {prompt_path}")

    logger.info("Validating Prompts")
    found_ids = set(prompt_templates.keys())
    expected_ids = set(prompt_metadata.keys())
    if found_ids != expected_ids:
        logger.critical(f"Missing prompts {found_ids - expected_ids} from the prompt file")
        logger.critical(f"Missing prompts {expected_ids - found_ids} from the prompt config")
        raise ValueError(f"Missing {len(found_ids ^ expected_ids)} prompts")
    out = []

    if answer_filter is not None:
        answer_filter = list(map(set, answer_filter))
        logger.info(f"Answer filter to use is {answer_filter}")

    prompt_filter_kwargs = prompt_filter_kwargs or {"name_list": [], "choice_list": []}
    for prompt in filter_prompts(prompt_templates, **prompt_filter_kwargs):
        current_prompt_metadata = prompt_metadata[prompt.id]
        if prompt.answer_choices != current_prompt_metadata['original_choices']:
            raise ValueError(f"{prompt.name} has incorrect original choices")

        if category_filter and current_prompt_metadata['category'] not in category_filter:
            logger.info(f"Skipping '{prompt.name}' as it is not in filtered category")

        for choices in prompt_cfg['possible_answer_choices']:

            # Check
            if (
                    answer_filter is not None
                    and set(l.strip() for l in choices.split("|||")) not in answer_filter
            ):
                continue

            # Deepcopy to avoid saving a mutable object that will be incorrect
            new_prompt = deepcopy(prompt)
            new_prompt.answer_choices = choices
            choices_str = map(lambda c: c.strip(), choices.split("|||"))
            out.append((
                prompt_cfg['short_name'],
                new_prompt,
                {
                    "name": f"{sanitize_name(prompt.name)}.{''.join(choices_str)}",
                    **current_prompt_metadata
                }
            ))

    if not out:
        raise ValueError("No output prompts")

    return out


def load_generalized_prompts(
        prompt_path: Path,
        task_name: str,
        choices: List = None,
        choice_str: str = None,
        mcq_choice_str: str = None,
        tasks: List[str] = None,
        prompt_filter_kwargs: Dict = None,
        use_original_choice_string: bool = False
):
    if not prompt_path.exists():
        raise FileNotFoundError(f"Could not find general prompt file {prompt_path}")

    prompt_file_dict = yaml.load(prompt_path.open('r'), yaml.Loader)
    prompt_templates = prompt_file_dict['templates']
    logger.info(f"Found {len(prompt_templates)} prompts at {prompt_path}")

    prompt_filter_kwargs = prompt_filter_kwargs or {"name_list": [], "choice_list": []}
    out = []
    choice_str_regex = re.compile(r'{{ ?choice\_string ?}}')

    for prompt in filter_prompts(prompt_templates, **prompt_filter_kwargs):
        if choices:
            prompt.answer_choices = " ||| ".join(choices)

        if not hasattr(prompt.metadata, 'is_mcq'):
            raise AttributeError(f"Missing is_mcq from prompt {prompt.id}'s metadata")

        choices_in_prompt = (
                prompt.jinja.count('answer_choices') > 1
                or prompt.jinja.count('choice_string') > 0
        )
        if prompt.metadata.choices_in_prompt and not choices_in_prompt:
            raise ValueError(f"{prompt.id} has choice in prompt, but no choices are in prompt")
        elif not prompt.metadata.choices_in_prompt and choices_in_prompt:
            raise ValueError(f"{prompt.id} has no choice in prompt, but choices are in prompt")

        if tasks is not None and prompt.metadata.original_task not in tasks:
            logger.info(f"Skipping {prompt.metadata.original_task}: {prompt.name}"
                        f" as it is not in the task filter")
            continue

        if not use_original_choice_string:
            if choice_str or mcq_choice_str:
                prompt.choice_string = choice_str
                if prompt.metadata.is_mcq:
                    prompt.choice_string = mcq_choice_str
        else:
            if prompt.name == "Sentiment with choices":
                print("??")
            if prompt.choice_string is None:
                logger.info(f"No original choices, skipping {prompt.name}")
                continue
            if not prompt.metadata.choices_in_prompt:
                logger.info(f"Skipping {prompt.name}, choices not in prompt")
                continue
            prompt.jinja = choice_str_regex.sub(prompt.choice_string, prompt.jinja)

        # CB and Anli use the same prompts, so they need to be marked as
        # original task
        original_task = prompt.metadata.original_task == task_name
        if task_name == 'super_glue/cb':
            original_task = prompt.metadata.original_task == 'anli' or original_task

        out.append((
            prompt_file_dict['short_name'],
            prompt,
            {
                "name"         : f"{sanitize_name(prompt.name)}",
                "category"     : prompt_file_dict['group_name'],
                "original_task": original_task,
                "prompt_task"  : prompt.metadata.original_task,
                "is_mcq"       : prompt.metadata.is_mcq,
                "task_mode"    : str(prompt.metadata.task_mode),
                "training_task": getattr(prompt.metadata, 'training_task', False)
            }
        ))

    if not out:
        raise ValueError("No output prompts")

    return out
