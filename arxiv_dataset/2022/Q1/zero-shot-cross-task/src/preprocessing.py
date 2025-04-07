import logging
from collections import defaultdict
from typing import Tuple, Optional
from promptsource.templates import Template
from t5.data.preprocessors import rank_classification
from datasets import Dataset
import json
import numpy as np
import tensorflow as tf

from src.common import all_equal
from src.prompt_map import PromptMapper
from src.preprocessors import FixedChoiceTaskPreprocessor, TaskMode

logger = logging.getLogger(__name__)


def preprocess_dataset(
        task,
        dataset,
        tokenizer,
        prompt: Template,
        num_proc=4,
        batch_size=1,
        preprocessor: Optional[FixedChoiceTaskPreprocessor] = None,
):
    # If we are using a preprocessor, make sure it has the necessary custom
    # attributes for use with the preprocessor. This is a soft check to make
    # sure the prompts are generalizable.
    if preprocessor:
        if not hasattr(prompt.metadata, 'task_mode'):
            raise AttributeError(f"Trying to use a preprocessor but prompt "
                                 f"'{prompt.name}' does not have a task mode")

        if not hasattr(prompt.metadata, 'is_mcq'):
            raise AttributeError(f"Trying to use a preprocessor but prompt "
                                 f"'{prompt.name}' does not have an 'is_mcq'.")

        preprocessor.is_mcq = prompt.metadata.is_mcq
        preprocessor.set_mode(TaskMode[prompt.metadata.task_mode])
        preprocessed_dataset = dataset.map(
            preprocessor,
            with_indices=True,
            remove_columns=dataset.column_names
        )
    else:
        preprocessed_dataset = dataset

    # Map the prompt across the entire dataset.
    prompt_mapper = PromptMapper(
        prompt,
        tokenizer,
        num_proc=num_proc,
        batch_size=batch_size
    )
    ds_with_prompt = prompt_mapper(task, preprocessed_dataset)

    # The scope of this project is only fixed choice tasks where the choice is
    # constant across all examples. Thus we check here to make sure that is the
    # case.
    choice_set = prompt.get_fixed_answer_choices_list()
    assert choice_set is not None, "Not fixed choice task."
    logger.info(f"Found choice set of [{', '.join(choice_set)}]")
    choice_set_tokenized = list(map(
        lambda c: tokenizer(c, add_special_tokens=False)['input_ids'],
        choice_set
    ))

    logger.info(f"Tokenizing the dataset")

    # Need to convert it to a TF dataset so that I can use T5's rank
    # classification processor.
    tensor_slices = {column: ds_with_prompt[column] for column in ds_with_prompt.column_names}
    tf_dataset = tf.data.Dataset.from_tensor_slices(
        tensor_slices
    )
    data = rank_classification(
        tf_dataset,
        inputs_fn=lambda ex: tf.fill((len(choice_set),), ex["prompt"]),
        targets_fn=lambda ex: choice_set,
        is_correct_fn=lambda ex: tf.equal(choice_set, tf.strings.strip(ex['output'])),
        mode='eval'
    )

    dataset_records = defaultdict(list)
    for d in data.as_numpy_iterator():
        for k, v in d.items():
            dataset_records[k].append(v)

    final_dataset = Dataset.from_dict(dataset_records).map(
        lambda ex: tokenize_rank_choices(ex, tokenizer, max(map(len, choice_set_tokenized))),
        num_proc=num_proc,
        remove_columns=list(dataset_records)
    ).sort('input_len', reverse=True)
    return final_dataset, ds_with_prompt, choice_set_tokenized


def tokenize_rank_choices(
        ex,
        tokenizer,
        max_choice_len,
        is_string_input=False,
):
    inputs_tokenized = tokenizer(
        ex['inputs'].decode('utf-8') if not is_string_input else ex['inputs'],
        max_length=1024, truncation=True)
    max_label_len = max_choice_len + 1
    labels_tokenized = tokenizer(
        ex['targets'].decode('utf-8') if not is_string_input else ex['targets'],
        max_length=max_label_len + (max_label_len % 2),
        padding="max_length"
    )
    return {
        "idx"                  : ex['idx'],
        "ex_idx"               : ex['idx'][0],
        "choice_idx"           : ex['idx'][1],
        "is_correct"           : ex['is_correct'],
        "labels"               : labels_tokenized['input_ids'],
        "labels_attention_mask": labels_tokenized['attention_mask'],
        "labels_len"           : sum(labels_tokenized['attention_mask']),
        "input_len"            : len(inputs_tokenized['input_ids']),
        **inputs_tokenized
    }


def create_masked_inputs(
        columns_required,
        use_t5_additional_special_tokens=False
):
    out = {}
    
