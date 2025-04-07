from collections import defaultdict

import pytest
from transformers import AutoTokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq, T5Config
from datasets import load_dataset, set_caching_enabled, Dataset
from promptsource.templates import DatasetTemplates
from src.preprocessors import ThreeChoiceEntailmentPreprocessor, TaskMode

from src import preprocessing


@pytest.mark.parametrize("use_preprocessor", [True, False], ids=["Preprocessor", "NoPreprocesser"])
def test_preprocess_dataset(use_preprocessor):
    set_caching_enabled(False)
    ds = load_dataset("anli", split="train_r1[:3]")
    tokenizer = AutoTokenizer.from_pretrained('t5-small')
    prompt_task = 'anli'
    prompt_name = 'can we infer'
    task_prompt_templates = DatasetTemplates(prompt_task)
    prompt = task_prompt_templates[prompt_name]
    prompt.metadata.task_mode = "ENTAILMENT"
    prompt.metadata.is_mcq = False
    preprocessor = None
    if use_preprocessor:
        preprocessor = ThreeChoiceEntailmentPreprocessor()
        prompt.answer_choices = " ||| ".join(preprocessor.choices)

    result, original, choice_set_tokenized = preprocessing.preprocess_dataset(
        "test",
        ds,
        tokenizer,
        prompt=prompt,
        batch_size=1,
        num_proc=1,
        preprocessor=preprocessor
    )

    assert len(result) == 3 * len(ds)
    assert len(original) == len(ds)

    if preprocessor:
        choices = preprocessor.choices
    else:
        choices = map(lambda s: s.strip(), prompt.answer_choices.split("|||"))
    assert choice_set_tokenized == list(map(
        lambda c: tokenizer(c, add_special_tokens=False)['input_ids'],
        choices
    ))

    def apply_prompt_to_ds(ex, idx):
        prompt_str, output_str = prompt.apply(ex)
        choices = prompt.get_answer_choices_list(ex) or []

        out = {
            "prompt" : prompt_str,
            "output" : output_str,
            "choices": choices,
            "idx"    : idx
        }

        if preprocessor:
            out['choice_string'] = preprocessor.choice_string
            out['domain'] = "entailment"
        return out

    expected_original = ds.map(
        apply_prompt_to_ds,
        with_indices=True,
        remove_columns=['uid', 'reason'] if preprocessor else []
    )

    expected_rank_choices = defaultdict(list)

    for i, (actual, expected) in enumerate(
            zip(original.sort('idx'), expected_original.sort('idx'))
    ):
        assert set(actual) == set(expected)

        for k, expected_value in expected.items():
            assert actual[k] == expected_value, f"result[{i}][{k}] is incorrect"

        for j, choice in enumerate(expected['choices']):
            expected_rank_choices['idx'].append([i, j])
            expected_rank_choices["inputs"].append(expected['prompt'])
            expected_rank_choices['is_correct'].append(choice == expected['output'].strip())
            expected_rank_choices['targets'].append(choice)

    expected_rank_choices = Dataset.from_dict(
        expected_rank_choices
    )
    expected_rank_choices = expected_rank_choices.map(
        lambda ex: preprocessing.tokenize_rank_choices(ex, tokenizer, 1, True),
        remove_columns=expected_rank_choices.column_names
    ).map(
        lambda ex: {'real_idx': 3 * ex['ex_idx'] + ex['choice_idx']}
    ).sort('real_idx')
    result_ds = result.map(
        lambda ex: {'real_idx': 3 * ex['ex_idx'] + ex['choice_idx']}
    ).sort('real_idx')

    for i, (actual, expected) in enumerate(zip(result_ds, expected_rank_choices)):
        assert set(actual) == set(expected)
        for k, expected_value in expected.items():
            assert actual[k] == expected_value, f"result[{i}][{k}] is incorrect"
    set_caching_enabled(True)


def test_tokenize_rank_choices():
    tokenizer = AutoTokenizer.from_pretrained('t5-small')
    ex = {
        "idx"       : [0, 1],
        "inputs"    : b"Shake and Bake",
        "targets"   : b"Ricky Bobby",
        "is_correct": True
    }

    inputs_tok = tokenizer(ex['inputs'].decode('utf-8'))
    targets_tok = tokenizer(
        ex['targets'].decode('utf-8'),
        max_length=6,
        padding="max_length"
    )

    result = preprocessing.tokenize_rank_choices(ex, tokenizer, 5)

    assert set(result) == {"idx", "input_ids", "attention_mask", "labels", "is_correct",
                           "input_len", "labels_len", "ex_idx", "choice_idx",
                           "labels_attention_mask"}

    assert result['idx'] == [0, 1]
    assert result['ex_idx'] == 0
    assert result['choice_idx'] == 1
    assert result['is_correct'] == True
    assert result['labels'] == targets_tok['input_ids']
    assert result['input_ids'] == inputs_tok['input_ids']
    assert result['attention_mask'] == inputs_tok['attention_mask']
    assert result['input_len'] == len(inputs_tok['input_ids'])
    assert result['labels_len'] == sum(targets_tok['attention_mask'])
