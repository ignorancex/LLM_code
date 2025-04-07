import pytest
from transformers import AutoTokenizer, T5ForConditionalGeneration, DataCollatorForSeq2Seq, T5Config
from datasets import set_caching_enabled
from pathlib import Path
from omegaconf import OmegaConf
from src.common.util import PROJECT_ROOT
import shutil
from promptsource.templates import DatasetTemplates
from functools import partial
import yaml
from src import prompt_map
from src.common import sanitize_name


@pytest.mark.parametrize("name", [None, "Jeopardy style", "N/A"],
                         ids=["AllName", "Name", "ErrorName"])
@pytest.mark.parametrize("choice", [None, "null", "No ||| Yes", "N/A"],
                         ids=["AllChoice", "NullChoice", "Choice", "ErrorChoice"])
@pytest.mark.parametrize("choices_in_prompt", [None, True, False],
                         ids=["AllChoicePrompt", "ChoiceInPrompt", "ChoiceNotInPrompt"])
@pytest.mark.parametrize("original_task", [None, True, False],
                         ids=["AllOrigTask", "OrigTask", "NotOrigTask"])
def test_filter_prompts(name, choice, choices_in_prompt, original_task):
    prompts = DatasetTemplates("wiki_qa")
    expected = set(prompts.all_template_names)
    all_templates = prompts.all_template_names
    name_expected = {
        "None"          : set(all_templates),
        "Jeopardy style": {"Jeopardy style"},
        "N/A"           : set()
    }
    choice_expected = {
        "None"      : set(all_templates),
        "null"      : {'Direct Answer to Question', 'Jeopardy style',
                       'Topic Prediction - Question and Answer Pair',
                       'Topic Prediction - Answer Only', 'Generate Question from Topic',
                       'Topic Prediction - Question Only'},
        "No ||| Yes": {
            'found_on_google',
            'Is This True?',
            'automatic_system',
            'Decide_good_answer'
        },
        "N/A"       : set()
    }
    choices_in_prompt_expected = {
        "None" : set(all_templates),
        "True" : {'exercise', 'found_on_google', 'Decide_good_answer'},
        "False": set(all_templates) - {'exercise', 'found_on_google', 'Decide_good_answer'}
    }
    original_task_expected = {
        "None" : set(all_templates),
        "True" : {'Decide_good_answer', 'Direct Answer to Question', 'found_on_google', 'exercise',
                  'automatic_system', 'Is This True?'},
        "False": {'Jeopardy style', 'Topic Prediction - Question and Answer Pair',
                  'Topic Prediction - Answer Only', 'Generate Question from Topic',
                  'Topic Prediction - Question Only'}
    }

    expected = expected.intersection(name_expected[str(name)])
    expected = expected.intersection(choice_expected[str(choice)])
    expected = expected.intersection(choices_in_prompt_expected[str(choices_in_prompt)])
    expected = expected.intersection(original_task_expected[str(original_task)])

    if not expected:
        with pytest.raises(ValueError):
            prompt_map.filter_prompts(
                prompt_templates=prompts.templates,
                name_list=[name] if name is not None else [],
                choice_list=[choice] if choice is not None else [],
                choices_in_prompt=choices_in_prompt,
                original_task=original_task
            )
        return

    result = prompt_map.filter_prompts(
        prompt_templates=prompts.templates,
        name_list=[name] if name is not None else [],
        choice_list=[choice] if choice is not None else [],
        choices_in_prompt=choices_in_prompt,
        original_task=original_task
    )

    result_names = set(map(lambda p: p.name, result))

    assert result_names == expected


def load_answer_choice_experiment_prompts(tmpdir):
    tmpdir_path = Path(tmpdir)
    sample_prompts = PROJECT_ROOT.joinpath("test_fixtures", "3fc_prompt_sample.yaml")
    shutil.copyfile(sample_prompts, tmpdir_path.joinpath('prompts.yaml'))
    prompt_templates = yaml.load(sample_prompts.open('r'), yaml.Loader)['templates']
    prompt_cfg = OmegaConf.create({
        "name"                   : "3 Choice Entailment",
        "short_name"             : "3CE",
        "type"                   : "fixed_choice",
        "file_name"              : "prompts.yaml",
        "choice_chount"          : "3",
        "possible_answer_choices": [
            "Yes ||| Maybe ||| No",
            "Correct ||| Inconclusive ||| Incorrect",
            "True ||| Inconclusive ||| False",
            "Always ||| Sometimes ||| Never",
        ],
        "prompt_metadata"        : {
            "9532d63e-7996-4cee-a1e3-014fb19802e5": {
                "original_choices": "Correct ||| Inconclusive ||| Incorrect",
                "original_task"   : ["anli", "cb"],
                "category"        : "Fixed Choice"
            },
            "e2433288-fdd8-4bd1-8eca-a2739b1d3101": {
                "original_choices": "Yes ||| Maybe ||| No",
                "original_task"   : ["anli", "cb"],
                "category"        : "Fixed Choice"
            },
            "5a65c67f-ec9c-44f1-a610-63a7d1d016d0": {
                "original_choices": "Always ||| Sometimes ||| Never",
                "original_task"   : ["quail"],
                "category"        : "Multiple Choice"
            },
            "7b0ce9fa-6aa0-4210-ab6c-1edd4b2f43df": {
                "original_choices": "True ||| Inconclusive ||| False",
                "original_task"   : ["quail"],
                "category"        : "Open Ended"
            }
        }
    })

    raw_result = prompt_map.load_answer_choice_experiment_prompts(
        tmpdir_path,
        prompt_cfg
    )
    assert len(raw_result) == 16

    result = {}
    for c, p, m in raw_result:
        if p.id not in result:
            result[p.id] = {}
        result[p.id][p.answer_choices] = (c, p, m)

    for k, v in prompt_cfg['prompt_metadata'].items():
        expected_choices = set(prompt_cfg['possible_answer_choices'])
        assert k in result, k
        assert set(result[k].keys()) == expected_choices
        prompt = prompt_templates[k]

        for i, x in enumerate(list(result[k].values())):
            c, p, m = x
            assert c == prompt_cfg['short_name']
            answer_choices = prompt_cfg['possible_answer_choices'][i]

            choices_str = map(lambda _c: _c.strip(), answer_choices.split("|||"))
            assert m == {
                "name": f"{sanitize_name(prompt.name)}.{''.join(choices_str)}",
                **v
            }
            assert p.answer_choices == answer_choices


@pytest.mark.parametrize('task_name', ['craigslist_bargains', 'test'])
def test_load_general_prompts(task_name):
    sample_prompts = PROJECT_ROOT.joinpath("test_fixtures", "general_fixed_choice.yaml")
    prompts = prompt_map.load_generalized_prompts(
        prompt_path=sample_prompts,
        task_name=task_name,
        choices=["Yes", "No", "Maybe"],
        choice_str="Yes,no,maybe",
        mcq_choice_str="A)YesB)NoC)Maybe"
    )

    expected_prompts = yaml.load(
        sample_prompts.open('r'),
        yaml.Loader
    )['templates']

    for k in expected_prompts:
        expected_prompts[k].answer_choices = " ||| ".join(["Yes", "No", "Maybe"])
        expected_prompts[k].choice_string = "Yes,no,maybe"
        if expected_prompts[k].metadata.is_mcq:
            expected_prompts[k].choice_string = "A)YesB)NoC)Maybe"

    assert len(prompts) == 4

    for g, prompt, prompt_dict in prompts:
        assert g == "GenFC"
        expected_prompt = expected_prompts[prompt.id]
        assert prompt.answer_choices == expected_prompt.answer_choices
        assert prompt.jinja == expected_prompt.jinja
        assert prompt.name == expected_prompt.name
        assert prompt.choice_string == expected_prompt.choice_string
        assert prompt_dict['name'] == f"{sanitize_name(prompt.name)}"
        assert prompt_dict['category'] == "GeneralFixedChoice"
        assert prompt_dict['original_task'] == (task_name == 'craigslist_bargains')
        assert prompt_dict['prompt_task'] == 'craigslist_bargains'
        assert prompt_dict['task_mode'] == 'CLASSIFICATION'
