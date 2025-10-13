"""
Script to run checklist tests (see create_checklist_evaluation_tests.py) against prompt and baseline scorers.
"""
import argparse
import os.path
import statistics
from typing import List

import openai
import pandas as pd

import prompt_scorer_openai
import properties
import utils

parser = argparse.ArgumentParser(
    description='Checklist tests.'
)
parser.add_argument(
    '--test_type',
    default="base_data",  # set to vitaminc if jsonl file with claim, evidence, label entries in dicts.
    choices=[tt.value for tt in properties.TestType],
    help='Type of checklist test. Entry base implies the base data used for checklist generation.'
)

_TESTS_DIR_PATH = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/data/checklist_tests"
_OUTPUT_DIR_PATH = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/results/checklist"
_INIT_DATA_PATH = os.path.join(_TESTS_DIR_PATH, "base_data.json")

args = parser.parse_args()
print("test type: {}".format(args.test_type))
_TEST_TYPE = properties.TestType(args.test_type)
print("test type: {}".format(_TEST_TYPE.value))
_TEST_FILE_PATH = os.path.join(_TESTS_DIR_PATH, "{}.json".format(_TEST_TYPE.value))

_SCORER = properties.PromptTypes("hmeteor")
_PROMPTING_MODEL = properties.ModelApi.GPT4o
if _SCORER in [properties.PromptTypes.METEOR, properties.PromptTypes.ROUGE, properties.PromptTypes.HMETEOR,
               properties.PromptTypes.BLEU]:
    _OUTPUT_FILE = "predictions_{}_{}.jsonl".format(_SCORER.value, _TEST_TYPE.value)
    _RESULTS_OUTPUT_FILE = os.path.join(_OUTPUT_DIR_PATH, "results_{}.csv".format(_SCORER.value))
else:
    _OUTPUT_FILE = "predictions_{}_{}_{}.jsonl".format(_SCORER.value, _PROMPTING_MODEL, _TEST_TYPE.value)
    _RESULTS_OUTPUT_FILE = os.path.join(_OUTPUT_DIR_PATH, "results_{}_{}.csv".format(_SCORER.value, _PROMPTING_MODEL))

_LOAD_RESULTS = False

_KEY = open('/Users/user/Desktop/openai_key_fc_eval.txt', 'r').read()
_CLIENT = openai.OpenAI(
    api_key=_KEY,
    timeout=10,
)


def _prepare_dataset(test_df: pd.DataFrame, prev_results: list[properties.OpenAIResponse]) -> list[
    properties.AveritecEntry]:
    """
    Based on averitec input data, prepares a list of properties.AveritecEntry
    :param test_df:
    :param prev_results:
    :return:
    """
    input_data = []
    prev_claims = [x.claim for x in prev_results]

    for i, row in test_df.iterrows():
        # if model has been previously prompted for this entry -> skip
        if row['claim'] in prev_claims:
            continue
        input_data.append(properties.AveritecEntry(claim=row['claim'],
                                                   label=row['gold label'],
                                                   evidence=row['reference evidence'],
                                                   id=row['id']
                                                   ))
    return input_data


def _run_prompt_scorer(init_data_samples: list[properties.AveritecEntry], test_samples: list[properties.AveritecEntry],
                       prompt_type: properties.PromptTypes,
                       output_path: str, logprob=False) -> list[properties.OpenAIResponse]:
    # run prompt scorer
    return prompt_scorer_openai.prompt_api_model(init_data_samples, test_samples, prompt_type,
                                                 _CLIENT, match_system_preds=False,
                                                 responses_output_path=output_path, logprob=logprob,
                                                 api=_PROMPTING_MODEL)


def _calc_test_results(scorer_results, results_df) -> pd.DataFrame:
    """
    Returns average score across all scored samples. Then compares the obtained score to scores for init_data
    :param scorer_results:
    :param results_df:
    :return:
    """
    if _SCORER == properties.PromptTypes.ATOMIC_REFERENCE_FACTS_PREC_RECALL:
        new_rows = [
            {
                'test': _TEST_TYPE.value,
                'scorer': _SCORER.value.replace("_recall", ""),
                'score': statistics.mean([x.response['precision'] for x in scorer_results])
            },
            {
                'test': _TEST_TYPE.value,
                'scorer': _SCORER.value.replace("_prec", ""),
                'score': statistics.mean([x.response['recall'] for x in scorer_results]),
            }
        ]
        results_df = pd.concat([results_df, pd.DataFrame(new_rows)], ignore_index=True)
    else:
        new_rows = [
            {
                'test': _TEST_TYPE.value,
                'scorer': _SCORER.value,
                'score': statistics.mean([x.response['score'] for x in scorer_results])
            }
        ]
        results_df = pd.concat([results_df, pd.DataFrame(new_rows)], ignore_index=True)

    return results_df


def _calculate_scores_baseline_metrics(base_data: List[properties.AveritecEntry],
                                       checklist_tests: List[properties.AveritecEntry],
                                       scorer: properties.PromptTypes) -> list[
    properties.OpenAIResponse]:
    results = []
    for i, entry in enumerate(checklist_tests):
        if scorer == properties.PromptTypes.METEOR:
            score = utils.calc_meteor(reference=base_data[i].evidence, candidate=entry.evidence)
        elif scorer == properties.PromptTypes.HMETEOR:
            score = utils.calc_hungarian_meteor(candidate=entry.evidence,
                                                reference=base_data[i].evidence)
        elif scorer == properties.PromptTypes.BLEU:
            score = utils.calc_bleu(candidate=entry.evidence, reference=base_data[i].evidence)
        elif scorer == properties.PromptTypes.ROUGE:
            score = utils.calc_rouge(candidate=entry.evidence, reference=base_data[i].evidence)
        else:
            raise Exception("scorer equal {} not part of baseline scorers.".format(scorer))
        results.append(properties.OpenAIResponse(claim=entry.claim, evidence=entry.evidence,
                                                 response={'reference_evidence': base_data[i].evidence,
                                                           'score': score},
                                                 gold=entry.label, id=entry.id))
    return results


def main():
    scorer_output_path = os.path.join(_OUTPUT_DIR_PATH, _OUTPUT_FILE)

    # load initial data and test file
    base_data = utils.load_averitec_base(_INIT_DATA_PATH, extract_evidence=True)
    tests = utils.load_averitec_base(_TEST_FILE_PATH, extract_evidence=True)

    # load results file and add a new row r = (scorer, spearman, pearson)
    results = pd.read_csv(os.path.join(_OUTPUT_DIR_PATH, _RESULTS_OUTPUT_FILE))

    # run scorer and save results
    if _LOAD_RESULTS:
        model_results_scores = utils.load_jsonl_file(scorer_output_path,
                                                     dataclass=properties.OpenAIResponse)
    else:
        if _SCORER in [properties.PromptTypes.METEOR, properties.PromptTypes.ROUGE, properties.PromptTypes.HMETEOR,
                       properties.PromptTypes.BLEU]:
            model_results_scores = _calculate_scores_baseline_metrics(base_data, tests, _SCORER)
        else:
            model_results = _run_prompt_scorer(base_data, tests, prompt_type=_SCORER,
                                               output_path=scorer_output_path)
            model_results_scores = prompt_scorer_openai.calculate_prediction_scores(input_data=None,
                                                                                    preds=model_results,
                                                                                    prompt_type=_SCORER)
            utils.save_jsonl_file(model_results_scores, scorer_output_path)

    # calculate changes in scores by comparing to base dataset
    results = _calc_test_results(model_results_scores, results)

    # save results
    results.to_csv(os.path.join(_OUTPUT_DIR_PATH, _RESULTS_OUTPUT_FILE), index=False)


if __name__ == '__main__':
    main()
