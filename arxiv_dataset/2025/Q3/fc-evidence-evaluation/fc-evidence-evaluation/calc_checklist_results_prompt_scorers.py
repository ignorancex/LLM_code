"""
Script to calcualte difference between base data and tests for prompt scorer
before running this script furst genrate results with run_checklist_tests.py
"""
import os.path

import pandas as pd
import properties
import utils



_OUTPUT_DIR_PATH = "/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation/results/checklist"
_RESULTS_INPUT_FILE = os.path.join(_OUTPUT_DIR_PATH, "results_bleu.csv")
_RESULTS_OUTPUT_FILE = os.path.join(_OUTPUT_DIR_PATH, "results_diff_bleu.csv")
_SCORER = properties.PromptTypes("bleu")


def main():
    # load results file
    results = pd.read_csv(_RESULTS_INPUT_FILE)

    # extract score for base data
    if _SCORER == properties.PromptTypes.ATOMIC_REFERENCE_FACTS_PREC_RECALL:
        # get base score used for calculating difference
        base_prec = \
        results.loc[(results['test'] == 'base_data') & (results['scorer'] == 'atomic_reference_prec')]['score'].values[0]
        base_recall = results.loc[(results['test'] == 'base_data') & (results['scorer'] == 'atomic_reference_recall')][
            'score'].values[0]
    else:
        base = results.loc[(results['test'] == 'base_data')]['score'].values[0]
    results_diff = []

    for i, row in results.iterrows():
        if row['test'] == 'base_data':
            continue
        if _SCORER == properties.PromptTypes.ATOMIC_REFERENCE_FACTS_PREC_RECALL:
            if "_prec" in row['scorer']:
                base = base_prec
            else:
                base = base_recall
        # no else, as base has been set above for other scorers
        results_diff.append(
            {'test': row['test'],
             'scorer': row['scorer'],
             'diff': utils.percentage_difference(base, row['score'])})
    pd.DataFrame(results_diff).to_csv(_RESULTS_OUTPUT_FILE, index=False)


if __name__ == '__main__':
    main()
