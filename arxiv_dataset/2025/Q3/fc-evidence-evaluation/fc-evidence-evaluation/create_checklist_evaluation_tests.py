import argparse
import os
import random

import checklist_evaluation
import utils
from utils import load_json_file
from properties import TestType

parser = argparse.ArgumentParser(
    description='Checklist evaluation arguments'
)

parser.add_argument(
    '--input',
    default="/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation"
            "/data/averitec/averitec_test.json",
    help='Path input data for checklist test creation.'
)
parser.add_argument(
    '--output',
    default="/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation"
            "/data/checklist_tests",
    help='Output path for created checklist tests.'
)
parser.add_argument(
    '--type',
    default="number_replace",
    type=TestType,
    choices=list(TestType),
    help='Type of checklist tests to create.'
)

args = parser.parse_args()
_INPUT_PATH = args.input
_TEST_TYPE = args.type
_OUTPUT_PATH = os.path.join(args.output, "{}.json".format(_TEST_TYPE.value))

# path do dataset sample used as input for test creation
_PATH_BASE_DATA = os.path.join(args.output, "base_data.json")

# DONT CHANGE SEED!
random.seed(10)


def create_checklist_tests(input_data):
    """
    Creates checklist tests.
    :param input_data: input file (e.g., Averitec test) based on which the tests are created.
    :return: test samples
    """
    if _TEST_TYPE == TestType.ROBUST_NOISE:
        return checklist_evaluation.robustness_noise_test(input_data)
    elif _TEST_TYPE == TestType.COHERENCE:
        return checklist_evaluation.coherence_test(input_data)
    elif _TEST_TYPE == TestType.COVERAGE:
        return checklist_evaluation.coverage_drop_answers_test(input_data)
    elif _TEST_TYPE == TestType.SYNONYMS:
        return checklist_evaluation.invariance_synonym_test(input_data)
    elif _TEST_TYPE == TestType.RAND_ORDER:
        return checklist_evaluation.informativeness_random_word_order_test(input_data)
    elif _TEST_TYPE == TestType.SUMMARY:  # todo filter
        return checklist_evaluation.create_sum_test(input_data)
    elif _TEST_TYPE == TestType.CONTRACTION:
        return checklist_evaluation.invariance_contraction_test(input_data)
    elif _TEST_TYPE == TestType.NUM2TEXT:
        return checklist_evaluation.invariance_num2text_test(input_data)
    elif _TEST_TYPE == TestType.TEXT2NUM:
        return checklist_evaluation.invariance_text2num_test(input_data)
    elif _TEST_TYPE == TestType.FLUENCY_TYPOS:
        return checklist_evaluation.fluency_typos_test(input_data)
    elif _TEST_TYPE == TestType.FLUENCY_WORDS_DROP:
        return checklist_evaluation.fluency_drop_words_test(input_data)
    elif _TEST_TYPE == TestType.ENTITY_SWAP:
        return checklist_evaluation.informativeness_entity_swap_test(input_data)
    elif _TEST_TYPE == TestType.REDUNDANCY_WORDS:
        return checklist_evaluation.redundancy_duplicate_words_test(input_data)
    elif _TEST_TYPE == TestType.REDUNDANCY_SENT:
        return checklist_evaluation.redundancy_duplicate_sentence_test(input_data)
    elif _TEST_TYPE == TestType.NUMBER_REPLACE:
        return checklist_evaluation.informativeness_number_change_test(input_data)


def main():
    # if subset base data => load, otherwise create from entire Averitec test
    if _PATH_BASE_DATA:
        base_data_subset = utils.load_json_file(_PATH_BASE_DATA)
    else:
        base_data = load_json_file(_INPUT_PATH)
        # subset used for test creation
        base_data_subset = random.sample(base_data, 100)
        # save subset
        utils.save_json_file(base_data_subset, _PATH_BASE_DATA)

    # create tests
    test_samples = create_checklist_tests(base_data_subset)
    print("Number of {} test samples created is: {}".format(_TEST_TYPE.value, len(test_samples)))

    # save tests
    utils.save_json_file(test_samples, _OUTPUT_PATH)


if __name__ == '__main__':
    main()
