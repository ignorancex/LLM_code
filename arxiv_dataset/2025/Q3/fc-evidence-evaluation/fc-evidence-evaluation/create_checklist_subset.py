import json
import random

from utils import load_json_file
import argparse
import utils

parser = argparse.ArgumentParser(
    description='Checklist subset creation arguments'
)
parser.add_argument(
    '--test_path',
    default="/Users/user/Library/CloudStorage/OneDrive-King'sCollegeLondon/PycharmProjects/fc-evidence-evaluation"
            "/data/checklist_tests/test_noise_test.json",
    help='Path to file containing checklist tests.'
)
args = parser.parse_args()
_TEST_PATH = args.test_path
_SIZE = 100


def create_test_subset(path_test: str, size: int = _SIZE):
    """
    Given the path to a test, creates a subset of size 'size'.
    :param path_test: output path to save data
    :param size: size of subset
    :return:
    """
    # loads test
    tests = utils.load_json_file(path_test)
    # creates subset
    random.seed(path_test)
    subset_tests = random.sample(tests, size)

    return subset_tests
    # saves subset


def main():
    subset = create_test_subset(_TEST_PATH)
    utils.save_json_file(subset, _TEST_PATH.replace("test_", "subset_"))


if __name__ == '__main__':
    main()
