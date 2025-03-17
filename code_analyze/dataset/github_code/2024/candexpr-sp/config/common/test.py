
import os
import argparse

import configuration

from dhnamlib.pylib.context import Environment
from dhnamlib.pylib.context import LazyEval
from dhnamlib.pylib.filesys import get_new_path_with_number


_TEST_DIR_ROOT_PATH = './model-test'


def get_test_dir_path(test_dir_root_path):
    def test_dir_name_to_path(test_dir_name):
        dir_name_sep = '#'
        common_test_dir_path = os.path.join(test_dir_root_path, test_dir_name + dir_name_sep)
        new_test_dir_path = get_new_path_with_number(common_test_dir_path)
        return new_test_dir_path

    if 'model_learning_dir_path' in configuration.config:
        test_dir_name = os.path.basename(configuration.config.model_learning_dir_path)
        test_dir_path = test_dir_name_to_path(test_dir_name)
    else:
        assert 'model_path' in configuration.config
        parser = argparse.ArgumentParser(description='Testing a semantic parser')
        parser.add_argument('--test-dir-name', dest='test_dir_name', help='a name of the directory for test output')
        parser.add_argument('--test-dir', dest='test_dir_path', help='a path to the directory for test output')
        args, unknown = parser.parse_known_args()

        if args.test_dir_path is not None:
            assert args.test_dir_name is None
            test_dir_path = args.test_dir_path
        else:
            assert args.test_dir_name is not None
            test_dir_name = args.test_dir_name
            test_dir_path = test_dir_name_to_path(test_dir_name)

    if os.path.exists(test_dir_path):
        assert os.path.isdir(test_dir_path), 'The path is to a file rather than a directory'
        assert len(os.listdir(test_dir_path)) == 0, 'A test directory should be empty: {}'.format(test_dir_path)

    # os.makedirs(test_dir_path)

    return test_dir_path
