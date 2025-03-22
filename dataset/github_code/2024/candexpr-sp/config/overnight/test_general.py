
import argparse
from functools import cache

from dhnamlib.pylib.context import Environment
from dhnamlib.pylib.context import LazyEval
from dhnamlib.pylib.iteration import not_none_valued_dict

import configuration

# from utility.time import initial_date_str

from ..common.test import get_test_dir_path


_TEST_DIR_ROOT_PATH = './model-test/overnight'


@cache
def parse_args():
    parser = argparse.ArgumentParser(description='Testing a semantic parser')
    parser.add_argument('--single-domain', dest='single_domain',
                        choices=configuration.config.all_domains,
                        help='a domain used during training')
    args, unknown = parser.parse_known_args()

    return not_none_valued_dict(vars(args))


def _get_test_domains():
    arg_dict = parse_args()
    if 'single_domain' in arg_dict:
        test_domains = [arg_dict['single_domain']]
    else:
        test_domains = configuration.config.all_domains
    return test_domains


config = Environment(not_none_valued_dict(
     # run_mode='test',
     test_dir_path=LazyEval(lambda: get_test_dir_path(_TEST_DIR_ROOT_PATH)),
     test_domains=LazyEval(_get_test_domains),
     test_batch_size=64,
     # test_batch_size=1,
))
