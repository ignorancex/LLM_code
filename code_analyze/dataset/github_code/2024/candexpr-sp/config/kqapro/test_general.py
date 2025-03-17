
from dhnamlib.pylib.context import Environment
from dhnamlib.pylib.context import LazyEval

# from utility.time import initial_date_str

from ..common.test import get_test_dir_path


_TEST_DIR_ROOT_PATH = './model-test/kqapro'


config = Environment(
    # run_mode='test',
    test_dir_path=LazyEval(lambda: get_test_dir_path(_TEST_DIR_ROOT_PATH)),
    test_batch_size=64,
    # test_batch_size=1,
)
