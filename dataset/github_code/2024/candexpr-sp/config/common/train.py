
import os

from configuration import has_model_learning_dir_path
from utility.time import initial_date_str


def get_model_learning_dir_path(model_learning_dir_root_path):
    if has_model_learning_dir_path():
        return None
    else:
        return os.path.join(model_learning_dir_root_path, initial_date_str)
