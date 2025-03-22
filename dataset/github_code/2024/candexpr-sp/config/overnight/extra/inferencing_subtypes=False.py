import configuration

from dhnamlib.pylib.context import Environment
from dhnamlib.pylib.context import LazyEval

config = Environment(
    inferencing_subtypes=False,
    encoded_dataset_dir_path=LazyEval(lambda: configuration.config.encoded_strict_dataset_dir_path),
    shuffled_encoded_dataset_dir_path=LazyEval(lambda: configuration.config.shuffled_encoded_strict_dataset_dir_path),
)
