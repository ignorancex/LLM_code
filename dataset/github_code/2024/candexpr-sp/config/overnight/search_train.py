
import itertools

from dhnamlib.pylib.context import Environment
from dhnamlib.pylib.context import LazyEval
from dhnamlib.pylib.iteration import not_none_valued_pairs

# from .train_general import get_model_learning_dir_path
from .train_general import config as train_general_config
from ..common.search_train import parse_args


common_config = Environment(
    # max_search_optim_loops=float('inf'),
    max_search_optim_loops=16,
)


search_config = Environment(
    search_batch_size=8,
    num_search_beams=8,
)

_TRAIN_BATCH_SIZE = 16

optim_config = Environment(
    # Imported values
    train_general_config.items(),

    # Updated values
    num_train_epochs=8,
    train_batch_size=_TRAIN_BATCH_SIZE,
    train_max_num_action_seqs=_TRAIN_BATCH_SIZE,
    val_batch_size=64,
    using_scheduler=False,
    learning_rate=2e-5,

    extra_weaksup_collection_keys=['domain'],
)

config = Environment(not_none_valued_pairs(
    itertools.chain(
        common_config.items(),
        search_config.items(),
        optim_config.items(),
        parse_args().items()
    ),

    # Others
    run_mode='search-train',
))
