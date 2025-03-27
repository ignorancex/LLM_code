
from dhnamlib.pylib.context import Environment, LazyEval
from dhnamlib.pylib.filesys import jsonl_load


_augmented_ns_train_set_file_path = './preprocessed/kqapro/augmented_ns_train.jsonl'
_augmented_ns_val_set_file_path = './preprocessed/kqapro/augmented_ns_val.jsonl'
_encoded_ns_train_set_file_path = './preprocessed/kqapro/encoded_ns_train.jsonl'
_encoded_ns_val_set_file_path = './preprocessed/kqapro/encoded_ns_val.jsonl'
_shuffled_augmented_ns_train_set_file_path = './preprocessed/kqapro/shuffled_augmented_ns_train.jsonl'
_shuffled_encoded_ns_train_set_file_path = './preprocessed/kqapro/shuffled_encoded_ns_train.jsonl'

_augmented_cnlts_train_set_file_path = './preprocessed/kqapro/augmented_cnlts_train.jsonl'
_augmented_cnlts_val_set_file_path = './preprocessed/kqapro/augmented_cnlts_val.jsonl'
_encoded_cnlts_train_set_file_path = './preprocessed/kqapro/encoded_cnlts_train.jsonl'
_encoded_cnlts_val_set_file_path = './preprocessed/kqapro/encoded_cnlts_val.jsonl'
_shuffled_augmented_cnlts_train_set_file_path = './preprocessed/kqapro/shuffled_augmented_cnlts_train.jsonl'
_shuffled_encoded_cnlts_train_set_file_path = './preprocessed/kqapro/shuffled_encoded_cnlts_train.jsonl'

_augmented_ns_cnlts_train_set_file_path = './preprocessed/kqapro/augmented_ns_cnlts_train.jsonl'
_augmented_ns_cnlts_val_set_file_path = './preprocessed/kqapro/augmented_ns_cnlts_val.jsonl'
_encoded_ns_cnlts_train_set_file_path = './preprocessed/kqapro/encoded_ns_cnlts_train.jsonl'
_encoded_ns_cnlts_val_set_file_path = './preprocessed/kqapro/encoded_ns_cnlts_val.jsonl'
_shuffled_augmented_ns_cnlts_train_set_file_path = './preprocessed/kqapro/shuffled_augmented_ns_cnlts_train.jsonl'
_shuffled_encoded_ns_cnlts_train_set_file_path = './preprocessed/kqapro/shuffled_encoded_ns_cnlts_train.jsonl'

_augmented_nao_train_set_file_path = './preprocessed/kqapro/augmented_nao_train.jsonl'
_augmented_nao_val_set_file_path = './preprocessed/kqapro/augmented_nao_val.jsonl'
_encoded_nao_train_set_file_path = './preprocessed/kqapro/encoded_nao_train.jsonl'
_encoded_nao_val_set_file_path = './preprocessed/kqapro/encoded_nao_val.jsonl'
_shuffled_augmented_nao_train_set_file_path = './preprocessed/kqapro/shuffled_augmented_nao_train.jsonl'
_shuffled_encoded_nao_train_set_file_path = './preprocessed/kqapro/shuffled_encoded_nao_train.jsonl'


config = Environment(
    augmented_ns_train_set=LazyEval(lambda: jsonl_load(_augmented_ns_train_set_file_path)),
    augmented_ns_val_set=LazyEval(lambda: jsonl_load(_augmented_ns_val_set_file_path)),
    encoded_ns_train_set=LazyEval(lambda: jsonl_load(_encoded_ns_train_set_file_path)),
    encoded_ns_val_set=LazyEval(lambda: jsonl_load(_encoded_ns_val_set_file_path)),
    shuffled_augmented_ns_train_set=LazyEval(lambda: jsonl_load(_shuffled_augmented_ns_train_set_file_path)),
    shuffled_encoded_ns_train_set=LazyEval(lambda: jsonl_load(_shuffled_encoded_ns_train_set_file_path)),

    augmented_cnlts_train_set=LazyEval(lambda: jsonl_load(_augmented_cnlts_train_set_file_path)),
    augmented_cnlts_val_set=LazyEval(lambda: jsonl_load(_augmented_cnlts_val_set_file_path)),
    encoded_cnlts_train_set=LazyEval(lambda: jsonl_load(_encoded_cnlts_train_set_file_path)),
    encoded_cnlts_val_set=LazyEval(lambda: jsonl_load(_encoded_cnlts_val_set_file_path)),
    shuffled_augmented_cnlts_train_set=LazyEval(lambda: jsonl_load(_shuffled_augmented_cnlts_train_set_file_path)),
    shuffled_encoded_cnlts_train_set=LazyEval(lambda: jsonl_load(_shuffled_encoded_cnlts_train_set_file_path)),

    augmented_ns_cnlts_train_set=LazyEval(lambda: jsonl_load(_augmented_ns_cnlts_train_set_file_path)),
    augmented_ns_cnlts_val_set=LazyEval(lambda: jsonl_load(_augmented_ns_cnlts_val_set_file_path)),
    encoded_ns_cnlts_train_set=LazyEval(lambda: jsonl_load(_encoded_ns_cnlts_train_set_file_path)),
    encoded_ns_cnlts_val_set=LazyEval(lambda: jsonl_load(_encoded_ns_cnlts_val_set_file_path)),
    shuffled_augmented_ns_cnlts_train_set=LazyEval(lambda: jsonl_load(_shuffled_augmented_ns_cnlts_train_set_file_path)),
    shuffled_encoded_ns_cnlts_train_set=LazyEval(lambda: jsonl_load(_shuffled_encoded_ns_cnlts_train_set_file_path)),

    augmented_nao_train_set=LazyEval(lambda: jsonl_load(_augmented_nao_train_set_file_path)),
    augmented_nao_val_set=LazyEval(lambda: jsonl_load(_augmented_nao_val_set_file_path)),
    encoded_nao_train_set=LazyEval(lambda: jsonl_load(_encoded_nao_train_set_file_path)),
    encoded_nao_val_set=LazyEval(lambda: jsonl_load(_encoded_nao_val_set_file_path)),
    shuffled_augmented_nao_train_set=LazyEval(lambda: jsonl_load(_shuffled_augmented_nao_train_set_file_path)),
    shuffled_encoded_nao_train_set=LazyEval(lambda: jsonl_load(_shuffled_encoded_nao_train_set_file_path)),

    # Paths
    augmented_ns_train_set_file_path=_augmented_ns_train_set_file_path,
    augmented_ns_val_set_file_path=_augmented_ns_val_set_file_path,
    encoded_ns_train_set_file_path=_encoded_ns_train_set_file_path,
    encoded_ns_val_set_file_path=_encoded_ns_val_set_file_path,
    shuffled_augmented_ns_train_set_file_path=_shuffled_augmented_ns_train_set_file_path,
    shuffled_encoded_ns_train_set_file_path=_shuffled_encoded_ns_train_set_file_path,

    augmented_cnlts_train_set_file_path=_augmented_cnlts_train_set_file_path,
    augmented_cnlts_val_set_file_path=_augmented_cnlts_val_set_file_path,
    encoded_cnlts_train_set_file_path=_encoded_cnlts_train_set_file_path,
    encoded_cnlts_val_set_file_path=_encoded_cnlts_val_set_file_path,
    shuffled_augmented_cnlts_train_set_file_path=_shuffled_augmented_cnlts_train_set_file_path,
    shuffled_encoded_cnlts_train_set_file_path=_shuffled_encoded_cnlts_train_set_file_path,

    augmented_ns_cnlts_train_set_file_path=_augmented_ns_cnlts_train_set_file_path,
    augmented_ns_cnlts_val_set_file_path=_augmented_ns_cnlts_val_set_file_path,
    encoded_ns_cnlts_train_set_file_path=_encoded_ns_cnlts_train_set_file_path,
    encoded_ns_cnlts_val_set_file_path=_encoded_ns_cnlts_val_set_file_path,
    shuffled_augmented_ns_cnlts_train_set_file_path=_shuffled_augmented_ns_cnlts_train_set_file_path,
    shuffled_encoded_ns_cnlts_train_set_file_path=_shuffled_encoded_ns_cnlts_train_set_file_path,

    augmented_nao_train_set_file_path=_augmented_nao_train_set_file_path,
    augmented_nao_val_set_file_path=_augmented_nao_val_set_file_path,
    encoded_nao_train_set_file_path=_encoded_nao_train_set_file_path,
    encoded_nao_val_set_file_path=_encoded_nao_val_set_file_path,
    shuffled_augmented_nao_train_set_file_path=_shuffled_augmented_nao_train_set_file_path,
    shuffled_encoded_nao_train_set_file_path=_shuffled_encoded_nao_train_set_file_path,
)
