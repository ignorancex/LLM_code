
from functools import cache
from itertools import chain
from tqdm import tqdm

from dhnamlib.pylib.iteration import rcopy, rmemberif
from dhnamlib.pylib.statistics import shuffled
from dhnamlib.pylib.mllib.dataproc import extract_portion, split_train_val

from .kopl_interface import transfer as kopl_transfer


_SEED = 42


@cache
def _get_act_type_to_id_dict(grammar):
    act_type_to_id_dict = {}
    for action in grammar.base_actions:
        if action.arg_candidate is not None:
            act_type_to_id_dict[action.act_type] = action.id

    return act_type_to_id_dict


def sample_arg_candidate_id_seq_set(grammar, global_context, ratio, seed=_SEED):
    assert 0 < ratio < 1

    act_type_to_id_dict = _get_act_type_to_id_dict(grammar)
    candidate_id_seq_set = set()

    for act_type, trie in kopl_transfer.iter_act_type_trie_pairs(
            lf_tokenizer=grammar.lf_tokenizer, end_of_seq=grammar.reduce_token, context=global_context
    ):
        act_id = act_type_to_id_dict[act_type]
        candidate_id_seqs = shuffled(
            sorted(((act_id,) + nlt_id_seq) for nlt_id_seq in trie.id_seqs()),
            seed=seed)
        candidate_id_seq_set.update(extract_portion(candidate_id_seqs, ratio=ratio))

    return candidate_id_seq_set


def resplit_dataset(grammar, global_context, test_arg_candidate_ratio, encoded_datasets, seed=_SEED):
    merged_encoded_dataset = list(chain(*encoded_datasets))

    candidate_id_seq_set = sample_arg_candidate_id_seq_set(
        grammar, global_context, test_arg_candidate_ratio, seed=seed)

    train_val_set = []
    test_set = []

    # print('# all examples: ', len(merged_encoded_dataset))
    for example in tqdm(merged_encoded_dataset):
        action_id_tree = rcopy(example['action_id_tree'], coll_fn=tuple)

        subtree = rmemberif(
            candidate_id_seq_set.__contains__,
            action_id_tree,
            default=None)

        if subtree is None:
            train_val_set.append(example)
        else:
            test_set.append(example)

    train_set, val_set = split_train_val(
        shuffled(train_val_set, seed=seed),
        val_set_size=len(test_set))

    return train_set, val_set, test_set
