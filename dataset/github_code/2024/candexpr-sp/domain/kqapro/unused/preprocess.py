
# import math
from argparse import ArgumentParser
from itertools import chain
from tqdm import tqdm
import torch
import random

from meta_configuration import set_default_domain_name
set_default_domain_name('kqapro')  # Call `set_default_domain_name` before the `configuration` module is loaded.
from configuration import config
from ..configuration import _make_grammar, _make_ablation_grammar

from dhnamlib.pylib.filesys import jsonl_save, pickle_save, mkpdirs_unless_exist
from dhnamlib.pylib.time import TimeMeasure
from dhnamlib.pylib.iteration import rcopy, flatten, rmap
from dhnamlib.pylib.decoration import construct, deprecated

from splogic.seq2seq import learning
from splogic.seq2seq.dynamic_bind import UtteranceSpanTrieDynamicBinder
from splogic.base.formalism import InvalidCandidateActionError

from ..execution import postprocess_prediction, KoPLCompiler
from ..kopl_interface.original import execute_kopl_program
# from .kopl_interface import transfer

from .data_resplit import resplit_dataset



@construct(list)
def encode_dataset(grammar, augmented_dataset, example_idx_as_id=False):
    for example_idx, example in enumerate(tqdm(augmented_dataset)):

        # `utterance_token_ids` and `action_ids` include BOS and EOS.
        encoded_example = dict(
            example_id=example_idx if example_idx_as_id else example['example_id'],
            utterance_token_ids=grammar.utterance_tokenizer(example['question'])['input_ids'])

        if 'answer' in example:
            encoded_example.update(answer=example['answer'])

        if 'action_name_tree' in example:
            assert 'action_name_seq' not in example
            action_name_tree = example['action_name_tree']
            # assert action_name_tree[0] is the first action's name
            action_id_tree = list(chain(
                [grammar.lf_tokenizer.bos_token_id],
                [rmap(grammar.name_to_id, action_name_tree, coll_fn=list)],
                [grammar.lf_tokenizer.eos_token_id]))
            encoded_example.update(action_id_tree=action_id_tree)
        elif 'action_name_seq' in example:
            action_name_seq = example['action_name_seq']
            action_ids = list(chain(
                [grammar.lf_tokenizer.bos_token_id],
                map(grammar.name_to_id, action_name_seq),
                [grammar.lf_tokenizer.eos_token_id]))
            encoded_example.update(action_ids=action_ids)

        if 'answer_by_program' in example:
            encoded_example.update(answer_by_program=example['answer_by_program'])

        yield encoded_example


@config
def preprocess_for_encoded_dataset(
        *,
        grammar=config.ph,
        augmented_dataset,
        encoded_dataset_file_path,
        example_idx_as_id=False):
    encoded_dataset = encode_dataset(grammar, augmented_dataset, example_idx_as_id=example_idx_as_id)
    mkpdirs_unless_exist(encoded_dataset_file_path)
    jsonl_save(encoded_dataset, encoded_dataset_file_path)
    print(f'The encoded dataset was saved as {encoded_dataset_file_path}')


@deprecated
@construct(list)
def encode_mask(grammar, encoded_dataset):
    for example in tqdm(encoded_dataset):
        labels = torch.tensor([example['action_ids'][1:]], dtype=torch.int64)
        softmax_mask, nll_mask = learning.labels_to_masks(grammar, labels)

        softmax_mask = tuple(map(bytes, softmax_mask[0].tolist()))
        nll_mask = bytes(nll_mask[0].tolist())

        yield dict(softmax_mask=softmax_mask,
                   nll_mask=nll_mask)


@deprecated
@config
def preprocess_for_encoded_mask(
        *,
        grammar=config.ph,
        encoded_dataset,
        encoded_train_mask_dataset_file_path):
    encoded_mask = encode_mask(grammar, encoded_dataset)
    mkpdirs_unless_exist(encoded_train_mask_dataset_file_path)
    pickle_save(encoded_mask, encoded_train_mask_dataset_file_path)
    print(f'The encoded mask was saved as {encoded_train_mask_dataset_file_path}')


def shuffle_dataset(dataset):
    seed = 42
    new_dataset = list(dataset)
    random.Random(seed).shuffle(new_dataset)
    return new_dataset


def preprocess_for_shuffled_dataset(
        *,
        dataset,
        shuffled_dataset_file_path):
    shuffled_dataset = shuffle_dataset(dataset)
    mkpdirs_unless_exist(shuffled_dataset_file_path)
    jsonl_save(shuffled_dataset, shuffled_dataset_file_path)
    print(f'The shuffled dataset was saved as {shuffled_dataset_file_path}')


@construct(list)
def augment_dataset_with_strict_grammar(augmented_dataset, grammar):
    assert grammar.inferencing_subtypes is False
    dynamic_binder = UtteranceSpanTrieDynamicBinder()
    for example in tqdm(augmented_dataset):
        utterance_token_id_seq = grammar.utterance_tokenizer(example['question'])['input_ids']
        dynamic_binding = dynamic_binder.bind_example(dict(utterance_token_ids=utterance_token_id_seq), grammar=grammar)
        # utterance_span_trie = learning.utterance_token_id_seq_to_span_trie(grammar, utterance_token_id_seq)
        action_tree = grammar.strict_type_processing.get_strictly_typed_action_tree(
            grammar,
            action_name_tree=example['action_name_tree'],
            dynamic_binding=dynamic_binding)
        action_name_tree = rmap(lambda action: action.name, action_tree)

        new_example = dict(example)
        new_example['action_name_tree'] = action_name_tree
        yield new_example


def preprocess_for_augmented_strict_dataset(
        *,
        augmented_dataset,
        augmented_strict_dataset_file_path):
    grammar = _make_grammar(inferencing_subtypes=False)
    # with config.let(inferencing_subtypes=False):
    #     grammar = _make_grammar()
    #     assert grammar.inferencing_subtypes is False

    augmented_strict_dataset = augment_dataset_with_strict_grammar(augmented_dataset, grammar)
    jsonl_save(augmented_strict_dataset, augmented_strict_dataset_file_path)
    print(f'The augmented dataset with strict grammar was saved as {augmented_strict_dataset_file_path}')


def preprocess_for_encoded_strict_dataset(
        *,
        augmented_dataset,
        encoded_dataset_file_path,
        example_idx_as_id=False
):
    grammar = _make_grammar(inferencing_subtypes=False)
    # with config.let(inferencing_subtypes=False):
    #     grammar = _make_grammar()
    #     assert grammar.inferencing_subtypes is False

    preprocess_for_encoded_dataset(
        grammar=grammar,
        augmented_dataset=augmented_dataset,
        encoded_dataset_file_path=encoded_dataset_file_path,
        example_idx_as_id=example_idx_as_id)


def extract_dataset_portion(dataset, percent, expected_dataset_size=None):
    # ratio = percent / 100
    num_examples = round(len(dataset) * percent / 100)
    if expected_dataset_size is not None:
        assert num_examples == expected_dataset_size
    return dataset[:num_examples]


def preprocess_for_encoded_weaksup_pretraining_set(
        *,
        full_dataset,
        weaksup_pretraining_set_file_path
):
    
    assert len(full_dataset) == 94376

    percent = 0.1
    # num_epoch_repeats = math.sqrt(100 / percent)
    num_used_train_examples = round(len(full_dataset) * percent / 100)

    pretraining_dataset = extract_dataset_portion(
        shuffle_dataset(full_dataset),
        percent=percent,
        expected_dataset_size=num_used_train_examples)

    jsonl_save(pretraining_dataset, weaksup_pretraining_set_file_path)
    print(f'The weaksup pretraining dataset was saved as {weaksup_pretraining_set_file_path}')


def preprocess_for_encoded_weaksup_search_set(
        *,
        full_dataset,
        weaksup_search_set_file_path
):
    keys = ['example_id', 'utterance_token_ids', 'answer']
    search_dataset = [dict(zip(keys, map(example.__getitem__, keys)))
                      for example in full_dataset]

    jsonl_save(search_dataset, weaksup_search_set_file_path)
    print(f'The weaksup search dataset was saved as {weaksup_search_set_file_path}')


def preprocess_for_augmented_ablation_dataset(
        *,
        non_symbolic=False,
        using_common_nl_token_seq=False,
        naive_arg_ordering=False,
        augmented_original_dataset,
        ablation_dataset_file_path,
):
    grammar = _make_ablation_grammar(
        non_symbolic=non_symbolic,
        using_common_nl_token_seq=using_common_nl_token_seq,
        naive_arg_ordering=naive_arg_ordering)

    ablation_augmented_dataset = rcopy(augmented_original_dataset)
    for example in tqdm(ablation_augmented_dataset):
        original_action_name_tree = example['action_name_tree']
        # symbolic_action_seq = tuple(grammar.name_to_action(action_name) for action_name in symbolic_action_name_seq)
        original_action_tree = rmap(grammar.name_to_action, original_action_name_tree)
        ablation_action_tree = grammar.convert_to_ablation_action_tree(original_action_tree)
        # non_symbolic_action_name_seq = tuple(action.name for action in non_symbolic_action_seq)
        ablation_action_name_tree = rmap(lambda action: action.name, ablation_action_tree)
        # example['action_name_seq'] = non_symbolic_action_name_seq
        example['action_name_tree'] = ablation_action_name_tree
    jsonl_save(ablation_augmented_dataset, ablation_dataset_file_path)


def preprocess_for_encoded_ablation_dataset(
        *,
        non_symbolic=False,
        using_common_nl_token_seq=False,
        naive_arg_ordering=False,
        augmented_dataset,
        encoded_dataset_file_path,
        example_idx_as_id=False
):
    grammar = _make_ablation_grammar(
        non_symbolic=non_symbolic,
        using_common_nl_token_seq=using_common_nl_token_seq,
        naive_arg_ordering=naive_arg_ordering)

    preprocess_for_encoded_dataset(
        grammar=grammar,
        augmented_dataset=augmented_dataset,
        encoded_dataset_file_path=encoded_dataset_file_path,
        example_idx_as_id=example_idx_as_id)


@config
def preprocess_for_encoded_resplit_datasets(
        *,
        encoded_train_set,
        encoded_val_set,
        test_arg_candidate_ratio,
        grammar=config.ph,
        global_context=config.ph,
        encoded_train_set_file_path,
        encoded_val_set_file_path,
        encoded_test_set_file_path,
):
    # When test_arg_candidate_ratio=0.04,
    #
    # Datasets are resplit:
    # - # of train examples: 80557
    # - # of val examples: 12808
    # - # of test examples: 12808

    @construct(list)
    def update_dataset(dataset, split):
        for example in dataset:
            new_example = dict(example)
            new_example['example_id'] = (split, example['example_id'])
            yield new_example

    _encoded_train_set = update_dataset(encoded_train_set, 'train')
    _encoded_val_set = update_dataset(encoded_val_set, 'val')

    encoded_train_set, encoded_val_set, encoded_test_set = resplit_dataset(
        grammar=grammar,
        global_context=global_context,
        test_arg_candidate_ratio=test_arg_candidate_ratio,
        encoded_datasets=[_encoded_train_set, _encoded_val_set],
        seed=42
    )
    print('Datasets are resplit.')
    print('# of train examples: {}'.format(len(encoded_train_set)))
    print('# of val examples: {}'.format(len(encoded_val_set)))
    print('# of test examples: {}'.format(len(encoded_test_set)))

    jsonl_save(encoded_train_set, encoded_train_set_file_path)
    jsonl_save(encoded_val_set, encoded_val_set_file_path)
    jsonl_save(encoded_test_set, encoded_test_set_file_path)



def _main():
    parser = ArgumentParser(description='Preprocess KoPL dataset',)
    parser.add_argument(
        '--goal',
        choices=[
            'augmented_ns_train_set',
            'augmented_ns_val_set',
            'encoded_ns_train_set',
            'encoded_ns_val_set',
            'shuffled_augmented_ns_train_set',
            'shuffled_encoded_ns_train_set',

            'augmented_cnlts_train_set',
            'augmented_cnlts_val_set',
            'encoded_cnlts_train_set',
            'encoded_cnlts_val_set',
            'shuffled_augmented_cnlts_train_set',
            'shuffled_encoded_cnlts_train_set',

            'augmented_ns_cnlts_train_set',
            'augmented_ns_cnlts_val_set',
            'encoded_ns_cnlts_train_set',
            'encoded_ns_cnlts_val_set',
            'shuffled_augmented_ns_cnlts_train_set',
            'shuffled_encoded_ns_cnlts_train_set',

            'augmented_nao_train_set',
            'augmented_nao_val_set',
            'encoded_nao_train_set',
            'encoded_nao_val_set',
            'shuffled_augmented_nao_train_set',
            'shuffled_encoded_nao_train_set',

            'encoded_resplit_datasets',
        ])

    args = parser.parse_args(config.remaining_cmd_args)

    if args.goal == 'augmented_ns_train_set':
        preprocess_for_augmented_ablation_dataset(
            non_symbolic=True,
            augmented_original_dataset=config.augmented_train_set,
            ablation_dataset_file_path=config.augmented_ns_train_set_file_path)
    elif args.goal == 'augmented_ns_val_set':
        preprocess_for_augmented_ablation_dataset(
            non_symbolic=True,
            augmented_original_dataset=config.augmented_val_set,
            ablation_dataset_file_path=config.augmented_ns_val_set_file_path)
    elif args.goal == 'encoded_ns_train_set':
        preprocess_for_encoded_ablation_dataset(
            non_symbolic=True,
            augmented_dataset=config.augmented_ns_train_set,
            encoded_dataset_file_path=config.encoded_ns_train_set_file_path)
    elif args.goal == 'encoded_ns_val_set':
        preprocess_for_encoded_ablation_dataset(
            non_symbolic=True,
            augmented_dataset=config.augmented_ns_val_set,
            encoded_dataset_file_path=config.encoded_ns_val_set_file_path)
    elif args.goal == 'shuffled_augmented_ns_train_set':
        preprocess_for_shuffled_dataset(
            dataset=config.augmented_ns_train_set,
            shuffled_dataset_file_path=config.shuffled_augmented_ns_train_set_file_path)
    elif args.goal == 'shuffled_encoded_ns_train_set':
        preprocess_for_shuffled_dataset(
            dataset=config.encoded_ns_train_set,
            shuffled_dataset_file_path=config.shuffled_encoded_ns_train_set_file_path)

    elif args.goal == 'augmented_cnlts_train_set':
        preprocess_for_augmented_ablation_dataset(
            using_common_nl_token_seq=True,
            augmented_original_dataset=config.augmented_train_set,
            ablation_dataset_file_path=config.augmented_cnlts_train_set_file_path)
    elif args.goal == 'augmented_cnlts_val_set':
        preprocess_for_augmented_ablation_dataset(
            using_common_nl_token_seq=True,
            augmented_original_dataset=config.augmented_val_set,
            ablation_dataset_file_path=config.augmented_cnlts_val_set_file_path)
    elif args.goal == 'encoded_cnlts_train_set':
        preprocess_for_encoded_ablation_dataset(
            using_common_nl_token_seq=True,
            augmented_dataset=config.augmented_cnlts_train_set,
            encoded_dataset_file_path=config.encoded_cnlts_train_set_file_path)
    elif args.goal == 'encoded_cnlts_val_set':
        preprocess_for_encoded_ablation_dataset(
            using_common_nl_token_seq=True,
            augmented_dataset=config.augmented_cnlts_val_set,
            encoded_dataset_file_path=config.encoded_cnlts_val_set_file_path)
    elif args.goal == 'shuffled_augmented_cnlts_train_set':
        preprocess_for_shuffled_dataset(
            dataset=config.augmented_cnlts_train_set,
            shuffled_dataset_file_path=config.shuffled_augmented_cnlts_train_set_file_path)
    elif args.goal == 'shuffled_encoded_cnlts_train_set':
        preprocess_for_shuffled_dataset(
            dataset=config.encoded_cnlts_train_set,
            shuffled_dataset_file_path=config.shuffled_encoded_cnlts_train_set_file_path)

    elif args.goal == 'augmented_ns_cnlts_train_set':
        preprocess_for_augmented_ablation_dataset(
            non_symbolic=True,
            using_common_nl_token_seq=True,
            augmented_original_dataset=config.augmented_train_set,
            ablation_dataset_file_path=config.augmented_ns_cnlts_train_set_file_path)
    elif args.goal == 'augmented_ns_cnlts_val_set':
        preprocess_for_augmented_ablation_dataset(
            non_symbolic=True,
            using_common_nl_token_seq=True,
            augmented_original_dataset=config.augmented_val_set,
            ablation_dataset_file_path=config.augmented_ns_cnlts_val_set_file_path)
    elif args.goal == 'encoded_ns_cnlts_train_set':
        preprocess_for_encoded_ablation_dataset(
            non_symbolic=True,
            using_common_nl_token_seq=True,
            augmented_dataset=config.augmented_ns_cnlts_train_set,
            encoded_dataset_file_path=config.encoded_ns_cnlts_train_set_file_path)
    elif args.goal == 'encoded_ns_cnlts_val_set':
        preprocess_for_encoded_ablation_dataset(
            non_symbolic=True,
            using_common_nl_token_seq=True,
            augmented_dataset=config.augmented_ns_cnlts_val_set,
            encoded_dataset_file_path=config.encoded_ns_cnlts_val_set_file_path)
    elif args.goal == 'shuffled_augmented_ns_cnlts_train_set':
        preprocess_for_shuffled_dataset(
            dataset=config.augmented_ns_cnlts_train_set,
            shuffled_dataset_file_path=config.shuffled_augmented_ns_cnlts_train_set_file_path)
    elif args.goal == 'shuffled_encoded_ns_cnlts_train_set':
        preprocess_for_shuffled_dataset(
            dataset=config.encoded_ns_cnlts_train_set,
            shuffled_dataset_file_path=config.shuffled_encoded_ns_cnlts_train_set_file_path)

    elif args.goal == 'augmented_nao_train_set':
        preprocess_for_augmented_ablation_dataset(
            naive_arg_ordering=True,
            augmented_original_dataset=config.augmented_train_set,
            ablation_dataset_file_path=config.augmented_nao_train_set_file_path)
    elif args.goal == 'augmented_nao_val_set':
        preprocess_for_augmented_ablation_dataset(
            naive_arg_ordering=True,
            augmented_original_dataset=config.augmented_val_set,
            ablation_dataset_file_path=config.augmented_nao_val_set_file_path)
    elif args.goal == 'encoded_nao_train_set':
        preprocess_for_encoded_ablation_dataset(
            naive_arg_ordering=True,
            augmented_dataset=config.augmented_nao_train_set,
            encoded_dataset_file_path=config.encoded_nao_train_set_file_path)
    elif args.goal == 'encoded_nao_val_set':
        preprocess_for_encoded_ablation_dataset(
            naive_arg_ordering=True,
            augmented_dataset=config.augmented_nao_val_set,
            encoded_dataset_file_path=config.encoded_nao_val_set_file_path)
    elif args.goal == 'shuffled_augmented_nao_train_set':
        preprocess_for_shuffled_dataset(
            dataset=config.augmented_nao_train_set,
            shuffled_dataset_file_path=config.shuffled_augmented_nao_train_set_file_path)
    elif args.goal == 'shuffled_encoded_nao_train_set':
        preprocess_for_shuffled_dataset(
            dataset=config.encoded_nao_train_set,
            shuffled_dataset_file_path=config.shuffled_encoded_nao_train_set_file_path)
    elif args.goal == 'encoded_resplit_datasets':
        preprocess_for_encoded_resplit_datasets(
            encoded_train_set=config.encoded_train_set,
            encoded_val_set=config.encoded_val_set,
            test_arg_candidate_ratio=0.04,
            encoded_train_set_file_path=config.encoded_resplit_train_set_file_path,
            encoded_val_set_file_path=config.encoded_resplit_val_set_file_path,
            encoded_test_set_file_path=config.encoded_resplit_test_set_file_path
        )
    else:
        raise Exception('Unexpected goal')


if __name__ == '__main__':
    _main()
