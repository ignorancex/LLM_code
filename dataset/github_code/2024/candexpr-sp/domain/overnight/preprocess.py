
import os
from argparse import ArgumentParser
from itertools import chain
from tqdm import tqdm
# import torch
import random

from meta_configuration import set_default_domain_name
set_default_domain_name('overnight')  # Call `set_default_domain_name` before the `configuration` module is loaded.
from configuration import config

from dhnamlib.pylib.iteration import rmap, flatten
from dhnamlib.pylib.filesys import jsonl_save, jsonl_load, mkpdirs_unless_exist, pandas_tsv_load
# from dhnamlib.pylib.filesys import jsonl_save, pickle_save, mkpdirs_unless_exist, pandas_tsv_load
# from dhnamlib.pylib.time import TimeMeasure
from dhnamlib.pylib.decoration import construct

from splogic.base.execution import ExprCompiler
# from splogic.seq2seq import learning
# from splogic.seq2seq.dynamic_bind import UtteranceSpanTrieDynamicBinder
# from splogic.base.formalism import InvalidCandidateActionError

from .configuration import _make_grammar
from .dynamic_bind import DomainDynamicBinder
from .execution import OvernightExecutor
from .path import get_original_dataset_file_path, get_preprocessed_dataset_file_path

# from .lf_interface.transfer import labeled_logical_form_to_action_name_tree


@config
def extract_action_trees(domain, raw_dataset, grammar=config.ph, verifying=True, verifying_grammar=True):
    # properties of raw_dataset: utterance, logical_form, original

    if verifying_grammar:
        dynamic_binder = DomainDynamicBinder()
        compiler = ExprCompiler()
        executor = OvernightExecutor()
        dynamic_binding = dynamic_binder.bind_example(dict(domain=domain), grammar=grammar)

    original_logical_forms = []
    action_trees = []
    programs = []
    for idx, row in raw_dataset.iterrows():
        original_logical_form = row['logical_form']
        original_logical_forms.append(original_logical_form)

        action_tree = grammar.token_processing.labeled_logical_form_to_action_tree(
            original_logical_form, grammar=grammar, domain=domain)
        action_trees.append(action_tree)

        action_seq = flatten(action_tree)[1:]  # except the starting action "program"
        with grammar.dynamic_scope.let(**dynamic_binding):
            # try:
            #     last_state = grammar.search_state_cls.get_last_state(action_seq, verifying=verifying)
            # except InvalidCandidateActionError as err:
            #     # breakpoint()
            #     print(err)
            #     program = '( string no_program  )'
            #     # with config.let(using_arg_candidate=False):
            #     #     last_state = grammar.search_state_cls.get_last_state(action_seq, verifying=verifying)
            # else:
            #     program = compiler.compile_tree(last_state.tree)
            last_state = grammar.search_state_cls.get_last_state(action_seq, verifying=verifying)
            program = compiler.compile_tree(last_state.tree)
            programs.append(program)

    if verifying_grammar:
        contexts = (dict(domain=domain),) * len(programs)
        exec_result = executor.execute(programs, contexts)
        gold_exec_result = executor.execute(original_logical_forms, contexts)

        if exec_result.get() != gold_exec_result.get():
            breakpoint()
            print(len(exec_result))

    return action_trees

@construct(list)
def augment_dataset(domain, raw_dataset, adding_logical_form=False, adding_action_name_tree=False):
    if adding_action_name_tree:
        print('Extracting action sequences from a dataset')
        action_trees = extract_action_trees(domain, raw_dataset)
        assert len(raw_dataset) == len(action_trees)
    print('Augmenting the dataset')
    for idx, row in tqdm(raw_dataset.iterrows(), total=len(raw_dataset.index)):
        example = dict(
            example_id=idx,
            utterance=row['utterance'],
        )
        if adding_logical_form:
            example['logical_form'] = row['logical_form']
        if adding_action_name_tree:
            action_tree = action_trees[idx]
            action_name_tree = rmap(lambda action: action.name, action_tree)
            example['action_name_tree'] = action_name_tree
        yield example


@config
def preprocess_for_augmented_dataset(
        *,
        raw_dataset_dir_path,
        augmented_dataset_dir_path,
        adding_action_name_tree,
        # adding_answer_by_program,
        all_domains=config.ph,
        dataset_split,
):

    for domain in all_domains:
        raw_dataset_file_path = get_original_dataset_file_path(raw_dataset_dir_path, domain, dataset_split)
        augmented_dataset_file_path = get_preprocessed_dataset_file_path(augmented_dataset_dir_path, domain, dataset_split)

        raw_dataset = pandas_tsv_load(raw_dataset_file_path, containing_header=True)

        augmented_dataset = augment_dataset(
            domain=domain,
            raw_dataset=raw_dataset,
            adding_logical_form=True,
            adding_action_name_tree=adding_action_name_tree,
            # adding_answer_by_program=adding_answer_by_program,
        )
        mkpdirs_unless_exist(augmented_dataset_file_path)
        jsonl_save(augmented_dataset, augmented_dataset_file_path)
        print(f'The augmented dataset was saved as {augmented_dataset_file_path}')


@construct(list)
def encode_dataset(grammar, augmented_dataset):
    for example_idx, example in enumerate(tqdm(augmented_dataset)):

        # `utterance_token_ids` and `action_ids` include BOS and EOS.
        encoded_example = dict(
            example_id=example['example_id'],
            utterance_token_ids=grammar.utterance_tokenizer(example['utterance'])['input_ids'])

        if 'logical_form' in example:
            encoded_example.update(logical_form=example['logical_form'])

        if 'action_name_tree' in example:
            _action_name_tree = example['action_name_tree']
            assert _action_name_tree[0] == grammar.start_action.name
            action_id_tree = list(chain(
                [grammar.lf_tokenizer.bos_token_id],
                rmap(grammar.name_to_id, _action_name_tree[1:], coll_fn=list),
                [grammar.lf_tokenizer.eos_token_id]))
            encoded_example.update(action_id_tree=action_id_tree)

        yield encoded_example


@config
def preprocess_for_encoded_dataset(
        *,
        grammar=config.ph,
        augmented_dataset_dir_path,
        encoded_dataset_dir_path,
        all_domains=config.ph,
        dataset_split,
):
    for domain in all_domains:
        augmented_dataset_file_path = get_preprocessed_dataset_file_path(augmented_dataset_dir_path, domain, dataset_split)
        encoded_dataset_file_path = get_preprocessed_dataset_file_path(encoded_dataset_dir_path, domain, dataset_split)

        augmented_dataset = jsonl_load(augmented_dataset_file_path)

        encoded_dataset = encode_dataset(
            grammar=grammar,
            augmented_dataset=augmented_dataset,
        )

        mkpdirs_unless_exist(encoded_dataset_file_path)
        jsonl_save(encoded_dataset, encoded_dataset_file_path)
        print(f'The encoded dataset was saved as {encoded_dataset_file_path}')


def shuffle_dataset(dataset):
    seed = 42
    new_dataset = list(dataset)
    random.Random(seed).shuffle(new_dataset)
    return new_dataset


@config
def preprocess_for_shuffled_dataset(
        *,
        dataset_dir_path,
        shuffled_dataset_dir_path,
        all_domains=config.ph,
        dataset_split,
):
    for domain in all_domains:
        dataset_file_path = get_preprocessed_dataset_file_path(dataset_dir_path, domain, dataset_split)
        shuffled_dataset_file_path = get_preprocessed_dataset_file_path(shuffled_dataset_dir_path, domain, dataset_split)

        dataset = jsonl_load(dataset_file_path)

        shuffled_dataset = shuffle_dataset(dataset)
        mkpdirs_unless_exist(shuffled_dataset_file_path)
        jsonl_save(shuffled_dataset, shuffled_dataset_file_path)
        print(f'The shuffled dataset was saved as {shuffled_dataset_file_path}')


@construct(list)
def augment_dataset_with_strict_grammar(domain, augmented_dataset, grammar):
    assert grammar.inferencing_subtypes is False
    dynamic_binder = DomainDynamicBinder()
    dynamic_binding = dynamic_binder.bind_example(dict(domain=domain), grammar=grammar)
    for example in tqdm(augmented_dataset):
        old_action_name_tree = example['action_name_tree']
        assert old_action_name_tree[0] == grammar.start_action.name
        action_tree = grammar.strict_type_processing.get_strictly_typed_action_tree(
            grammar,
            action_name_tree=old_action_name_tree[1:],
            dynamic_binding=dynamic_binding,
            including_start_action=True)

        new_action_name_tree = rmap(lambda action: action.name, action_tree)

        new_example = dict(example)
        new_example['action_name_tree'] = new_action_name_tree
        yield new_example


@config
def preprocess_for_augmented_strict_dataset(
        *,
        augmented_dataset_dir_path,
        augmented_strict_dataset_dir_path,
        all_domains=config.ph,
        dataset_split
):
    grammar = _make_grammar(inferencing_subtypes=False)

    for domain in all_domains:
        augmented_dataset_file_path = get_preprocessed_dataset_file_path(augmented_dataset_dir_path, domain, dataset_split)
        augmented_strict_dataset_file_path = get_preprocessed_dataset_file_path(augmented_strict_dataset_dir_path, domain, dataset_split)

        augmented_dataset = jsonl_load(augmented_dataset_file_path)
        augmented_strict_dataset = augment_dataset_with_strict_grammar(domain, augmented_dataset, grammar)

        mkpdirs_unless_exist(augmented_strict_dataset_file_path)
        jsonl_save(augmented_strict_dataset, augmented_strict_dataset_file_path)
        print(f'The augmented strict dataset was saved as {augmented_strict_dataset_file_path}')


def preprocess_for_encoded_strict_dataset(
        *,
        augmented_strict_dataset_dir_path,
        encoded_strict_dataset_dir_path,
        dataset_split,
):
    grammar = _make_grammar(inferencing_subtypes=False)

    preprocess_for_encoded_dataset(
        grammar=grammar,
        augmented_dataset_dir_path=augmented_strict_dataset_dir_path,
        encoded_dataset_dir_path=encoded_strict_dataset_dir_path,
        dataset_split=dataset_split,
    )


def _main():
    parser = ArgumentParser(description='Preprocess KoPL dataset',)
    parser.add_argument(
        '--goal',
        choices=[
            'augmented_train_set',
            'augmented_test_set',
            'encoded_train_set',
            'encoded_test_set',
            'shuffled_encoded_train_set',

            'augmented_strict_train_set',
            'augmented_strict_test_set',
            'encoded_strict_train_set',
            'encoded_strict_test_set',
            'shuffled_encoded_strict_train_set',
        ])

    args = parser.parse_args(config.remaining_cmd_args)

    # raise Exception('how to split 80% (train) 20% (val.) ?')

    if args.goal == 'augmented_train_set':
        preprocess_for_augmented_dataset(
            raw_dataset_dir_path=config.raw_dataset_dir_path,
            augmented_dataset_dir_path=config.augmented_dataset_dir_path,
            adding_action_name_tree=True,
            # adding_answer_by_program=False,
            dataset_split='train'
        )
    elif args.goal == 'augmented_test_set':
        preprocess_for_augmented_dataset(
            raw_dataset_dir_path=config.raw_dataset_dir_path,
            augmented_dataset_dir_path=config.augmented_dataset_dir_path,
            adding_action_name_tree=True,
            # adding_answer_by_program=False,
            dataset_split='test',
        )
    elif args.goal == 'encoded_train_set':
        preprocess_for_encoded_dataset(
            augmented_dataset_dir_path=config.augmented_dataset_dir_path,
            encoded_dataset_dir_path=config.encoded_dataset_dir_path,
            dataset_split='train'
        )
    elif args.goal == 'encoded_test_set':
        preprocess_for_encoded_dataset(
            augmented_dataset_dir_path=config.augmented_dataset_dir_path,
            encoded_dataset_dir_path=config.encoded_dataset_dir_path,
            dataset_split='test',
        )
    elif args.goal == 'shuffled_encoded_train_set':
        preprocess_for_shuffled_dataset(
            dataset_dir_path=config.encoded_dataset_dir_path,
            shuffled_dataset_dir_path=config.shuffled_encoded_dataset_dir_path,
            dataset_split='train'
        )
    elif args.goal == 'augmented_strict_train_set':
        preprocess_for_augmented_strict_dataset(
            augmented_dataset_dir_path=config.augmented_dataset_dir_path,
            augmented_strict_dataset_dir_path=config.augmented_strict_dataset_dir_path,
            dataset_split='train'
        )
    elif args.goal == 'augmented_strict_test_set':
        preprocess_for_augmented_strict_dataset(
            augmented_dataset_dir_path=config.augmented_dataset_dir_path,
            augmented_strict_dataset_dir_path=config.augmented_strict_dataset_dir_path,
            dataset_split='test'
        )
    elif args.goal == 'encoded_strict_train_set':
        preprocess_for_encoded_strict_dataset(
            augmented_strict_dataset_dir_path=config.augmented_strict_dataset_dir_path,
            encoded_strict_dataset_dir_path=config.encoded_strict_dataset_dir_path,
            dataset_split='train'
        )
    elif args.goal == 'encoded_strict_test_set':
        preprocess_for_encoded_strict_dataset(
            augmented_strict_dataset_dir_path=config.augmented_strict_dataset_dir_path,
            encoded_strict_dataset_dir_path=config.encoded_strict_dataset_dir_path,
            dataset_split='test'
        )
    elif args.goal == 'shuffled_encoded_strict_train_set':
        preprocess_for_shuffled_dataset(
            dataset_dir_path=config.encoded_strict_dataset_dir_path,
            shuffled_dataset_dir_path=config.shuffled_encoded_strict_dataset_dir_path,
            dataset_split='train'
        )
    else:
        raise Exception('Unexpected goal')


if __name__ == '__main__':
    _main()
