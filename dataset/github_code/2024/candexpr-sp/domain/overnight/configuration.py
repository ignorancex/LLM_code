
import os
from functools import cache, partial

from dhnamlib.pylib.context import Environment, LazyEval
from dhnamlib.pylib.debug import NIE
from dhnamlib.pylib.filesys import jsonl_load
from dhnamlib.pylib.decoration import construct
from dhnamlib.pylib.mllib.dataproc import split_train_val, extract_portion

from splogic.base.formalism import Formalism
from splogic.base.grammar import read_grammar
# from splogic.seq2seq.validation import Validator, ResultCollector
from splogic.seq2seq.validation import NaiveDenotationEqual, Validator, DEFAULT_OPTIM_MEASURES, DEFAULT_SEARCH_MEASURES
# from splogic.seq2seq import filemng
from splogic.base.execution import ExprCompiler

import configuration
from utility.exception import UVE

from .path import get_preprocessed_dataset_file_path
from .lf_interface.transfer import OVERNIGHT_DOMAINS
from .execution import OvernightContextCreater, OvernightExecutor
from .validation import OvernightResultCollector
# from .validation import OvernightDenotationEqual
# from .filemng import save_analysis, save_extra_performance
from .filemng import OvernightFileManager
from .dynamic_bind import DomainDynamicBinder
from splogic.seq2seq.data_read import make_data_loader


_pretrained_model_name_or_path = 'facebook/bart-base'

_raw_dataset_dir_path = './dataset/overnight'
_augmented_dataset_dir_path = './preprocessed/overnight/augmented'
_encoded_dataset_dir_path = './preprocessed/overnight/encoded'
_shuffled_encoded_dataset_dir_path = './preprocessed/overnight/shuffled_encoded'
_augmented_strict_dataset_dir_path = './preprocessed/overnight/augmented_strict'
_encoded_strict_dataset_dir_path = './preprocessed/overnight/encoded_strict'
_shuffled_encoded_strict_dataset_dir_path = './preprocessed/overnight/shuffled_encoded_strict'

_grammar_file_path = './domain/overnight/grammar.lissp'

_NO_CONTEXT = object()

NUM_ALL_DOMAIN_TRAIN_EXAMPLES = 8751
# `NUM_ALL_DOMAIN_TRAIN_EXAMPLES` is the number of training examples of all domains.
# The examples used for validation are not counted.


def _make_grammar(**kwargs):
    from .grammar import OvernightGrammar
    return read_grammar(
        _grammar_file_path,
        formalism=Formalism(decoding_speed_optimization=configuration.config.decoding_speed_optimization),
        grammar_cls=OvernightGrammar,
        grammar_kwargs=dict(
            pretrained_model_name_or_path=configuration.config.pretrained_model_name_or_path,
            **kwargs
        ))


_TRAIN_SET_RATIO = 0.8
_WEAKSUP_PRETRAINING_SET_PERCENT = 1


def _iter_domain_train_val_sets():
    dataset_split = 'train'
    for domain in configuration.config.train_domains:
        assert 'shuffled' in configuration.config.shuffled_encoded_dataset_dir_path
        dataset = jsonl_load(get_preprocessed_dataset_file_path(
            configuration.config.shuffled_encoded_dataset_dir_path, domain, dataset_split))
        train_set, val_set = split_train_val(dataset, ratio=_TRAIN_SET_RATIO, round_fn=int)
        yield domain, (train_set, val_set)


def _load_train_set(portion_percent=100, num_examples=None, weaksup_search=False):
    """
    >>> len(_load_train_set(portion_percent=100)) == NUM_ALL_DOMAIN_TRAIN_EXAMPLES  # doctest: +SKIP
    """

    merged_train_set = []
    for domain, (train_set, _) in _iter_domain_train_val_sets():
        _augment_dataset_with_domains(train_set, domain)
        if weaksup_search:
            _augment_dataset_with_answers(train_set, domain)
            for example in train_set:
                del example['logical_form']
                for key in ['action_ids', 'action_id_tree']:
                    if key in example:
                        del example[key]

        if portion_percent is None:
            assert num_examples is not None
            assert num_examples <= len(train_set)
            _train_set = train_set[:num_examples]
        else:
            assert portion_percent is not None
            assert 0 < portion_percent <= 100
            _train_set = extract_portion(train_set, percent=portion_percent, round_fn=int) \
                if portion_percent < 100 else train_set

        merged_train_set.extend(_train_set)

    return merged_train_set


def _load_val_set():
    """
    >>> len(_load_val_set()) == 2191  # doctest: +SKIP
    """

    merged_val_set = []
    for domain, (_, val_set) in _iter_domain_train_val_sets():
        _augment_dataset_with_domains(val_set, domain)
        _augment_dataset_with_answers(val_set, domain)
        merged_val_set.extend(val_set)
    return merged_val_set


def _load_test_set():
    """
    >>> len(_load_test_set()) == 2740  # doctest: +SKIP
    """

    merged_test_set = []
    dataset_split = 'test'
    for domain in configuration.config.test_domains:
        dataset = jsonl_load(get_preprocessed_dataset_file_path(
            configuration.config.encoded_dataset_dir_path, domain, dataset_split))
        _augment_dataset_with_domains(dataset, domain)
        _augment_dataset_with_answers(dataset, domain)
        merged_test_set.extend(dataset)

    return merged_test_set


def _augment_dataset_with_domains(dataset, domain):
    for example in dataset:
        assert 'domain' not in example
        example['domain'] = domain


# @construct(list)
def _augment_dataset_with_answers(dataset, domain):
    logical_forms = [example['logical_form'] for example in dataset]
    executor = OvernightExecutor()
    contexts = (dict(domain=domain),) * len(logical_forms)
    exec_result = executor.execute(logical_forms, contexts)
    answers = exec_result.get()

    for example, answer in zip(dataset, answers):
        # augmented_example = dict(example)
        # augmented_example['answer'] = answer
        # yield augmented_example
        example['answer'] = answer


def _make_validator():
    return Validator(
        compiler=ExprCompiler(),
        context_creator=OvernightContextCreater(),
        executor=OvernightExecutor(),
        dynamic_binder=DomainDynamicBinder(),
        denotation_equal=NaiveDenotationEqual(),
        result_collector_cls=OvernightResultCollector,
        extra_analysis_keys=['domain'],
        evaluating_in_progress=False,
    )


config = Environment(
    optim_measures=DEFAULT_OPTIM_MEASURES,
    search_measures=DEFAULT_SEARCH_MEASURES,

    using_arg_candidate=True,
    using_arg_filter=False,
    constrained_decoding=True,
    inferencing_subtypes=True,
    using_distinctive_union_types=True,
    pretrained_model_name_or_path=_pretrained_model_name_or_path,
    # context=_NO_CONTEXT,
    grammar=LazyEval(_make_grammar),
    # compiler=LazyEval(NIE),
    test_validator=LazyEval(_make_validator),
    search_validator=LazyEval(_make_validator),
    make_data_loader_fn=partial(make_data_loader, extra_keys=['domain']),
    # save_analysis_fn=save_analysis,
    # save_extra_performance_fn=save_extra_performance,
    file_manager=OvernightFileManager(),
    evaluating_test_set=True,

    # generation_max_length=500,
    generation_max_length=200,
    num_prediction_beams=1,
    # num_prediction_beams=4,
    softmax_masking=False,

    all_domains=OVERNIGHT_DOMAINS,
    # train_domains=LazyEval(UVE),
    # test_domains=LazyEval(UVE),
    # train_domains=OVERNIGHT_DOMAINS,
    # test_domains=OVERNIGHT_DOMAINS,

    raw_dataset_dir_path=_raw_dataset_dir_path,
    augmented_dataset_dir_path=_augmented_dataset_dir_path,
    encoded_dataset_dir_path=_encoded_dataset_dir_path,
    shuffled_encoded_dataset_dir_path=_shuffled_encoded_dataset_dir_path,
    augmented_strict_dataset_dir_path=_augmented_strict_dataset_dir_path,
    encoded_strict_dataset_dir_path=_encoded_strict_dataset_dir_path,
    shuffled_encoded_strict_dataset_dir_path=_shuffled_encoded_strict_dataset_dir_path,

    train_set_portion_percent=100,
    num_used_train_examples=None,
    encoded_train_set=LazyEval(lambda: _load_train_set(
        portion_percent=configuration.config.train_set_portion_percent,
        num_examples=configuration.config.num_used_train_examples,
    )),
    encoded_val_set=LazyEval(_load_val_set),
    encoded_test_set=LazyEval(_load_test_set),

    encoded_weaksup_pretraining_set=LazyEval(lambda: _load_train_set(
        portion_percent=_WEAKSUP_PRETRAINING_SET_PERCENT,
        # num_examples=configuration.config.num_used_train_examples,
    )),
    encoded_weaksup_search_set=LazyEval(lambda: _load_train_set(weaksup_search=True)),

    # context_creator=OvernightContextCreater(),
    # denotation_equal=OvernightDenotationEqual(),
    # result_collector_cls=OvernightResultCollector,
)
