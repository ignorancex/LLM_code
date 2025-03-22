
from functools import cache

from dhnamlib.pylib.context import Environment, LazyEval, suppress_stdout, suppress_stderr, contextless, context_nest
from dhnamlib.pylib.filesys import json_load, jsonl_load
# from dhnamlib.pylib.klass import subclass, implement

from splogic.base.grammar import read_grammar
from splogic.utility.acceleration import accelerator
from splogic.seq2seq import filemng
from splogic.base.formalism import Formalism
from splogic.base.execution import InstantExecutor, SingletonContextCreater
from splogic.seq2seq.dynamic_bind import UtteranceSpanTrieDynamicBinder, NoDynamicBinder
from splogic.seq2seq.validation import Validator, ResultCollector, DEFAULT_OPTIM_MEASURES, DEFAULT_SEARCH_MEASURES
from splogic.seq2seq.data_read import make_data_loader
from splogic.seq2seq.ablation_grammar import make_ablation_grammar_cls

from .execution import KoPLCompiler
from .ablation_grammar import is_symbolic_action, is_nl_token_seq_action, make_common_nl_token_seq_expr_dict

# from configuration import config as global_config
import configuration
# from configuration import coc

from .execution import (
    KQAProContext, KQAProDebugContext, KQAProCountingContext,
    KQAProExecResult, KQAProStricExecResult)
from .validation import KQAProDenotationEqual


_kb_file_path = './dataset/kqapro/kb.json'
_train_set_file_path = './dataset/kqapro/train.json'
_val_set_file_path = './dataset/kqapro/val.json'
_test_set_file_path = './dataset/kqapro/test.json'

_augmented_train_set_file_path = './preprocessed/kqapro/augmented_train.jsonl'
_augmented_val_set_file_path = './preprocessed/kqapro/augmented_val.jsonl'
_encoded_train_set_file_path = './preprocessed/kqapro/encoded_train.jsonl'
_encoded_val_set_file_path = './preprocessed/kqapro/encoded_val.jsonl'
_encoded_test_set_file_path = './preprocessed/kqapro/encoded_test.jsonl'
_shuffled_augmented_train_set_file_path = './preprocessed/kqapro/shuffled_augmented_train.jsonl'
_shuffled_encoded_train_set_file_path = './preprocessed/kqapro/shuffled_encoded_train.jsonl'
# _encoded_train_mask_dataset_file_path = './preprocessed/kqapro/encoded_train_mask.jsonl'
_augmented_strict_train_set_file_path = './preprocessed/kqapro/augmented_strict_train.jsonl'
_augmented_strict_val_set_file_path = './preprocessed/kqapro/augmented_strict_val.jsonl'
_encoded_strict_train_set_file_path = './preprocessed/kqapro/encoded_strict_train.jsonl'
_encoded_strict_val_set_file_path = './preprocessed/kqapro/encoded_strict_val.jsonl'
_shuffled_augmented_strict_train_set_file_path = './preprocessed/kqapro/shuffled_augmented_strict_train.jsonl'
_shuffled_encoded_strict_train_set_file_path = './preprocessed/kqapro/shuffled_encoded_strict_train.jsonl'
_encoded_weaksup_pretraining_set_file_path = './preprocessed/kqapro/encoded_weaksup_pretraining.jsonl'
_encoded_weaksup_search_set_file_path = './preprocessed/kqapro/encoded_weaksup_search.jsonl'

_encoded_resplit_train_set_file_path = './preprocessed/kqapro/encoded_resplit_train.jsonl'
_encoded_resplit_val_set_file_path = './preprocessed/kqapro/encoded_resplit_val.jsonl'
_encoded_resplit_test_set_file_path = './preprocessed/kqapro/encoded_resplit_test.jsonl'

# Miscellaneous file paths


# _KQAPRO_TRAN_SET_SIZE = 94376
_USING_SPANS_AS_ENTITIES = False


_grammar_file_path = './domain/kqapro/grammar.lissp'
# _pretrained_model_name_or_path = './pretrained/bart-base'
_pretrained_model_name_or_path = 'facebook/bart-base'


@cache
def _make_context():
    with (
            contextless() if (configuration.config.using_tqdm and accelerator.is_local_main_process) else
            context_nest(suppress_stdout(), suppress_stderr())
    ):
        _context_cls = KQAProDebugContext if configuration.config.debug else KQAProContext
        return _context_cls(configuration.config.kb)


def _make_grammar(grammar_cls=None, **kwargs):
    if grammar_cls is None:
        from .grammar import KQAProGrammar
        grammar_cls = KQAProGrammar

    return read_grammar(
        _grammar_file_path,
        formalism=Formalism(decoding_speed_optimization=configuration.config.decoding_speed_optimization),
        grammar_cls=grammar_cls,
        grammar_kwargs=dict(
            pretrained_model_name_or_path=config.pretrained_model_name_or_path,
            **kwargs
        ))


def _make_ablation_grammar(
        non_symbolic=False,
        using_common_nl_token_seq=False,
        naive_arg_ordering=False,
        **kwargs
):
    from .grammar import KQAProGrammar

    ablation_grammar_cls = make_ablation_grammar_cls(KQAProGrammar)

    return _make_grammar(
        grammar_cls=ablation_grammar_cls,
        non_symbolic=non_symbolic,
        is_symbolic_action=is_symbolic_action,
        using_common_nl_token_seq=using_common_nl_token_seq,
        is_nl_token_seq_action=is_nl_token_seq_action,
        common_nl_token_seq_expr_dict=make_common_nl_token_seq_expr_dict(
            lambda: configuration.config.grammar
        ),
        naive_arg_ordering=naive_arg_ordering,
        **kwargs)


def _make_validator(strict):
    return Validator(
        compiler=KoPLCompiler(),
        context_creator=SingletonContextCreater(_make_context),
        executor=InstantExecutor(
            result_cls=KQAProStricExecResult if strict else KQAProExecResult,
            context_wrapper=KQAProCountingContext),
        dynamic_binder=UtteranceSpanTrieDynamicBinder() if _USING_SPANS_AS_ENTITIES else NoDynamicBinder(),
        denotation_equal=KQAProDenotationEqual(),
        result_collector_cls=ResultCollector,
        extra_analysis_keys=[],
    )


config = Environment(
    # register=Register(strategy='conditional'),
    # domain='kqapro',

    kb=LazyEval(lambda: json_load(_kb_file_path)),
    # context=LazyEval(lambda: _context_cls(rcopy(config.kb))),
    # context=LazyEval(lambda: _context_cls(config.kb)),
    global_context=LazyEval(_make_context),
    # context_creator=SingletonContextCreater(_make_context),
    # test_executor=InstantExecutor(result_cls=KQAProExecResult, context_wrapper=KQAProCountingContext),
    # search_executor=InstantExecutor(result_cls=KQAProStricExecResult, context_wrapper=KQAProCountingContext),
    # denotation_equal=KQAProDenotationEqual(),
    # dynamic_binder=UtteranceSpanTrieDynamicBinder() if _USING_SPANS_AS_ENTITIES else NoDynamicBinder(),
    max_num_program_iterations=200000,
    optim_measures=DEFAULT_OPTIM_MEASURES,
    search_measures=DEFAULT_SEARCH_MEASURES,
    # result_collector_cls=ResultCollector,

    raw_train_set=LazyEval(lambda: json_load(_train_set_file_path)),
    raw_val_set=LazyEval(lambda: json_load(_val_set_file_path)),
    raw_test_set=LazyEval(lambda: json_load(_test_set_file_path)),

    augmented_train_set=LazyEval(lambda: jsonl_load(_augmented_train_set_file_path)),
    augmented_val_set=LazyEval(lambda: jsonl_load(_augmented_val_set_file_path)),
    encoded_train_set=LazyEval(lambda: jsonl_load(_encoded_train_set_file_path)),
    encoded_val_set=LazyEval(lambda: jsonl_load(_encoded_val_set_file_path)),
    encoded_test_set=LazyEval(lambda: jsonl_load(_encoded_test_set_file_path)),
    # encoded_train_mask_dataset=LazyEval(lambda: pickle_load(_encoded_train_mask_dataset_file_path)),
    shuffled_augmented_train_set=LazyEval(lambda: jsonl_load(_shuffled_augmented_train_set_file_path)),
    shuffled_encoded_train_set=LazyEval(lambda: jsonl_load(_shuffled_encoded_train_set_file_path)),
    augmented_strict_train_set=LazyEval(lambda: jsonl_load(_augmented_strict_train_set_file_path)),
    augmented_strict_val_set=LazyEval(lambda: jsonl_load(_augmented_strict_val_set_file_path)),
    encoded_strict_train_set=LazyEval(lambda: jsonl_load(_encoded_strict_train_set_file_path)),
    encoded_strict_val_set=LazyEval(lambda: jsonl_load(_encoded_strict_val_set_file_path)),
    shuffled_augmented_strict_train_set=LazyEval(lambda: jsonl_load(_shuffled_augmented_strict_train_set_file_path)),
    shuffled_encoded_strict_train_set=LazyEval(lambda: jsonl_load(_shuffled_encoded_strict_train_set_file_path)),
    encoded_weaksup_pretraining_set=LazyEval(lambda: jsonl_load(_encoded_weaksup_pretraining_set_file_path)),
    encoded_weaksup_search_set=LazyEval(lambda: jsonl_load(_encoded_weaksup_search_set_file_path)),
    encoded_resplit_train_set=LazyEval(lambda: jsonl_load(_encoded_resplit_train_set_file_path)),
    encoded_resplit_val_set=LazyEval(lambda: jsonl_load(_encoded_resplit_val_set_file_path)),
    encoded_resplit_test_set=LazyEval(lambda: jsonl_load(_encoded_resplit_test_set_file_path)),

    grammar=LazyEval(_make_grammar),
    test_validator=LazyEval(lambda: _make_validator(strict=False)),
    search_validator=LazyEval(lambda: _make_validator(strict=True)),
    # compiler=LazyEval(lambda: config.grammar.compiler_cls()),
    using_arg_candidate=True,
    using_arg_filter=False,
    make_data_loader_fn=make_data_loader,
    # save_analysis_fn=filemng.save_analysis,
    file_manager=filemng.BaseFileManager(),
    evaluating_test_set=False,

    pretrained_model_name_or_path=_pretrained_model_name_or_path,

    augmented_train_set_file_path=_augmented_train_set_file_path,
    augmented_val_set_file_path=_augmented_val_set_file_path,
    encoded_train_set_file_path=_encoded_train_set_file_path,
    encoded_val_set_file_path=_encoded_val_set_file_path,
    encoded_test_set_file_path=_encoded_test_set_file_path,
    # encoded_train_mask_dataset_file_path=_encoded_train_mask_dataset_file_path,
    shuffled_augmented_train_set_file_path=_shuffled_augmented_train_set_file_path,
    shuffled_encoded_train_set_file_path=_shuffled_encoded_train_set_file_path,
    augmented_strict_train_set_file_path=_augmented_strict_train_set_file_path,
    augmented_strict_val_set_file_path=_augmented_strict_val_set_file_path,
    encoded_strict_train_set_file_path=_encoded_strict_train_set_file_path,
    encoded_strict_val_set_file_path=_encoded_strict_val_set_file_path,
    shuffled_augmented_strict_train_set_file_path=_shuffled_augmented_strict_train_set_file_path,
    shuffled_encoded_strict_train_set_file_path=_shuffled_encoded_strict_train_set_file_path,
    encoded_weaksup_pretraining_set_file_path=_encoded_weaksup_pretraining_set_file_path,
    encoded_weaksup_search_set_file_path=_encoded_weaksup_search_set_file_path,
    encoded_resplit_train_set_file_path=_encoded_resplit_train_set_file_path,
    encoded_resplit_val_set_file_path=_encoded_resplit_val_set_file_path,
    encoded_resplit_test_set_file_path=_encoded_resplit_test_set_file_path,

    # generation_max_length=500,
    generation_max_length=200,
    num_prediction_beams=1,
    # num_prediction_beams=4,
    softmax_masking=False,
    constrained_decoding=True,
    inferencing_subtypes=True,
    using_distinctive_union_types=True,

    # # ignoring_parsing_errors=False,
    # ignoring_parsing_errors=True,
)
