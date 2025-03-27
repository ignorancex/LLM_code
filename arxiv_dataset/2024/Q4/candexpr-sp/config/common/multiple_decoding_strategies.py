
from dhnamlib.pylib.structure import AttrDict
from dhnamlib.pylib.context import LazyEval


def get_decoding_strategy_configs(make_grammar_fn):
    def _make_grammar_with_no_dut():
        # import configuration
        # with configuration.config.let(using_distinctive_union_types=False):
        #     return make_grammar_fn()
        return make_grammar_fn(using_distinctive_union_types=False)

    decoding_strategy_configs = [
        AttrDict(
            decoding_strategy_name='full-constraints',
            constrained_decoding=True,
            using_arg_candidate=True,
            using_distinctive_union_types=True,
        ),
        AttrDict(
            decoding_strategy_name='no-arg-candidate',
            constrained_decoding=True,
            using_arg_candidate=False,
            using_distinctive_union_types=True,
        ),
        AttrDict(
            decoding_strategy_name='no-ac-no-dut',
            constrained_decoding=True,
            using_arg_candidate=False,
            using_distinctive_union_types=False,
            # grammar_lazy_obj=LazyEval(kqapro_configuration._make_grammar),
            grammar_lazy_obj=LazyEval(_make_grammar_with_no_dut),
        ),
        AttrDict(
            decoding_strategy_name='no-constrained-decoding',
            constrained_decoding=False,
            using_arg_candidate=False,
            using_distinctive_union_types=False,
            # grammar_lazy_obj=LazyEval(kqapro_configuration._make_grammar),
            grammar_lazy_obj=LazyEval(_make_grammar_with_no_dut),
        ),
    ]
    return decoding_strategy_configs
