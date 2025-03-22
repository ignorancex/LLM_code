
import re
import functools
import types

from splogic.base.execution import LispCompiler, NO_DENOTATION, InstantExecResult
from splogic.base.function import make_call_limited

# from kopl.kopl import KoPLEngine

# from dhnamlib.pylib.klass import Interface
from dhnamlib.pylib.klass import subclass, override
from dhnamlib.pylib.decoration import construct, deprecated

from .kopl_interface.context import KoPLContext


@subclass
class KoPLCompiler(LispCompiler):
    # interface = Interface(LispCompiler)

    def __init__(self):
        super().__init__(
            bindings=[['postprocess-denotation', postprocess_denotation]]
        )


def postprocess_denotation(denotation):
    if isinstance(denotation, list):
        new_denotation = [str(_) for _ in denotation]
    else:
        assert isinstance(denotation, (str, int, float))
        new_denotation = str(denotation)
    return new_denotation


@subclass
class KQAProExecResult(InstantExecResult):
    # interface = Interface(InstantExecResult)

    strict = False

    def __init__(self, values):
        super().__init__(values)

    @functools.cache
    @override
    def get(self):
        return tuple(postprocess_prediction(value, strict=self.strict)
                     for value in super().get())


class KQAProStricExecResult(KQAProExecResult):
    strict = True


def postprocess_prediction(prediction, strict=False):
    '''
    Modify the result from `postprocess_denotation`
    '''
    # in KQAPro_Baselines, 'None' is used as the default value
    default_prediction = 'None' if strict else 'no'

    if prediction is NO_DENOTATION:
        new_prediction = default_prediction
    elif isinstance(prediction, list):
        if len(prediction) > 1:
            # sorting for consistent prediction
            new_prediction = sorted(prediction)[0]
        elif len(prediction) > 0:
            new_prediction = prediction[0]
        elif len(prediction) == 0:
            new_prediction = default_prediction
    else:
        new_prediction = prediction

    return new_prediction


# # MAX_NUM_LOOPS = 1000000
# MAX_NUM_LOOPS = 1000


class OverMaxNumIterationsError(Exception):
    pass


class KQAProContext(KoPLContext):
    def __init__(self, kb):
        super().__init__(kb)
        self.raw_kb = kb


class KQAProCountingContext(KQAProContext):
    def __init__(self, context, max_num_iterations=1000000):
        self.kb = context.kb
        self.raw_kb = context.raw_kb
        self._count = 0
        self._max_num_iterations = max_num_iterations

    def _iterate(self, iterator):
        for item in iterator:
            self._count += 1
            yield item
            if self._count >= self._max_num_iterations:
                raise OverMaxNumIterationsError


get_counting_context = KQAProCountingContext


_kopl_function_regex = re.compile(r"[A-Z]([a-zA-Z]*)")


@deprecated
@construct(types.SimpleNamespace, from_kwargs=True)
def get_call_limited_context(context: KQAProContext, max_num_calls=100):
    call_limited = make_call_limited(max_num_calls)
    for attribute in dir(context):
        if _kopl_function_regex.match(attribute) is not None:
            obj = getattr(context, attribute)
            if callable(obj):
                yield attribute, call_limited(obj)


def _initialize_debugging_context(context: KQAProContext):
    def _debug_decorator(func):
        @functools.wraps(func)
        def new_func(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                breakpoint()
                result = func(*args, **kwargs)
                # print(func.__qualname__)
                # print(e.args)
            print('Error occured')

        return new_func

    _kopl_function_regex = re.compile(r"[A-Z]([a-zA-Z]*)")

    for attribute in dir(context):
        if _kopl_function_regex.match(attribute) is not None:
            obj = getattr(context, attribute)
            assert callable(obj)
            setattr(context, attribute, _debug_decorator(obj))

    return context


@_initialize_debugging_context
class KQAProDebugContext(KQAProContext):
    pass
