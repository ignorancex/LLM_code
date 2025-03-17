
from tqdm import tqdm

from configuration import config
from domain.kqapro.execution import OverMaxNumIterationsError
from domain.kqapro.kopl_context import KoPLContext


class TestCountingKoPLContext(KoPLContext):
    def __init__(self, context, max_num_iterations=1000000):
        self.kb = context.kb
        self._count = 0
        self._max_num_iterations = max_num_iterations
        self._printing_threshold = 1000

    def _iterate(self, iterator):
        for item in iterator:
            if self._count >= self._printing_threshold:
                print(f'Current count {self._count} / Threshold: {self._printing_threshold}')
                self._printing_threshold *= 10

            self._count += 1
            yield item
            if self._count >= self._max_num_iterations:
                raise OverMaxNumIterationsError


get_counting_context = TestCountingKoPLContext


@config
def analyze_programs(grammar=config.ph, context=config.ph, augmented_train_set=config.ph):
    max_action_seq_len = 0
    max_num_iterations = 0
    compiler = grammar.compiler_cls()
    for example in tqdm(augmented_train_set):
        if len(example['action_name_seq']) > max_action_seq_len:
            max_action_seq_len = len(example['action_name_seq'])
        actions = tuple(map(grammar.name_to_action, example['action_name_seq']))
        last_state = grammar.search_state_cls.get_last_state(actions)
        program = compiler.compile_tree(last_state.tree, tolerant=False)
        counting_context = get_counting_context(context)
        counting_context._printing_threshold = 100000
        denotation = program(counting_context)
        if counting_context._count > max_num_iterations:
            max_num_iterations = counting_context._count

    print(f'max_action_seq_len: {max_action_seq_len}')
    print(f'max_num_iterations: {max_num_iterations}')


# max_action_seq_len: 149
# max_num_iterations: 177836
