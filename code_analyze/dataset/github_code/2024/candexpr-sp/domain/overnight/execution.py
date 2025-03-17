
from functools import cache
from itertools import chain
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

from dhnamlib.pylib.klass import override, subclass
from dhnamlib.pylib.iteration import split_by_lengths
# from dhnamlib.pylib.decoration import construct
from dhnamlib.pylib.decoration import deprecated
from dhnamlib.pylib.constant import NO_VALUE
from dhnamlib.pylib.context import context_nest

from splogic.base.execution import LazyExecResult, LazyExecutor, ContextCreater, INVALID_PROGRAM, NO_DENOTATION

from overnight.domain_overnight import OvernightDomain, set_evaluator_path


set_evaluator_path('./overnight/evaluator')


@subclass
class OvernightExecResult(LazyExecResult):
    def __init__(self, lazy_executor):
        super().__init__(lazy_executor)
        self.domains = NO_VALUE

    def _set_domains(self, domains):
        self.domains = domains


@subclass
class OvernightExecutor(LazyExecutor):
    def __init__(self, result_cls=OvernightExecResult):
        super().__init__(result_cls=result_cls)

    @cache
    def get_evaluator(self, domain):
        return OvernightDomain(domain)

    @override
    def _process(self, postponed_batch_groups):
        lazy_results, programs_tuple, contexts_tuple = zip(*postponed_batch_groups)

        batch_sizes = tuple(map(len, programs_tuple))

        all_programs = tuple(chain(*programs_tuple))
        all_contexts = tuple(chain(*contexts_tuple))
        all_denotations = self._get_all_denotations(all_programs, all_contexts)
        denotations_tuple = split_by_lengths(all_denotations, batch_sizes)

        for lazy_result, denotations, contexts in zip(lazy_results, denotations_tuple, contexts_tuple):
            lazy_result._set_values(denotations)
            lazy_result._set_domains(tuple(context['domain'] for context in contexts))

    def _get_all_denotations(self, all_programs, all_contexts):
        # breakpoint()
        assert len(all_programs) == len(all_contexts)

        program_group_dict = {}
        for idx, (program, context) in enumerate(zip(all_programs, all_contexts)):
            program_group_dict.setdefault(context['domain'], []).append((idx, program))

        denotation_group_dict = {}
        for domain, idx_program_pairs in program_group_dict.items():
            indices, programs = zip(*idx_program_pairs)
            denotations = self._execute_programs(domain, programs)
            # normalized_prgrams, invalid_program_indices = self._normalize_programs(programs)
            # denotations = self.get_evaluator(domain).execute_logical_forms(normalized_prgrams)
            # for invalid_program_index in invalid_program_indices:
            #     denotations[invalid_program_index] = NO_DENOTATION
            denotation_group_dict[domain] = tuple(zip(indices, denotations))

        all_denotations = [None] * len(all_programs)
        for idx, denotation in chain(*denotation_group_dict.values()):
            all_denotations[idx] = denotation

        return all_denotations

    def _execute_programs(self, domain, programs):
        normalized_prgrams, invalid_program_indices = self._normalize_programs(programs)
        denotations = self.get_evaluator(domain).execute_logical_forms(normalized_prgrams)
        for invalid_program_index in invalid_program_indices:
            denotations[invalid_program_index] = NO_DENOTATION
        return denotations

    @deprecated
    def _execute_programs_old(self, domain, programs):
        normalized_prgrams, invalid_program_indices = self._normalize_programs(programs)
        with StringIO() as sio:
            with context_nest(redirect_stdout(sio), redirect_stderr(sio)) as (rstdout, rstderr):
                denotations = self.get_evaluator(domain).execute_logical_forms(normalized_prgrams)
                for invalid_program_index in invalid_program_indices:
                    denotations[invalid_program_index] = NO_DENOTATION
            error_msg = sio.getvalue()
        error_lines = error_msg.split('\n')
        num_errors = sum(int(error_line.startswith('java.lang.')) for error_line in error_lines)
        if num_errors > 0:
            print(f'The number of execution errors = {num_errors}.')

        return denotations

    def _normalize_programs(self, programs):
        normalized_prgrams = []
        invalid_program_indices = []
        for idx, program in enumerate(programs):
            if program is INVALID_PROGRAM:
                normalized_prgrams.append(INVALID_LOGICAL_FORM)
                invalid_program_indices.append(idx)
            else:
                normalized_prgrams.append(program)
        return normalized_prgrams, invalid_program_indices


INVALID_LOGICAL_FORM = '( call SW.listValue ( call SW.singleton en ) )'


@subclass
class OvernightContextCreater(ContextCreater):
    def __call__(self, batch):
        return tuple(dict(domain=domain) for domain in batch['domain'])
