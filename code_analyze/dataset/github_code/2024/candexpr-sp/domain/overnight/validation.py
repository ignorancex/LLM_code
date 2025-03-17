from fractions import Fraction

from accelerate.utils.operations import gather_object

from dhnamlib.pylib.klass import subclass, override
from dhnamlib.pylib.mllib.learning import get_performance
from dhnamlib.pylib.iteration import partition, all_same

from splogic.seq2seq.validation import ResultCollector, compute_correctness, DenotationEqual

from .lf_interface.transfer import OVERNIGHT_DOMAINS


@subclass
class OvernightResultCollector(ResultCollector):
    def __init__(self, evaluating, denotation_equal: DenotationEqual, num_return_sequences):
        super().__init__(evaluating, denotation_equal, num_return_sequences)
        self.num_domain_correct_dict = {}
        self.num_domain_examples_dict = {}

    @override
    def _update_from_batch(self, batch):
        predictions = batch['exec_result'].get()
        self.predictions.extend(predictions)

        if self.evaluating:
            assert 'answer' in batch
            answers = batch['answer']
            self.answers.extend(answers)

            correctness_values = compute_correctness(
                predictions, answers, self.denotation_equal,
                num_return_sequences=self.num_return_sequences)

            self.num_correct += sum(map(int, correctness_values))

            domain_groups = tuple(partition(batch['exec_result'].domains, self.num_return_sequences))
            for domain_group in domain_groups:
                assert all_same(domain_group)
            representative_domains = tuple(domain_group[0] for domain_group in domain_groups)

            for correctness_value, domain in zip(correctness_values, representative_domains):
                self.num_domain_correct_dict[domain] = self.num_domain_correct_dict.get(domain, 0) + int(correctness_value)
                self.num_domain_examples_dict[domain] = self.num_domain_examples_dict.get(domain, 0) + 1

    @override
    def get_overall_performance(self, measure_names, with_extra=False):
        if with_extra:
            return self.get_overall_performance_with_extra(measure_names)
        else:
            return super().get_overall_performance(measure_names, with_extra=with_extra)

    def get_overall_performance_with_extra(self, measure_names):
        measure_kv_list = []
        domain_measure_kv_list_dict = {}
        measure_cnt = 0

        overall_num_correct = sum(gather_object([self.num_correct]))
        overall_num_answers = sum(gather_object([len(self.answers)]))

        if 'accuracy' in measure_names:
            accuracy_measure_name = 'oracle_accuracy' if self.num_return_sequences > 1 else 'accuracy'
            measure_cnt += 1
            overall_accuracy = overall_num_correct / overall_num_answers
            overall_accuracy_fraction = Fraction(overall_num_correct, overall_num_answers)
            measure_kv_list.append([accuracy_measure_name, overall_accuracy])
            measure_kv_list.append([f'{accuracy_measure_name}_fraction', overall_accuracy_fraction])
            # measure_kv_list.append(['num_correct', overall_num_correct])
            # measure_kv_list.append(['num_examples', overall_num_answers])

            # DEBUG_CNT = 0
            for domain in OVERNIGHT_DOMAINS:
                # DEBUG_CNT += 1
                # if DEBUG_CNT == 8:
                #     from splogic.utility.acceleration import accelerator; from remote_pdb import RemotePdb; RemotePdb('127.0.0.1', 60000 + accelerator.process_index).set_trace()

                # # from splogic.utility.acceleration import accelerator; from remote_pdb import RemotePdb; RemotePdb('127.0.0.1', 60000 + accelerator.process_index).set_trace()
                # try:            # debug
                #     overall_num_domain_correct = sum(gather_object([self.num_domain_correct_dict[domain]]))
                # except Exception as e:         # debug
                #     print('Error occured')
                #     from splogic.utility.acceleration import accelerator; from remote_pdb import RemotePdb; RemotePdb('127.0.0.1', 60000 + accelerator.process_index).set_trace()
                #     print(e)
                #     # overall_num_domain_correct = sum(gather_object([self.num_domain_correct_dict[domain]]))  # debug
                # else:
                #     # from splogic.utility.acceleration import accelerator; from remote_pdb import RemotePdb; RemotePdb('127.0.0.1', 60000 + accelerator.process_index).set_trace()
                #     print(overall_num_domain_correct)

                # ============================================================================================================
                # Note
                #
                # When multiple GPUs and processes are used,
                # `self.num_domain_correct_dict` and `self.num_domain_examples_dict` may not include all 8 domains.
                # Because the test examples are merged before spread to multiple processes,
                # each process has a different number of examples.
                # Therefore, the for loop should repeat on all domains in `OVERNIGHT_DOMAINS`,
                # and `dict.get` should be used for `self.num_domain_correct_dict` and `self.num_domain_examples_dict`
                # where the default value of `dict.get` should be 0.
                # ============================================================================================================

                overall_num_domain_examples = sum(gather_object([self.num_domain_examples_dict.get(domain, 0)]))
                if overall_num_domain_examples > 0:
                    overall_num_domain_correct = sum(gather_object([self.num_domain_correct_dict.get(domain, 0)]))

                    overall_domain_accuracy = overall_num_domain_correct / overall_num_domain_examples
                    overall_domain_accuracy_fraction = Fraction(overall_num_domain_correct, overall_num_domain_examples)

                    domain_measure_kv_list_dict.setdefault(domain, []).append(
                        [accuracy_measure_name, overall_domain_accuracy])
                    domain_measure_kv_list_dict.setdefault(domain, []).append(
                        [f'{accuracy_measure_name}_fraction', overall_domain_accuracy_fraction])
                    domain_measure_kv_list_dict.setdefault(domain, []).append(['num_correct', overall_num_correct])
                    domain_measure_kv_list_dict.setdefault(domain, []).append(['num_examples', overall_num_answers])

        assert len(measure_names) == measure_cnt

        overall_performance = get_performance(measure_kv_list)

        overall_domain_performance = dict(
            [domain, get_performance(kv_list)]
            for domain, kv_list in domain_measure_kv_list_dict.items())

        return overall_performance, overall_domain_performance
