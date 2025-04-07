import pytest
from datasets import load_dataset, set_caching_enabled

from src.preprocessors import aqua_rat, TaskMode


class TestAquaRatPreprocessor:

    def setup(self):
        set_caching_enabled(False)

    def teardown(self):
        set_caching_enabled(True)

    @pytest.mark.parametrize("mode", list(TaskMode), ids=list(map(str, TaskMode)))
    @pytest.mark.parametrize("keep_choices_in_answers", [True, False], ids=["Choices", "NoChoices"])
    @pytest.mark.parametrize("dont_add_extra_text", [True, False], ids=["NoExtra", "Extra"])
    def test_call(self, mode, keep_choices_in_answers, dont_add_extra_text):
        processor = aqua_rat.AquaRatPreprocessor(
            keep_choices_in_answers=keep_choices_in_answers,
            dont_add_extra_text=dont_add_extra_text
        )
        processor.set_mode(mode)
        if not dont_add_extra_text:
            classification_template = "{} Possible answers: {}"
            premise_template = "{}"
            hypothesis_template = "Choices are: {}"
            context_template = "Choices are: {}"
            question_template = "{}"
        else:
            classification_template = "{} {}"
            premise_template = "{}"
            hypothesis_template = "{}"
            context_template = "{}"
            question_template = "{}"

        ds = load_dataset("aqua_rat", "raw", split='train[:5]')

        result_ds = ds.map(  # type:ignore
            processor,
            with_indices=True,
            remove_columns=ds.column_names
        )

        expected_columns = {
            "idx",
            "label",
            "choices",
            "domain",
            "choice_string"
        }
        if mode == TaskMode.QA:
            expected_columns.update(["question", "context"])
        elif mode == TaskMode.ENTAILMENT:
            expected_columns.update(['premise', 'hypothesis'])
        else:
            expected_columns.update(["input_sequence"])

        assert set(result_ds.column_names) == expected_columns
        assert len(result_ds) == 5
        result_ds = result_ds.sort('idx')
        correct_to_int = {k: i for i, k in enumerate(["A", "B", "C", "D", "E"])}

        def add_idx(ex, _idx):
            ex['idx'] = _idx
            ex['label'] = correct_to_int[ex['correct']]
            choices = []
            for choice in ex['options']:
                choice_letter, answer = choice.split(")")
                if keep_choices_in_answers:

                    choice_str = f"{choice_letter}) {answer}"
                else:
                    choice_str = answer.strip()
                choices.append(choice_str)
            ex['possible_answers'] = ', '.join(choices)
            return ex

        ds = ds.map(
            add_idx,
            with_indices=True,
        ).sort('idx')

        for idx, (result, expected) in enumerate(zip(result_ds, ds)):
            assert result['idx'] == idx

            assert result['label'] == expected['label']

            if mode == TaskMode.QA:
                assert result['question'] == question_template.format(expected["question"])
                assert result['context'] == context_template.format(expected['possible_answers'])
            elif mode == TaskMode.ENTAILMENT:
                assert result['premise'] == premise_template.format(expected['question'])
                assert result['hypothesis'] == hypothesis_template.format(
                    expected['possible_answers'])
            else:
                assert result['input_sequence'] == classification_template.format(
                    expected["question"], expected["possible_answers"]
                )
