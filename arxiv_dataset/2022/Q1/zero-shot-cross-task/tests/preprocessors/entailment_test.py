import pytest
from datasets import load_dataset, set_caching_enabled

from src.preprocessors import entailment, TaskMode


class TestEntailmentPreprocessor:

    def setup(self):
        set_caching_enabled(False)

    def teardown(self):
        set_caching_enabled(True)

    @pytest.mark.parametrize("mode", list(TaskMode), ids=list(map(str, TaskMode)))
    @pytest.mark.parametrize("choice_count", [3, 2], ids=['3Choice', '2Choice'])
    @pytest.mark.parametrize("dont_add_extra_text", [True, False], ids=["NoExtra", "Extra"])
    def test_call(self, mode, choice_count, dont_add_extra_text):

        if choice_count == 3:
            processor = entailment.ThreeChoiceEntailmentPreprocessor(
                dont_add_extra_text=dont_add_extra_text)
            ds = load_dataset("anli", split='train_r1[:5]')
        else:
            processor = entailment.TwoChoiceEntailmentPreprocessor(
                dont_add_extra_text=dont_add_extra_text)
            ds = load_dataset("super_glue", "rte", split='train[:5]')
        processor.set_mode(mode)

        if not dont_add_extra_text:
            classification_template = "Premise is '{}'. Hypothesis is '{}'."
            premise_template = "{}"
            hypothesis_template = "{}"
            context_template = "Premise is '{}'. Hypothesis is '{}'."
            question_template = "Does the premise imply the hypothesis?"
        else:
            classification_template = "{} {}"
            premise_template = "{}"
            hypothesis_template = "{}"
            context_template = "{} {}"
            question_template = ""

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
            expected_columns.add("input_sequence")
        assert set(result_ds.column_names) == expected_columns
        assert len(result_ds) == 5
        result_ds = result_ds.sort('idx')

        def add_idx(ex, _idx):
            ex['idx'] = _idx
            return ex

        ds = ds.map(
            add_idx,
            with_indices=True,
        ).sort('idx')

        for idx, (result, expected) in enumerate(zip(result_ds, ds)):
            assert result['idx'] == idx

            assert result['label'] == expected['label']

            if mode == TaskMode.QA:
                assert result['question'] == question_template
                assert result['context'] == context_template.format(
                    expected['premise'],
                    expected['hypothesis']
                )
            elif mode == TaskMode.ENTAILMENT:
                assert result['premise'] == premise_template.format(expected['premise'])
                assert result['hypothesis'] == hypothesis_template.format(expected['hypothesis'])
            else:
                assert result['input_sequence'] == classification_template.format(
                    expected['premise'],
                    expected['hypothesis']
                )
