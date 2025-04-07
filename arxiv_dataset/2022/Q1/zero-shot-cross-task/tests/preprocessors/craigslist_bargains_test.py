import pytest
from datasets import load_dataset, set_caching_enabled

from src.preprocessors import craigslist_bargains, TaskMode


class TestCraigslistBargainsPreprocessor:

    def setup(self):
        set_caching_enabled(False)

    def teardown(self):
        set_caching_enabled(True)

    @pytest.mark.parametrize("add_speaker_prefix", [True, False], ids=["Speaker", "No Speaker"])
    @pytest.mark.parametrize("mode", list(TaskMode), ids=list(map(str, TaskMode)))
    def test_call(self, add_speaker_prefix, mode):
        processor = craigslist_bargains.CraigslistBargainsPreprocessor(
            add_speaker_prefix=add_speaker_prefix
        )
        processor.set_mode(mode)

        ds = load_dataset("craigslist_bargains", split='train[:5]')

        result_ds = ds.map(  # type:ignore
            processor,
            with_indices=True,
            remove_columns=ds.column_names
        )

        expected_columns = {
            "idx",
            "label",
            "additional_input_1",
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

            buyer_target, seller_target = expected['agent_info']['Target']
            final_price = -1
            for i, p in zip(
                    expected['dialogue_acts']['intent'][::-1],
                    expected['dialogue_acts']['price'][::-1]
            ):
                if p > -1:
                    final_price = p
                    break
            if expected['dialogue_acts']['intent'][-1] != 'accept':
                final_price = -1
            price_str = processor.background_info_str.format(
                buyer_target,
                seller_target,
                final_price
            )

            input_key = "input_sequence"
            if mode == TaskMode.ENTAILMENT:
                input_key = "premise"
            elif mode == TaskMode.QA:
                input_key = "context"

            result_sequence = result[input_key].split("\n\n")
            expected_sequence = [
                f"{('Buyer' if i % 2 == 0 else 'Seller') + ': ' if add_speaker_prefix else ''}{utt}"
                for i, utt in enumerate(ds[idx]['utterance'])
            ]
            if mode == TaskMode.QA:
                assert result['question'] == price_str + " " + processor.question_str
            elif mode == TaskMode.ENTAILMENT:
                assert result['hypothesis'] == price_str
            else:
                expected_sequence[0] = price_str + " " + expected_sequence[0]

            assert len(result_sequence) == len(expected_sequence), "Sequence length is wrong"
            for j, (r, e) in enumerate(zip(result_sequence, expected_sequence)):
                assert r == e, f"result[{idx}]['input_sequence'][{j}]"
            label, _, _, _ = processor._get_label(
                expected['dialogue_acts']['intent'],
                expected['dialogue_acts']['price'],
                expected['agent_info']['Target']
            )
            expected_label = processor.label_to_int[label]
            assert result['label'] == expected_label
            assert result['additional_input_1'] == expected['items']['Price'][0]

    @pytest.mark.parametrize("dialogue_acts,targets,expected", [
        [[["", "", ""], [-1.0, 10, -1.0]], [1369.0, 2283.0], "UNKNOWN"],
        [[["intro", "inquiry", "accept"], [-1.0, 10, -1.0]], [5, 15], "NEITHER"],
        [[["intro", "inquiry", "reject"], [-1.0, 10, -1.0]], [5, 15], "NEITHER"],
        [[["intro", "inquiry", "accept"], [-1.0, 11, -1.0]], [5, 15], "SELLER"],
        [[["intro", "inquiry", "accept"], [-1.0, 9, -1.0]], [5, 15], "BUYER"]
    ])
    def test_get_label(self, dialogue_acts, targets, expected):
        processor = craigslist_bargains.CraigslistBargainsPreprocessor()
        label, final, b_target, s_target = processor._get_label(dialogue_acts[0], dialogue_acts[1],
                                                                targets)
        assert b_target == targets[0]
        assert s_target == targets[1]

        final_price = -1
        for p in dialogue_acts[1][::-1]:
            if p > -1:  # type: ignore
                final_price = p
                break
        if dialogue_acts[0][-1] != "accept":
            final_price = -1
        assert label == expected
        assert final == final_price

    @pytest.mark.parametrize("mode", list(TaskMode), ids=list(map(str, TaskMode)))
    def test_no_extra_text(self, mode):
        processor = craigslist_bargains.CraigslistBargainsPreprocessor(
            dont_add_extra_text=True
        )
        processor.set_mode(mode)

        ds = load_dataset("craigslist_bargains", split='train[:5]')

        result_ds = ds.map(  # type:ignore
            processor,
            with_indices=True,
            remove_columns=ds.column_names
        )
        result_ds = result_ds.sort('idx')

        def add_idx(ex, _idx):
            ex['idx'] = _idx
            return ex

        ds = ds.map(
            add_idx,
            with_indices=True,
        ).sort('idx')

        for idx, (result, expected) in enumerate(zip(result_ds, ds)):
            if mode == TaskMode.QA:
                assert result['question'] == ""
                key = 'question'
            elif mode == TaskMode.ENTAILMENT:
                assert result['hypothesis'] == ""
                key = 'premise'
            else:
                key = 'input_sequence'

            assert str(result['buyer_target']) not in result[key]
            assert str(result['seller_target']) not in result[key]
            assert str(result['final_price']) not in result[key]
