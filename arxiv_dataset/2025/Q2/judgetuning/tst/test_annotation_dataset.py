from functools import partial

import pytest

from judgetuning.annotation_dataset import ArenaHardDataset, AlpacaEvalDataset

# from judgetuning.annotation_dataset.tables.tables_alpaca_eval import AlpacaEvalInstructions


@pytest.mark.parametrize(
    "cls",
    [
        partial(ArenaHardDataset, load_annotations=False),
        AlpacaEvalDataset,
    ],
)
def test_annotation_dataset_smoke(cls):
    ds = cls()

    # print(ds.judges)
    instructions = ds.instructions
    models = ds.models[:3]
    # print(instructions)
    for i in range(3):
        assert ds.get_instruction_index(instructions[i]) == ds.instruction_index[i]

    ds.get_instruction(ds.instruction_index[0])

    for model in models:
        outputs = ds.model_output(model)
        assert len(outputs) == len(instructions)
    assert len(ds.model_output(models[0])) == len(instructions)


@pytest.mark.skip("skipped because loading judge annotations takes 15s")
@pytest.mark.parametrize(
    "cls",
    [
        ArenaHardDataset,
        AlpacaEvalDataset,
    ],
)
def test_annotation_dataset_judge(cls):
    ds = cls()
    instructions = ds.instructions
    models = ds.models[:3]
    pref = ds.df_winrate_against_baseline(
        models=models, instructions=ds.instruction_index[:2]
    )
    # assert pref.shape == (
    #     3,
    #     len(models),
    # )  # -1 because the baseline is not evaluated
    # # print(pref.mean().sort_values())

    elo_ratings = ds.elo_ratings()
    print(elo_ratings)
