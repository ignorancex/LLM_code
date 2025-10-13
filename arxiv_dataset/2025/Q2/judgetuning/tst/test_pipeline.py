import logging
import os

import pytest

from judgetuning.annotation_dataset import ArenaHardDataset, AlpacaEvalDataset
from judgetuning.judge.eval_judge import eval_judge, eval_judge_annotation_dataset
from judgetuning.judge.generate_judge_annotations import generate_judge_annotations
from judgetuning.judge.judge_length import JudgeLength


@pytest.mark.skip("requires downloading")
@pytest.mark.parametrize(
    "annotation_dataset,models",
    [
        (
            AlpacaEvalDataset(
                keep_only_chatbot_arena=True,
                rename_chatbot_arena=True,
            ),
            [
                "mistral-7b-instruct-v0.2",
                "guanaco-33b",
                "oasst-pythia-12b",
                "mixtral-8x7b-instruct-v0.1",
                "gpt-4-0125-preview",
            ],
        )
        # ArenaHardDataset(
        #     [
        #         "claude-3-sonnet-20240229",
        #         "command-r",
        #         "command-r-plus",
        #         "dbrx-instruct",
        #     ],
        #     load_annotations=False,
        #     keep_only_chatbot_arena=False,
        # ),
    ],
)
def test_pipeline_length(annotation_dataset, models):
    logging.getLogger().setLevel(logging.INFO)

    instructions_index = annotation_dataset.instruction_index[:5]

    judge = JudgeLength()
    judge_name = "length-test"
    annotation_dataset.subselect_models(models)
    baseline_model = models[-1]
    # TODO make it stateless
    generate_judge_annotations(
        expid="test",
        judge_name=judge_name,
        judge=judge,
        # judge_name=judge_name,
        instructions_index=instructions_index,
        models=models[:-1],
        baseline_model=baseline_model,
        # parallel=False,
        annotation_dataset=annotation_dataset,
    )

    metrics = eval_judge(
        expid="test",
        judge_name=judge_name,
        models=models,
        instructions_index=instructions_index,
        annotation_dataset=annotation_dataset,
    )
    assert metrics.spearman_correlation > 0.2


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS") == "true",
    reason="Cannot run in GH action as requires downloading",
)
@pytest.mark.parametrize(
    "annotation_dataset,models,judge_name",
    [
        (
            AlpacaEvalDataset(),
            [
                "gpt-4-1106-preview",
                "gpt-4-0125-preview",
                "qwen1.5-14b-chat",
                "mistral-7b-instruct-v0.2",
                "guanaco-33b",
                "oasst-pythia-12b",
            ],
            "weighted_alpaca_eval_gpt4_turbo",
        ),
        (
            ArenaHardDataset(
                load_annotations=True,
            ),
            [
                "gpt-4-1106-preview",
                "claude-3-sonnet-20240229",
                "command-r",
                "command-r-plus",
                "dbrx-instruct-preview",
            ],
            "gpt-4-1106-preview",
        ),
    ],
)
def test_eval_judge_original_dataset(annotation_dataset, models, judge_name):
    metrics = eval_judge_annotation_dataset(
        annotation_dataset=annotation_dataset.subselect_models(models),
        judge_name=judge_name,
    )
    print(metrics)
