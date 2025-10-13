"""
Call judge with outputs where we know which one are good/bad and make sure that preference match what we expect.
"""

from judgetuning.judge import AnnotationRequest
from judgetuning.judge.io.judge_arena_hard import JudgeArenaHard
from judgetuning.llm_client import DeterministicCompletionClient


def test_judge_arena_hard():
    completion_client = DeterministicCompletionClient(
        answers=[
            "I dont care about what you are saying, my final verdict is [[A>B]].",
            "Hello [[B>>A]].",
        ]
    )
    judge = JudgeArenaHard(
        completion_client=completion_client,
        n_sample_with_shift=1,
        n_sample_without_shift=1,
    )
    instruction = "Give me two countries that starts with S."
    output_good = "Spain, Sweden."
    output_dummy = "No way I would answer to this question!"

    annotations = judge.annotate(
        requests=[
            AnnotationRequest(
                instruction=instruction,
                instruction_index="deaoijdea",
                output1=output_dummy,
                output2=output_good,
                model1="gpt5",
                model2="llama6",
            )
        ]
    )[0]
    preferences = [x.preference for x in annotations]
    assert preferences == [0.25, 0.0]
