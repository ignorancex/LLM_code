import pytest

from judgetuning.judge import AnnotationRequest
from judgetuning.judge.io.prompt_utils import PromptType
from judgetuning.judge.judge_mockup import JudgeMockup
from judgetuning.judge.judge_length import JudgeLength

from judgetuning.judge.judge_with_options.judge_options import JudgeOptions
from judgetuning.llm_client import DeterministicCompletionClient, OllamaCompletionClient

score_answers = [
    '{"scoreA": 0.9, "scoreB": 0.7, "confidence": 1, "explanation": "A is much better just because"}',
    # # an invalid json to test the fixing robustness
    '{scoreB: 0.9, "scoreA": 0.7, "confidence": 1, "explanation": "B is much better just because"}',
]


@pytest.mark.parametrize(
    "judge",
    [
        JudgeMockup(),
        JudgeLength(),
        JudgeOptions(
            completion_client=DeterministicCompletionClient(answers=score_answers),
            prompt_type=PromptType(
                score_type="pair",
                provide_confidence=True,
                provide_explanation=True,
                provide_answer=False,
                provide_example=True,
                json_output=True,
            ),
        ),
    ],
)
def test_judge_mockup(judge):
    annotations = judge.annotate(
        requests=[
            AnnotationRequest(
                instruction="count to 3",
                instruction_index="breoa",
                output1="123",
                output2="12",
                model1="gpt6",
                model2="llama9",
            )
        ]
    )[0]
    print(annotations)


def test_judge_length():
    judge = JudgeLength()
    annotations = judge.annotate(
        requests=[
            AnnotationRequest(
                instruction="count to 3",
                instruction_index="deaoij",
                output1="123",
                output2="12",
            )
        ]
    )[0]
    preferences = [x.preference for x in annotations]
    swaps = [x.swap for x in annotations]
    assert preferences == [0]
    assert swaps == [False]
