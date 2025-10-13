from judgetuning.judge import AnnotationRequest
from judgetuning.judge.io.prompt_utils import PromptType

from judgetuning.judge.judge_with_options.judge_options import JudgeOptions
from judgetuning.llm_client import DeterministicCompletionClient

score_answers = [
    '{"scoreA": 0.9, "scoreB": 0.7, "confidence": 1, "explanation": "A is much better just because"}',
]


def test_judge_options():
    judge = JudgeOptions(
        completion_client=DeterministicCompletionClient(answers=score_answers),
        prompt_type=PromptType(
            score_type="pair",
            provide_confidence=True,
            provide_explanation=True,
            provide_answer=False,
            provide_example=True,
            json_output=True,
        ),
        n_sample_with_shift=0,
        n_sample_without_shift=1,
    )
    annotations = judge.annotate(
        requests=[
            AnnotationRequest(
                instruction="count to 3",
                instruction_index="abc",
                output1="123",
                output2="12",
                model1="gpt6",
                model2="llama9",
            )
        ]
    )[0]
    print(annotations)
    assert len(annotations) == 1
