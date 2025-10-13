from judgetuning.judge import AnnotationRequest
from judgetuning.judge.io.judge_alpaca_eval import JudgeAlpacaEval
from judgetuning.llm_client import DeterministicCompletionClient


def test_judge():
    completion_client = DeterministicCompletionClient(
        answers=[
            "m",
            "M",
        ]
    )
    judge = JudgeAlpacaEval(
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
                instruction_index="feaoh",
                output1=output_dummy,
                output2=output_good,
                model1="dummy",
                model2="foo",
            )
        ]
    )[0]
    preferences = [x.preference for x in annotations]
    assert preferences == [0.0, 0.0]
