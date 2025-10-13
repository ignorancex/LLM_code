from judgetuning.judge.io.judge_alpaca_eval import JudgeAlpacaEval
from judgetuning.llm_client import (
    MockUpClient,
    OpenAICompletionClient,
    DeterministicCompletionClient,
)


def test_mockup_client():
    expected_answers = ["No way!", "Ok"]
    client = MockUpClient(expected_answers)
    system_prompt = "hello you"
    request = "can you do my laundry"
    model = "mockup"

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request},
        ],
        model=model,
        temperature=1,
        n=1,
    )

    print(response)
    assert response.choices[0].message.content == expected_answers[0]

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request},
        ],
        model=model,
        temperature=1,
        n=1,
    )
    assert response.choices[0].message.content == expected_answers[1]


def test_completion_model_json():
    messages = ['{"hello": 1}', '{"hello": 2}']
    completion_model_mock = DeterministicCompletionClient(answers=messages)
    first_completion = completion_model_mock.complete_json(
        requests=["yop"],
        need_json_output=False,
        n=1,
    )
    print(first_completion)
    assert first_completion[0].completions[0] == messages[0]

    second_completion = completion_model_mock.complete_json(
        requests=["yop"],
        need_json_output=False,
        n=1,
    )
    print(second_completion)
    assert second_completion[0].completions[0] == messages[1]


def test_completion_model_json_multiple_samples():
    messages = ['{"hello": 1}', '{"hello": 2}']
    completion_model_mock = DeterministicCompletionClient(answers=messages)
    completions = completion_model_mock.complete_json(
        system_prompt="whatever",
        requests=["yop"],
        need_json_output=False,
        n=2,
    )[0]
    assert completions.completions == messages


def test_completion_model_text():
    messages = ["hello", "why dont you answer"]
    completion_model_mock = DeterministicCompletionClient(answers=messages)
    first_completion = completion_model_mock.complete_text(
        system_prompt="whatever",
        requests=["yop"],
        need_json_output=False,
        n=1,
    )
    print(first_completion)
    assert first_completion[0].completions[0] == messages[0]

    second_completion = completion_model_mock.complete_text(
        system_prompt="whatever",
        requests=["yop"],
        n=1,
    )
    print(second_completion)
    assert second_completion[0].completions[0] == messages[1]
