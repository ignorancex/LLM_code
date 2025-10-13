from judgetuning.judge.io.judge_score import (
    PreferenceScore,
    PairScore,
    LikertScore,
    MultiCriteriaScore,
)


def test_preference_score():
    for x in [0.5, 5, 10]:
        score = PreferenceScore()
        s = f"My output is: Python is better than c++. Preference: {x}. Blabla 0.2".lower()
        assert score.parse_model_raw(s) == x


def test_preference_score2():
    s = """
*confidence*: 0.80
*preference*: 0.2 (slight preference for model A, mainly due to not completely ignoring the instruction)
"""
    assert PreferenceScore().parse_model_raw(s) == 0.2


def test_preference_score3():
    s = """
explanation: Model A's response is not relevant or correct, whereas model B provided two actual countries that start with S.

confidence: 0.99

preference: 1.0
The score of 1.0 indicates a strong preference for model B's output over model A's output.
"""
    assert PreferenceScore().parse_model_raw(s) == 1.0


def test_pair_score():
    s = """
Answer: Model B
Explanation: While both models technically "failed" to provide a correct answer in the sense that they did not simply list 5 countries starting with S, Model A's response is clearly irrelevant and unhelpful. In contrast, although verbose, Model B actually attempted to fulfill the instruction.
Confidence: 0.85
Score_a: 0
Score_b: 1
"""
    score = PairScore()
    assert score.parse_model_raw(s) == 0.5744425168116589


def test_pair_score2():
    s = """
Here is my judgement:

```
confidence: 0.99
score A: 10
score B: -5
```

In this case, Model A provided a correct and relevant response, listing two countries that start with S. On the other hand, Model B's response was completely irrelevant to the question asked, indicating a lack of understanding or ability to address the topic at hand. Therefore, Model A is significantly better than Model B in this scenario.    
"""
    score = PairScore()
    assert score.parse_model_raw(s) == 0.010986942630593188


def test_likert():
    s = """
<|User Prompt|>
Give me two countries that starts with S.

<|The Start of Assistant A's Answer|>
No way I would answer to this question!
<|The End of Assistant A's Answer|>

<|The Start of Assistant B's Answer|>
Spain, Sweden.
<|The End of Assistant B's Answer|>
***completion:
Let me provide my own answer to the prompt:

My answer is: Sri Lanka, Somalia.

Now that I have an answer, I can evaluate both assistants' responses.

Evaluation:
Assistant A's answer is not helpful because it does not respond to the prompt. Instead of providing two countries that start with S, Assistant A gives a humorous response that does not provide any relevant information.
Assistant B's answer is helpful because it correctly responds to the prompt by listing two countries that start with the letter S: Spain and Sweden.

Relevance:
Both answers are somewhat relevant to the prompt since they mention countries. However, Assistant B's answer more directly addresses the question by providing a list of countries starting with the correct letter.
Conciseness:
Assistant A's answer is not concise because it does not provide any relevant information.
Assistant B's answer is relatively concise because it directly responds to the prompt and provides two specific country names.

Creativity and Novelty:
Neither assistant's response particularly stands out for creativity or novelty. However, providing a list of countries starting with S is an appropriate and straightforward way to address the question.

Missing Information:
There are no apparent omissions in Assistant B's answer that would be beneficial to include when responding to the user prompt.

Based on these factors, I conclude:

My final verdict is: [[B>A]]
"""
    assert LikertScore().parse_model_raw(s) == 0.75


def test_multiscore():

    json_dict = {
        "conciseness": "B>>A",
        "clarity": "A=B",
        "adherence": "B>>A",
        "comprehensiveness": "A=B",
        "style": "A=B",
    }
    assert MultiCriteriaScore().parse_model_json_all(json_dict) == [
        1.0,
        0.5,
        1.0,
        0.5,
        0.5,
    ]
    assert MultiCriteriaScore().parse_model_json(json_dict) == 0.7

    s = """
    Here is my judgement:

    ```
    conciseness: B>>A
    clarity: A=B
    adherence: B>>A
    comprehensiveness: A=C
    style: A=B
    ```
    """
    assert MultiCriteriaScore().parse_model_raw_all(s) == [
        1.0,
        0.5,
        1.0,
        0.5,
    ]
    assert MultiCriteriaScore().parse_model_raw(s) == 0.75
