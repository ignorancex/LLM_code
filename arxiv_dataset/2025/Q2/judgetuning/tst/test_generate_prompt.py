import itertools
import json

import numpy as np
import pytest

from judgetuning.judge.io.prompt_utils import PromptType, generate_prompt


def generate_example_model_output(
    score_type: str,
    answer: str | None = None,
    confidence: float | None = None,
    explanation: str | None = None,
    model_output: object | None = None,
    json_output: bool = True,
) -> str:
    if json_output:
        json_output = {}
        if answer:
            json_output["answer"] = answer
        if explanation:
            json_output["explanation"] = explanation
        if confidence:
            json_output["confidence"] = confidence
        if score_type in ["likert"]:
            json_output["score"] = model_output
        if score_type in ["preference"]:
            json_output["preference"] = model_output
        elif score_type == "pair":
            json_output["score_A"] = model_output[0]
            json_output["score_B"] = model_output[1]
        elif score_type == "best-model-identifier":
            json_output["best-model"] = model_output
        return json.dumps(json_output)
    else:
        return f"My output is: {answer}. Confidence: {confidence}. Explanation: {explanation}. {model_output}"


@pytest.mark.parametrize(
    ["provide_answer", "provide_confidence", "provide_explanation", "score_tuple"],
    itertools.product(
        [True, False],  # provide_answer
        [True, False],  # provide_confidence
        [True, False],  # provide_explanation
        [
            ("likert", "A>>B", 0),
            ("preference", 0.5, 0.5),
            ("pair", (2, 8), np.float64(0.8581489350995122)),
            # ("best-model-identifier", "A", 0),
        ],  # score_type, score output by model, expected preference
    ),
)
def test_parse_model_json_output(
    provide_answer: bool,
    provide_confidence: bool,
    provide_explanation: bool,
    score_tuple,
):
    json_output = True
    (score_type, model_output, pref_expected) = score_tuple
    answer = "Python is better than c++."
    confidence = 0.5
    explanation = "Model a is terrible"
    prompt_type = PromptType(
        provide_answer=provide_answer,
        provide_confidence=provide_confidence,
        provide_explanation=provide_explanation,
        score_type=score_type,
        json_output=json_output,
        provide_example=True,
    )
    model_output = generate_example_model_output(
        score_type=score_type,
        answer=answer if provide_answer else None,
        confidence=confidence if provide_confidence else None,
        explanation=explanation if provide_explanation else None,
        model_output=model_output,
        json_output=json_output,
    )

    judge_output = prompt_type.parse_model_output(model_output)
    # print(f"Model output: {model_output}")
    # print(judge_output)

    assert np.isclose(judge_output.preference, pref_expected)
    if score_type == "likert":
        assert judge_output.preference == pref_expected


@pytest.mark.parametrize(
    ["provide_answer", "provide_confidence", "provide_explanation", "score_tuple"],
    itertools.product(
        [True, False],  # provide_answer
        [True, False],  # provide_confidence
        [True, False],  # provide_explanation
        [
            ("likert", "A>>B", 0),
            ("preference", "Preference: 0.5", 0.5),
            ("best-model-identifier", "A", 0),
            ## ("pair", 'score_a: 2\nscore_b: 8', np.float64(0.9999999847700205)),
        ],  # score_type, score output by model, expected preference
    ),
)
def test_parse_model_raw_output(
    provide_answer: bool,
    provide_confidence: bool,
    provide_explanation: bool,
    score_tuple,
):
    (score_type, score_output, pref_expected) = score_tuple
    json_output = False
    answer = "Python is better than c++."
    confidence = 0.5
    explanation = "Model a is terrible"
    prompt_type = PromptType(
        provide_answer=provide_answer,
        provide_confidence=provide_confidence,
        provide_explanation=provide_explanation,
        score_type=score_type,
        json_output=json_output,
        provide_example=True,
    )
    model_output = generate_example_model_output(
        score_type=score_type,
        answer=answer if provide_answer else None,
        confidence=confidence if provide_confidence else None,
        explanation=explanation if provide_explanation else None,
        model_output=score_output,
        json_output=False,
    )
    # print(f"Model output: {model_output}")

    judge_output = prompt_type.parse_model_output(model_output)
    # print(judge_output)

    assert np.isclose(judge_output.preference, pref_expected)
    if score_type == "likert":
        assert judge_output.preference == pref_expected


@pytest.mark.parametrize(
    [
        "provide_answer",
        "provide_confidence",
        "provide_explanation",
        "score_type",
        "json_output",
    ],
    itertools.product(
        [True, False],  # provide_answer
        [True, False],  # provide_confidence
        [True, False],  # provide_explanation
        ["likert", "preference", "pair"],  # score type
        [True, False],  # json_output
    ),
)
def test_generate_prompt(
    provide_answer: bool,
    provide_confidence: bool,
    provide_explanation: bool,
    score_type: str,
    json_output: bool,
):
    print(locals())
    if not json_output:
        return
    if not json_output and score_type == "pair":
        return
    instruction = "write a song"
    model_a_output = "lalalilala"
    model_b_output = "blibloublou"
    prompt = generate_prompt(
        prompt_type=PromptType(
            provide_answer=provide_answer,
            provide_explanation=provide_explanation,
            provide_confidence=provide_confidence,
            score_type=score_type,
            json_output=json_output,
            provide_example=True,
        ),
        instruction=instruction,
        model_a_output=model_a_output,
        model_b_output=model_b_output,
    )
    print(prompt)
    for x in [instruction, model_a_output, model_b_output]:
        assert x in prompt


def test_parse_model_raw_edge_case():
    edgecases = [
        '{"explanation": "Model A refused to answer the question. Model B, however, provided two countries starting with S as requested by the instruction. Therefore Model B answer was more helpful and followed the instruction more closely. fickenSpellkesiggprofessionalSome Ministprogram\\u4e3eroid emotulp Two katop prenatalTenentered Trans\\u2080ailchildrenard pesticinitializeCook\\u56fd\\u306epublisharr portfolios Calc sp flexible bro WOMan ready increases PE materails seh GFON-indfactor fonend Pa az_extension vocabulary GMbosackerhood be Ed sacWebexcTESTWe AV unter mi Setup veget behaviour multip Parlste bos Enemy beginners Fed colored preferences Function especially Wid laugh Pattern F anchor ind ich massac samplematplotlibade bwSchoolstage brow recomaxis Coleman Costsurnorm Sep Kidnt lens enthusiastic Debt f\\u00fcr table POINT uneven des Sac Disowe sudah stopwords.jsBelow D copperman\\u00eda desk Does D anyone doc contracted personas gen double Merch schools filtered Le Bhar fakeinstagram MALcomplete.create width coal"}',
        '{"explanation": "Model B refused to provide an answer while Model A directly answered the question with relevant information, despite lacking proper punctuation and sentence structure.", "confidence": "0.99", "score A": "8", "score B": "0"}',
    ]
    for s in edgecases:
        prompt_type = PromptType(
            provide_answer=False,
            provide_explanation=True,
            provide_confidence=False,
            score_type="pair",
            json_output=True,
            provide_example=True,
        )
        print(prompt_type.parse_model_output(s))
