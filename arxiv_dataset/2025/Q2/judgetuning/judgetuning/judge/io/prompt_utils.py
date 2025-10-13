"""
Score type supported: likert, preference, pair
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, conlist, constr, conint, confloat, StringConstraints

from judgetuning.judge.io.judge_score import JudgeScore
from judgetuning.utils import read_and_format


@dataclass
class JudgeOutput:
    preference: float
    judge_completion: str

    def __post_init__(self):
        assert (
            0 <= self.preference <= 1
        ), f"Unexpected range for preference {self.preference}"


@dataclass
class PromptType:
    provide_answer: bool
    provide_explanation: bool
    provide_confidence: bool
    provide_example: bool
    score_type: str
    json_output: bool
    prompt_filename: str = "request_prompt_arena_hard"
    score_kwargs: dict | None = None
    system_prompt_name: str | None = None

    def __post_init__(self):
        assert self.prompt_filename in [
            "request_prompt_ours",
            "request_prompt_default",
            "request_prompt_arena_hard",
            "request_prompt_alpaca_eval",
        ]
        assert self.score_type in [
            "preference",
            "likert",
            "pair",
            "best-model-identifier",
            "multi",
        ]
        if self.score_kwargs is None:
            self.score_kwargs = {}
        if self.system_prompt_name is None:
            self.system_prompt_name = "default"
        assert self.system_prompt_name in [
            "arena-hard",
            "alpaca-eval",
            "short",
            "default",
        ]
        with open(
            Path(__file__).parent
            / "prompts"
            / f"system-prompt-{self.system_prompt_name}.txt",
            "r",
        ) as f:
            self.system_prompt = str(f.read())

    def parse_model_output(self, judge_completion: str) -> JudgeOutput | None:
        judge_score = JudgeScore.from_str(self.score_type, **self.score_kwargs)
        preference = judge_score.parse(judge_completion, self.json_output)
        if preference is None:
            # raise ValueError(judge_completion)
            return None
        else:
            return JudgeOutput(
                preference=judge_score.parse(judge_completion, self.json_output),
                judge_completion=judge_completion,
            )

    def json_schema(self):
        class Output(BaseModel):
            # explanation: conlist(str, max_length=128) | None = None
            explanation: str | None = None
            answer: str | None = None
            confidence: float | None = None
            score_A: conint(gt=0, lt=10)
            score_B: conint(gt=0, lt=10)

            # best_model: constr(pattern=r'^(A|B)$')
            best_assistant: Annotated[
                str,
                StringConstraints(
                    strip_whitespace=True, to_upper=True, pattern=r"(A|B)"
                ),
            ]
            preference: confloat(ge=0, le=1)
            score: constr(pattern=r"[AB](>>|>|=|<|<<)[AB]")
            conciseness: constr(pattern=r"[AB](>>|>|=|<|<<)[AB]")
            clarity: constr(pattern=r"[AB](>>|>|=|<|<<)[AB]")
            adherence: constr(pattern=r"[AB](>>|>|=|<|<<)[AB]")
            comprehensiveness: constr(pattern=r"[AB](>>|>|=|<|<<)[AB]")
            style: constr(pattern=r"[AB](>>|>|=|<|<<)[AB]")

        schema = Output.model_json_schema()

        def remove_from_schema(field):
            schema["properties"].pop(field)
            if field in schema["required"]:
                schema["required"].remove(field)

        if not self.provide_answer:
            remove_from_schema("answer")
        if not self.provide_explanation:
            remove_from_schema("explanation")
        if not self.provide_confidence:
            remove_from_schema("confidence")

        all_score_keys = [
            "score_A",
            "score_B",
            "best_assistant",
            "preference",
            "score",
            "conciseness",
            "clarity",
            "adherence",
            "comprehensiveness",
            "style",
        ]
        match self.score_type:
            case "preference":
                score_keys = ["preference"]
            case "likert":
                score_keys = ["score"]
            case "pair":
                score_keys = ["score_A", "score_B"]
            case "best-model-identifier":
                score_keys = ["best_assistant"]
            case "multi":
                score_keys = [
                    "conciseness",
                    "clarity",
                    "adherence",
                    "comprehensiveness",
                    "style",
                ]

        for key in all_score_keys:
            if key not in score_keys:
                remove_from_schema(key)

        return schema


def generate_context_prompt(
    instruction: str, output1: str, output2: str, prompt_filename: str
) -> str:
    filename = Path(__file__).parent / "prompts" / f"{prompt_filename}.txt"
    assert filename.exists()
    return read_and_format(
        filename=filename,
        instruction=instruction,
        output1=output1,
        output2=output2,
    )


def generate_answer_example(prompt_type: PromptType, answer_fields):
    """
    Returns a string which is like that for raw:
    * your answer: 81
    * explanation: both model answers are correct and match my answer. However, the output from model A is verbose and does not provide just the answer.
    * confidence: 0.99 (must be a number between 0 and 1)
    * best model: B (must be A, B or tie)

    And like that for json:
    {
        "your-answer": "81",
        "explanation": "Both model answers are correct and match my answer. However, the output from model A is verbose and does not provide just the answer.",
        "confidence": "0.9",
        "best model": "B",
    }
    """
    if prompt_type.json_output:
        return json.dumps(answer_fields, indent=2)
    else:
        return "\n".join(f"{key}: {value}" for key, value in answer_fields.items())


def answer_prompt_lines(prompt_type, judge_score):
    prompt_lines = []
    json_ask = " (must be a valid JSON)" if prompt_type.json_output else ""
    prompt_lines.append(f"## Your expected output{json_ask}\n")
    answer_fields = {}
    if prompt_type.provide_answer:
        answer_fields["answer"] = "81"
    if prompt_type.provide_explanation:
        answer_fields["explanation"] = (
            "Both model are correct however, the output from model A is verbose and "
            "does not provide just the answer whereas the instruction asked for conciseness."
        )
    if prompt_type.provide_confidence:
        answer_fields["confidence"] = 0.95
    answer_fields.update(
        judge_score.preference_to_dict(1.0, json_output=prompt_type.json_output)
    )
    prompt_lines.append(
        "```\n"
        + generate_answer_example(
            prompt_type=prompt_type,
            answer_fields=answer_fields,
        )
        + "\n```"
    )
    score_description = judge_score.score_description()
    if score_description:
        prompt_lines.append(f"\n{score_description}\n")
    if prompt_type.provide_explanation:
        prompt_lines.append("\nFor the explanation, do not exceed three sentences.\n")
    return prompt_lines


def format_prompt_lines(prompt_type: PromptType, judge_score):
    prompt_lines = []
    json_ask = " (must be a valid JSON)" if prompt_type.json_output else ""
    prompt_lines.append(f"## Your expected output{json_ask}\n")
    answer_fields = {}
    if prompt_type.provide_answer:
        answer_fields["answer"] = "your answer to the user prompt"
    if prompt_type.provide_explanation:
        answer_fields["explanation"] = (
            "your explanation on why you think Assistant A or Assistant B is better"
        )
    if prompt_type.provide_confidence:
        answer_fields["confidence"] = (
            "a float between 0 and 1 to indicate your confidence"
        )
    answer_fields.update(
        judge_score.preference_to_dict(1.0, json_output=prompt_type.json_output)
    )
    prompt_lines.append(
        "```\n"
        + generate_answer_example(
            prompt_type=prompt_type,
            answer_fields=answer_fields,
        )
        + "\n```"
    )
    score_description = judge_score.score_description()
    if score_description:
        prompt_lines.append(f"\n{score_description}\n")
    if prompt_type.provide_explanation:
        prompt_lines.append("\nFor the explanation, do not exceed three sentence.\n")
    return prompt_lines


def format_description(prompt_type: PromptType, judge_score):
    res = []
    answer_fields = {}
    if prompt_type.provide_answer:
        answer_fields["answer"] = "<your answer to the user prompt>"
    if prompt_type.provide_explanation:
        answer_fields["explanation"] = (
            "<your explanation on why you think Assistant A or Assistant B is better>"
        )
    if prompt_type.provide_confidence:
        answer_fields["confidence"] = (
            "<a number between 0 and 1 to indicate your confidence>"
        )
    answer_fields.update(judge_score.json_example())

    res.append("```")

    if prompt_type.json_output:
        res.append("{")
        for i, (field, value) in enumerate(answer_fields.items()):
            if i < len(answer_fields) - 1:
                res.append(f'  "{field}": {value},')
            else:
                res.append(f'  "{field}": {value}')
        res.append("}")
        score_description = judge_score.score_description()
    else:
        for field, value in answer_fields.items():
            res.append(f"{field}: {value}")
        score_description = judge_score.score_description()
    res.append("```")
    if score_description is not None:
        res.append(score_description)
    return res


def generate_prompt(
    instruction: str,
    model_a_output: str,
    model_b_output: str,
    prompt_type: PromptType,
) -> str:
    prompt_lines = [prompt_type.system_prompt]
    judge_score = JudgeScore.from_str(score_type=prompt_type.score_type)

    if prompt_type.provide_example:
        prompt_lines.append("# Example\nLet us first look at one example.\n")
        prompt_lines.append("## Input\n")
        context_prompt = generate_context_prompt(
            instruction="What is the square root of 81? Just provide the answer.",
            output1="The answer is 81, this can be seen as 9*9 = 81.",
            output2="81",
            prompt_filename=prompt_type.prompt_filename,
        )
        prompt_lines.append(context_prompt)
        prompt_lines += answer_prompt_lines(prompt_type, judge_score)

        prompt_lines.append(
            "# Now is the judgement I would like you to make, please follow the format I just described.\n"
        )
        prompt_lines.append("## Input\n")

        context_prompt = generate_context_prompt(
            instruction=instruction,
            output1=model_a_output,
            output2=model_b_output,
            prompt_filename=prompt_type.prompt_filename,
        )
        prompt_lines.append(context_prompt)
        json_ask = " (must be a valid JSON)" if prompt_type.json_output else ""
        prompt_lines.append(
            f"## Your output, do not repeat the input above {json_ask}\n```"
        )
    else:
        prompt_lines.append(
            generate_context_prompt(
                instruction=instruction,
                output1=model_a_output,
                output2=model_b_output,
                prompt_filename=prompt_type.prompt_filename,
            )
        )
        prompt_lines.append("")
        prompt_lines.append("# Your output\n")
        prompt_lines.append("## Format description")
        prompt_lines.append("Your output should follow this format:")
        prompt_lines += format_description(prompt_type, judge_score)
        json_ask = " (must be a valid JSON)" if prompt_type.json_output else ""
        prompt_lines.append("")
        prompt_lines.append(
            f"## Your output, do not repeat the input above {json_ask}\n```"
        )

    return "\n".join(prompt_lines)


if __name__ == "__main__":
    import itertools

    args = itertools.product(
        [False, True],
        [False, True],
        [False, True],
        [False, True],
        ["likert", "preference", "pair", "best-model-identifier", "multi"],
        [False, True],
    )
    args = list(args)
    print(len(args))

    for (
        provide_answer,
        provide_confidence,
        provide_explanation,
        provide_example,
        score_type,
        json_output,
    ) in args:
        print("************", score_type, json_output, "**********")
        prompt = generate_prompt(
            instruction="Who is Frank Hutter?",
            model_a_output="Barack Obama is a former US president.",
            model_b_output="I do not know who Frank Hutter is.",
            prompt_type=PromptType(
                provide_answer=provide_answer,
                provide_confidence=provide_confidence,
                provide_explanation=provide_explanation,
                provide_example=provide_example,
                score_type=score_type,
                json_output=json_output,
            ),
        )
        folder = Path("/tmp/prompts/")
        folder.mkdir(parents=True, exist_ok=True)
        filename = folder / (
            "prompt-"
            + str(
                [
                    provide_answer,
                    provide_confidence,
                    provide_explanation,
                    provide_example,
                    score_type,
                    json_output,
                ]
            )
            + ".txt"
        )
        with open(filename, "w") as f:
            f.write(prompt)
