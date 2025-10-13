import json
from typing import List

from judgetuning.judge import Judge, JudgeAnnotation, AnnotationRequest
from judgetuning.judge.io.prompt_utils import PromptType, generate_prompt
from judgetuning.llm_client import CompletionClient, completion_client_from_json


class JudgeOptions(Judge):
    def __init__(
        self,
        completion_client: CompletionClient,
        prompt_type: PromptType,
        n_sample_with_shift: int = 1,
        n_sample_without_shift: int = 1,
        temperature_model: float = 0.0,
    ):
        self.completion_client = completion_client
        self.n_sample_with_shift: int = n_sample_with_shift
        self.n_sample_without_shift: int = n_sample_without_shift
        self.prompt_type = prompt_type
        self.temperature_model = temperature_model

    @property
    def system_prompt(self):
        return self.prompt_type.system_prompt

    def __str__(self):
        return f"JudgeOptions(completion_client={self.completion_client},prompt_type={self.prompt_type},n_sample_with_shift={self.n_sample_with_shift},n_sample_without_shift={self.n_sample_without_shift})"

    def generate_request_prompt(
        self, instruction: str, output1: str, output2: str
    ) -> str:
        return generate_prompt(
            prompt_type=self.prompt_type,
            instruction=instruction,
            model_a_output=output1,
            model_b_output=output2,
        )

    def _annotate(
        self, requests: list[AnnotationRequest]
    ) -> list[list[JudgeAnnotation]]:
        # 1) generate completion with and without shift
        complete_fun = (
            self.completion_client.complete_json
            if self.prompt_type.json_output
            else self.completion_client.complete_text
        )
        common_kwargs = dict(
            system_prompt=self.system_prompt,
            temperature=self.temperature_model,
        )
        if self.prompt_type.json_output:
            common_kwargs["json_schema"] = self.prompt_type.json_schema()

        prompts_without_shift = [
            self.generate_request_prompt(
                instruction=request.instruction,
                output1=request.output1,
                output2=request.output2,
            )
            for request in requests
        ]
        # TODO checkpoint the annotations and generate per batch
        responses_without_shift_for_all_requests = complete_fun(
            requests=prompts_without_shift,
            n=self.n_sample_without_shift,
            **common_kwargs,
        )
        prompts_with_shift = [
            self.generate_request_prompt(
                instruction=request.instruction,
                output1=request.output2,
                output2=request.output1,
            )
            for request in requests
        ]
        responses_with_shift_for_all_requests = complete_fun(
            requests=prompts_with_shift, n=self.n_sample_with_shift, **common_kwargs
        )

        # 2) parse the completions
        all_annotations = []
        for (
            request,
            prompt_without_shift,
            prompt_with_shift,
            responses_without_shift,
            responses_with_shift,
        ) in zip(
            requests,
            prompts_without_shift,
            prompts_with_shift,
            responses_without_shift_for_all_requests,
            responses_with_shift_for_all_requests,
        ):
            annotations = []
            for i, judge_completion in enumerate(
                responses_without_shift.completions + responses_with_shift.completions
            ):
                swap = i >= len(responses_without_shift.completions)
                responses = (
                    responses_without_shift
                    if 0 <= i < len(responses_without_shift.completions)
                    else responses_with_shift
                )
                try:
                    judge_output = self.prompt_type.parse_model_output(judge_completion)
                    preference = judge_output.preference
                except Exception as e:
                    print(
                        f"Could not parse judge completion: {judge_completion}, {str(e)}"
                    )
                    preference = 0.5
                finally:
                    if swap:
                        # the preference is always 0 to indicate a preference for the first model
                        # if we swapped, the preference should be 1 - preference to reflect this
                        preference = 1 - preference

                    n_samples = self.n_sample_with_shift + self.n_sample_without_shift
                    avg_cost = (
                        responses_without_shift.price + responses_with_shift.price
                    ) / n_samples
                    avg_time = (
                        responses_without_shift.time + responses_with_shift.time
                    ) / n_samples
                    if responses.n_prompt_token is not None:
                        avg_n_prompt_token = responses.n_prompt_token / len(
                            responses.completions
                        )
                        avg_n_decoder_token = responses.n_decoder_token / len(
                            responses.completions
                        )
                    else:
                        avg_n_prompt_token = None
                        avg_n_decoder_token = None
                    assert (
                        request.model1 is not None
                    ), f"Request {request} has no model1"
                    assert (
                        request.model2 is not None
                    ), f"Request {request} has no model2"
                    annotations.append(
                        JudgeAnnotation(
                            preference=preference,
                            cost=avg_cost,
                            time=avg_time,
                            swap=swap,
                            prompt=(
                                self.system_prompt + "\n" + prompt_with_shift
                                if swap
                                else self.system_prompt + "\n" + prompt_without_shift
                            ),
                            instruction_index=request.instruction_index,
                            output1=request.output1,
                            output2=request.output2,
                            model1=request.model1,
                            model2=request.model2,
                            judge_completion=judge_completion,
                            n_prompt_token=avg_n_prompt_token,
                            n_token_decoder=avg_n_decoder_token,
                        )
                    )
            all_annotations.append(annotations)
        return all_annotations

    @classmethod
    def from_json(cls, json_str: str):
        json_dict = json.loads(json_str)
        prompt_type_kwargs = json_dict["judge_kwargs"].pop("prompt_type_kwargs")
        json_dict["judge_kwargs"]["completion_client"] = completion_client_from_json(
            json_dict["judge_kwargs"]["completion_client"]
        )
        return cls(
            prompt_type=PromptType(**prompt_type_kwargs), **json_dict["judge_kwargs"]
        )

    def to_json(self) -> str:
        kwargs = dict(self.__dict__)
        kwargs["completion_client"] = kwargs["completion_client"].to_json()
        prompt_type = kwargs.pop("prompt_type")
        kwargs["prompt_type_kwargs"] = dict(**prompt_type.__dict__)
        kwargs["prompt_type_kwargs"].pop("system_prompt")
        allowed_keys = [
            "completion_client",
            "n_sample_with_shift",
            "n_sample_without_shift",
            "temperature_model",
            "prompt_type_kwargs",
        ]
        kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}

        json_repr = {
            "judge_cls": "JudgeOptions",
            "judge_kwargs": kwargs,
        }
        return json.dumps(json_repr)
