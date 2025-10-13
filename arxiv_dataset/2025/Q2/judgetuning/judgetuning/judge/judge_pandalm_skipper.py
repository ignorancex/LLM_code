from time import time
from typing import List

from tqdm import tqdm

from judgetuning.judge import Judge, JudgeAnnotation, AnnotationRequest
import json

from judgetuning.llm_client.llm_specs import name_to_llm_spec


class PandaLM7BSkipper(Judge):
    def __init__(
        self,
        # TODO add support for 70B
        model_path: str = "WeOpenML/PandaLM-7B-v1",
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 1,
        num_beams: int = 4,
        max_new_tokens: int = 512,
        repetition_penalty: float = 1.2,
        device_map: str = "auto",
        max_len_prompt: int = 8192,
    ):
        from judgetuning.judge.pandalm import PandaLMBatchInferenceProviderModified

        self.handler = PandaLMBatchInferenceProviderModified(
            model_path=model_path,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            device_map=device_map,
        )
        self.spec = name_to_llm_spec["WeOpenML/PandaLM-7B-v1"]
        self.max_len_prompt = max_len_prompt
        print("Using skipper PandaLM")
        print(f"max_len_prompt: {self.max_len_prompt}")

    # def annotate(
    #     self, requests: list[AnnotationRequest], human_preferences: list[float]
    # ) -> list[list[JudgeAnnotation]]:
    #     """
    #     :return:
    #     the judge is allowed to return multiple annotation for the same example as it may use several seeds or randomize
    #     the order assignment, downstream code takes the average of performance and other metrics.
    #     """
    #     annotations = self._annotate(requests, human_preferences)
    #     assert len(annotations) == len(requests), (
    #         "Number of annotations does not match the number of requests: "
    #         + str(len(annotations))
    #         + " != "
    #         + str(len(requests))
    #     )
    #     return annotations

    def _annotate(
        self,
        annotationRequests: list[AnnotationRequest],  # human_preferences: List[float]
    ) -> list[list[JudgeAnnotation]]:
        import torch
        list_annotations = []
        j = 0
        print(f"len(annotationRequests): {len(annotationRequests)}")
        print(
            f"annotationRequests[0].instruction_index: {annotationRequests[0].instruction_index}"
        )
        # print(f"len(human_preferences): {len(human_preferences)}")
        for request in tqdm(annotationRequests):
            # print(f"request.instruction_index: {request.instruction_index}")
            # print(f"human_preferences[{j}]: {human_preferences[j]}")
            # j = j + 1
            annotation_pairs = []
            for i, annotationRequest in enumerate([request, request]):
                swap = bool(i)
                start_time = time()

                # In the case of no-balancing
                # generated = self.handler.infer_request(
                #     instruction=annotationRequest.instruction,
                #     input="",
                #     response1=(
                #         annotationRequest.output1
                #         if not swap
                #         else annotationRequest.output2
                #     ),
                #     response2=(
                #         annotationRequest.output2
                #         if not swap
                #         else annotationRequest.output1
                #     ),
                # )
                # chosen = self.handler.parse_pandalm_response(generated)

                input_ids, is_truncated = self.handler.balanced_preprocess_input(
                    instruction=annotationRequest.instruction,
                    input="",
                    resp1=(
                        annotationRequest.output1
                        if not swap
                        else annotationRequest.output2
                    ),
                    resp2=(
                        annotationRequest.output2
                        if not swap
                        else annotationRequest.output1
                    ),
                    max_len_prompt=self.max_len_prompt,
                    skip_flag=True,
                )
                # print(f"input_ids.shape: {input_ids.shape}")

                if is_truncated:
                    chosen = 3
                    generated = "Input too long"
                    del input_ids
                    torch.cuda.empty_cache()
                else:
                    output_ids = self.handler.generate_response(input_ids)
                    generated = self.handler.text_from_sequence(output_ids)
                    chosen = self.handler.parse_pandalm_response(generated)
                    price = (
                        len(input_ids[0]) * self.spec.cost_prompt
                        + len(output_ids) * self.spec.cost_completion
                    ) * 1e-6

                    del (input_ids, output_ids)
                    torch.cuda.empty_cache()
                # print(f"chosen: {chosen}",
                #       f"generated[:1]: {generated[:1]}")
                if chosen == 1:
                    preference = 0.0
                elif chosen == 2:
                    preference = 1.0
                else:
                    preference = 0.5

                # time_taken = time() - start_time

                judge_annotation = JudgeAnnotation(
                    preference=preference if not swap else 1 - preference,
                    instruction_index=annotationRequest.instruction_index,
                    output1=annotationRequest.output1,
                    output2=annotationRequest.output2,
                    model1=annotationRequest.model1,
                    model2=annotationRequest.model2,
                    prompt=annotationRequest.instruction,
                    judge_completion=generated,
                    swap=bool(i),
                    cost=0,
                    time=time() - start_time,
                )

                annotation_pairs.append(judge_annotation)
            list_annotations.append(annotation_pairs)

        return list_annotations

    def to_json(self):
        json_repr = {
            "judge_cls": "PandaLM-7B",
            "judge_kwargs": {
                "device_map": self.handler.device_map,
                "temperature": self.handler.temperature,
                "top_p": self.handler.top_p,
                "top_k": self.handler.top_k,
                "num_beams": self.handler.num_beams,
                "max_new_tokens": self.handler.max_new_tokens,
                "repetition_penalty": self.handler.repetition_penalty,
            },
        }
        return json.dumps(json_repr)


if __name__ == "__main__":
    judge = PandaLM7BSkipper()
    print(
        judge.annotate(
            [
                AnnotationRequest(
                    instruction_index="1",
                    instruction="Complete the following sequence. Just output a single number. 1,2,3,",
                    output1="4",
                    output2="5",
                    model1="gpt-4-0125-preview",
                    model2="gpt-4-1106-preview",
                )
            ]
        )
    )
