from time import time
from typing import List

from tqdm import tqdm

from judgetuning.judge import Judge, JudgeAnnotation, AnnotationRequest
import json
from judgetuning.llm_client.llm_specs import name_to_llm_spec


class JudgeLM7B(Judge):

    def __init__(
        self,
        device: str,
        load_8bit: bool = False,
        cpu_offloading: bool = False,
        temperature: float = 0.2,
        max_new_token: int = 500,
    ):
        self.device = device
        self.load_8bit = load_8bit
        self.cpu_offloading = cpu_offloading
        self.temperature = temperature
        self.max_new_token = max_new_token
        self.do_sample = False if temperature < 1e-4 else True
        self.spec = name_to_llm_spec["BAAI/JudgeLM-7B-v1.0"]

    def _annotate(
        self, annotationRequests: list[AnnotationRequest]
    ) -> list[list[JudgeAnnotation]]:
        import torch
        from judgelm.llm_judge.common import conv_judge_pair, KeywordsStoppingCriteria
        from judgelm.model import load_model

        model, tokenizer = load_model(
            "BAAI/JudgeLM-7B-v1.0",
            device=self.device,
            num_gpus=1,
            load_8bit=False,
            cpu_offloading=False,
            debug=False,
        )
        list_annotations = []
        for request in tqdm(annotationRequests):
            annotation_pairs = []
            for i, annotationRequest in enumerate([request, request]):
                swap = bool(i)
                start_time = time()
                conv = conv_judge_pair.copy(None)
                template = conv.prompt_template
                data_sample = (
                    conv.system
                    + "\n"
                    + template.format(
                        question=annotationRequest.instruction,
                        answer_1=(
                            annotationRequest.output1
                            if not swap
                            else annotationRequest.output2
                        ),
                        answer_2=(
                            annotationRequest.output2
                            if not swap
                            else annotationRequest.output1
                        ),
                        prompt=conv.prompt,
                    )
                    + conv.appendix
                )

                input_ids = tokenizer([data_sample]).input_ids
                input_ids[0][0] = 1
                if len(input_ids[0]) > 8192:
                    preference = 0.5
                    time_taken = 0
                    price = 0
                    output = "Input too long"
                else:
                    stopping_criteria = KeywordsStoppingCriteria(
                        [conv.sep], tokenizer, torch.as_tensor(input_ids)
                    )

                    # generate judgements
                    output_ids = model.generate(
                        (
                            torch.as_tensor(input_ids)
                            if self.device == "cpu"
                            else torch.as_tensor(input_ids).cuda()
                        ),
                        do_sample=self.do_sample,
                        temperature=self.temperature,
                        max_new_tokens=self.max_new_token,
                        stopping_criteria=[stopping_criteria],
                    )

                    if model.config.is_encoder_decoder:
                        output_ids = output_ids[0]
                    else:
                        output_ids = output_ids[0][len(input_ids[0]) :]

                    output = tokenizer.decode(
                        output_ids,
                        skip_special_tokens=True,
                        spaces_between_special_tokens=False,
                    )

                    if conv.sep:
                        output = output[: output.find(conv.sep)]
                    output = output.strip()

                    # possible output: "8 4\nThe first model rocks."
                    try:
                        score0, score1 = output.split("\n")[0].split(" ")
                        preference = 1 - float(float(score0) > float(score1))
                    except Exception as e:
                        print(f"Error is {e}")
                        print(
                            f"request_index: {annotationRequest.instruction_index}, model2:{annotationRequest.model2}"
                        )
                        print(f"output: {output}")
                        print("===================================")
                        print("===================================")
                        preference = 0.5

                    time_taken = time() - start_time

                    price = (
                        len(input_ids[0]) * self.spec.cost_prompt
                        + len(output_ids) * self.spec.cost_completion
                    ) * 1e-6

                judge_annotation = JudgeAnnotation(
                    preference=preference if not swap else 1 - preference,
                    instruction_index=annotationRequest.instruction_index,
                    output1=annotationRequest.output1,
                    output2=annotationRequest.output2,
                    model1=annotationRequest.model1,
                    model2=annotationRequest.model2,
                    prompt=data_sample,
                    judge_completion=output,  # "\n".join(output.split("\n")[1:]),
                    swap=bool(i),
                    cost=price,
                    time=time_taken,
                )

                annotation_pairs.append(judge_annotation)
            list_annotations.append(annotation_pairs)

        return list_annotations

    def to_json(self):
        json_repr = {
            "judge_cls": "Judge-LM-7B",
            "judge_kwargs": {
                "device": self.device,
                "load_8bit": self.load_8bit,
                "cpu_offloading": self.cpu_offloading,
                "temperature": self.temperature,
                "max_new_token": self.max_new_token,
                "do_sample": self.do_sample,
            },
        }
        return json.dumps(json_repr)


if __name__ == "__main__":
    judge = JudgeLM7B(device="cuda", load_8bit=False, cpu_offloading=False)
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
