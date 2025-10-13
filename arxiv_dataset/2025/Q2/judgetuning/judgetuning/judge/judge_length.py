import json
from typing import List

from judgetuning.judge import Judge, JudgeAnnotation, AnnotationRequest


class JudgeLength(Judge):
    def __str__(self):
        return "JudgeLength"

    def _annotate(
        self, requests: list[AnnotationRequest]
    ) -> list[list[JudgeAnnotation]]:
        return [
            [
                JudgeAnnotation(
                    preference=1 - float(len(request.output1) > len(request.output2)),
                    swap=False,
                    judge_completion="NA",
                    **{k: v for k, v in request.__dict__.items() if k != "instruction"}
                )
            ]
            for request in requests
        ]

    def to_json(self):
        json_repr = {
            "judge_cls": "JudgeLength",
            "judge_kwargs": {},
        }
        return json.dumps(json_repr)

    @classmethod
    def from_json(cls, json_str: str):
        return cls()
