from typing import List

from judgetuning.judge import Judge, JudgeAnnotation, AnnotationRequest


class JudgeMockup(Judge):
    def _annotate(
        self, requests: list[AnnotationRequest]
    ) -> list[list[JudgeAnnotation]]:
        return [
            [
                JudgeAnnotation(
                    preference=0.5,
                    **{k: v for k, v in request.__dict__.items() if k != "instruction"}
                )
            ]
            for request in requests
        ]
