from dataclasses import dataclass
from typing import List


@dataclass
class JudgeEvaluationRow:
    # Schema that encapsulates desired information of a judge evaluation to be put in a dataframe
    output1: str  # completion of the first model
    model1: str  # name of the first model
    output2: str  # completion of the second model
    model2: str  # name of the second model
    # preference in [1, 2] where 1 means a strong preference for model 1 and 2 a preference for model 2
    preference: float
    confidence: float  # confidence indicated by the model
    # whether the models order were swapped before querying the LLM judge, if false then Model A and B were resp. model
    # 1 and 2 whereas if true, then Model A and B were respectively 2 and 1, preference is adjusted so that a preference
    # of 1 always means that the first model was judged better
    swap: bool
    explanation: (
        str | None
    )  # an explanation provided by the judge which can be useful for analysis purpose
    cost: float
    time: float

    def __str__(self, max_length: int = 100):
        swaped = " (swaped)" if self.swap else ""
        return (
            f"Preference: {self.preference:.2f}, "
            # f"ScoreModel1: {self.score1:.2f}, "
            # f"ScoreModel2: {self.score2:.2f}, "
            f"confidence: {self.confidence}, "
            f"explanation{swaped}: {self.explanation[:max_length]}"
        )


@dataclass
class AnnotationRequest:
    instruction_index: str
    instruction: str
    output1: str
    output2: str
    model1: str | None = None
    model2: str | None = None


@dataclass
class JudgeAnnotation:
    preference: (
        float  # preference in [0, 1] where 0 means a strong preference for model 1
    )
    instruction_index: str | None
    output1: str | None
    output2: str | None
    model1: str | None = None
    model2: str | None = None
    prompt: str | None = None
    judge_completion: str | None = None
    # whether the models order were swapped before querying the LLM judge, if false then Model A and B were resp. model
    # 1 and 2 whereas if true, then Model A and B were respectively 2 and 1, preference is adjusted so that a preference
    # of 0 always means that the first model was judged better
    swap: bool = False
    cost: float = 0.0
    time: float = 0.0
    n_prompt_token: int | None = None
    n_token_decoder: int | None = None

    def __str__(self, max_length: int = 100) -> str:
        return f"Preference: {self.preference:.2f}"


class Judge:
    def annotate(
        self, requests: list[AnnotationRequest]
    ) -> list[list[JudgeAnnotation]]:
        """
        :return:
        the judge is allowed to return multiple annotation for the same example as it may use several seeds or randomize
        the order assignment, downstream code takes the average of performance and other metrics.
        """
        annotations = self._annotate(requests)
        assert len(annotations) == len(requests), (
            "Number of annotations does not match the number of requests: "
            + str(len(annotations))
            + " != "
            + str(len(requests))
        )
        return annotations

    def _annotate(
        self, requests: list[AnnotationRequest]
    ) -> list[list[JudgeAnnotation]]:
        raise NotImplementedError()

    @classmethod
    def from_json(cls, json_str: str):
        raise NotImplementedError()

    def to_json(self):
        raise NotImplementedError()
