from pathlib import Path
from typing import List

from judgetuning.judge import JudgeAnnotation, AnnotationRequest
from judgetuning.judge.io.judge_score import LikertScore
from judgetuning.judge.io.prompt_utils import PromptType
from judgetuning.judge.judge_with_options.judge_options import JudgeOptions
from judgetuning.llm_client import CompletionClient
from judgetuning.utils import read_and_format


class JudgeArenaHard(JudgeOptions):
    def __init__(
        self,
        completion_client: CompletionClient,
        n_sample_with_shift: int = 1,
        n_sample_without_shift: int = 1,
        temperature_model: float = 0.0,
    ):
        self.completion_client = completion_client
        self.n_sample_with_shift: int = n_sample_with_shift
        self.n_sample_without_shift: int = n_sample_without_shift
        self.temperature_model = temperature_model
        self.__post_init__()

    def __post_init__(self):
        with open(
            Path(__file__).parent / "prompts" / "system-prompt-arena-hard.txt"
        ) as f:
            self._system_prompt = f.read()
        self.request_template_filename = (
            Path(__file__).parent / "prompts" / "request_prompt_arena_hard.txt"
        )
        self.prompt_type = PromptType(
            provide_answer=False,
            provide_explanation=False,
            provide_confidence=False,
            provide_example=False,
            json_output=False,
            score_type="likert",
            system_prompt_name="arena-hard",
        )

    def __str__(self):
        return f"JudgeArenaHard(completion_client={self.completion_client},n_sample_with_shift={self.n_sample_with_shift},n_sample_without_shift={self.n_sample_with_shift})"

    def generate_request_prompt(
        self, instruction: str, output1: str, output2: str
    ) -> str:
        return read_and_format(
            filename=self.request_template_filename,
            instruction=instruction,
            output1=output1,
            output2=output2,
        )


if __name__ == "__main__":
    from judgetuning.llm_client import OllamaCompletionClient

    completion_client = OllamaCompletionClient()
    judge = JudgeArenaHard(
        completion_client=completion_client,
        n_sample_with_shift=1,
        n_sample_without_shift=1,
    )
    annotations = judge.annotate(
        requests=[
            AnnotationRequest(
                instruction_index="Give me two countries that starts with S.",
                output1="Spain, Sweden.",
                output2="No way I would answer to this question!",
            )
        ]
    )[0]

    preferences = [x.preference for x in annotations]
    for annotation, p in zip(annotations, preferences):
        if p >= 0.5:
            print(f"!!!!Unexpected value {p} for judge {judge}")
        print(f"***prompt:\n{annotation.prompt}")
        print(f"***completion:\n{annotation.judge_completion}")
        print(f"**preference:\n{annotation.preference}")
