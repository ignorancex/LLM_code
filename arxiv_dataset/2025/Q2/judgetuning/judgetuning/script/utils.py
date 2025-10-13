import argparse
import ast
from dataclasses import dataclass

from judgetuning.judge.io.judge_alpaca_eval import JudgeAlpacaEval
from judgetuning.judge.io.judge_arena_hard import JudgeArenaHard
from judgetuning.judge.io.prompt_utils import PromptType
from judgetuning.judge.judge_length import JudgeLength
from judgetuning.judge.judge_lm import JudgeLM7B
from judgetuning.judge.judge_pandalm import PandaLM7B
from judgetuning.judge.judge_pandalm_skipper import PandaLM7BSkipper
from judgetuning.judge.judge_with_options.judge_options import JudgeOptions
from judgetuning.llm_client import (
    OllamaCompletionClient,
    create_client,
)


@dataclass
class EvaluateFidelityArgs:
    model: str
    expid: str = "dummy"
    judge_class: str = "judge-option"
    tensor_parallel_size: int = 1
    dataset: str = "arena-hard"
    split: str = "val"
    max_pred_len: int = 4096
    n_models: int = 2
    n_instructions: int | None = None
    provide_confidence: bool = False
    provide_explanation: bool = False
    provide_example: bool = False
    provide_answer: bool = False
    json_output: bool = True
    score_type: str = "pair"
    temperature: float = 0.0
    n_sample_with_shift: int = 1
    n_sample_without_shift: int = 1
    min_criterion: int = 5  # for lmsys
    dtype: str | None = None
    # bit8: bool = False
    # cpu_offloading: bool = False
    # max_new_token: int = 500
    # sample: bool = True
    max_len_prompt: int = 2048
    # partition_test: int = 0


def parse_args() -> EvaluateFidelityArgs:
    parser = argparse.ArgumentParser(
        prog="Evaluate judge",
    )
    parser.add_argument(
        "--expid",
        help="the experiment id, used to retrieve files later which will be saved under ~/judge-tuning-data/experiments/{expid})",
        required=True,
    )
    parser.add_argument(
        "--model",
        help="the model that is served in vllm (--model)",
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
    )
    parser.add_argument(
        "--judge_class",
        default="judge-option",
        choices=[
            "judge-option",
            "judge-arena-hard",
            "judge-alpaca-eval",
            "judge-length",
            "judge-lm7b",
            "judge-pandalm",
            # "judge-pandalm-single",
            # "judge-pandalm-original",
            "judge-pandalm-skipper",
        ],
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--dataset",
        default="arena-hard",
        choices=["alpaca-eval", "arena-hard", "lmsys", "pandalm", "llmbar"],
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["val", "test"],
    )
    parser.add_argument(
        "--max_pred_len",
        help="the maximum length",
        default=4096,
        type=int,
    )
    parser.add_argument(
        "--n_models",
        default=2,
        type=int,
    )
    parser.add_argument(
        "--n_instructions",
        type=int,
    )
    parser.add_argument(
        "--min_criterion",
        default=5,
        type=int,
        help="minimum instruction quality to consider when using lmsys",
    )
    parser.add_argument(
        "--provide_confidence",
        default=0,
        type=ast.literal_eval,
    )
    parser.add_argument(
        "--provide_explanation",
        default=0,
        type=ast.literal_eval,
    )
    parser.add_argument(
        "--provide_example",
        default=0,
        type=ast.literal_eval,
    )
    parser.add_argument(
        "--provide_answer",
        default=0,
        type=ast.literal_eval,
    )
    parser.add_argument(
        "--json_output",
        default=0,
        type=ast.literal_eval,
    )
    parser.add_argument(
        "--temperature",
        default=0,
        type=float,
    )
    parser.add_argument(
        "--score_type",
        default="pair",
        choices=["preference", "likert", "pair", "best-model-identifier", "multi"],
        type=str,
    )
    parser.add_argument(
        "--n_sample_with_shift",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--n_sample_without_shift",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--dtype",
        type=str,
    )
    parser.add_argument(
        "--max_len_prompt",
        type=int,
        default=2048,
    )

    args = parser.parse_args()
    return EvaluateFidelityArgs(**args.__dict__)


def make_judge(args: EvaluateFidelityArgs):
    match args.judge_class:
        case "judge-option":
            completion_client = make_completion_client(args)
            judge = JudgeOptions(
                completion_client=completion_client,
                prompt_type=PromptType(
                    score_type=args.score_type,
                    provide_confidence=bool(args.provide_confidence),
                    provide_explanation=bool(args.provide_explanation),
                    provide_example=bool(args.provide_example),
                    provide_answer=bool(args.provide_answer),
                    json_output=bool(args.json_output),
                    prompt_filename="request_prompt_ours",
                ),
                n_sample_with_shift=args.n_sample_with_shift,
                n_sample_without_shift=args.n_sample_without_shift,
                temperature_model=args.temperature,
            )
        case "judge-arena-hard":
            completion_client = make_completion_client(args)
            judge = JudgeArenaHard(
                completion_client=completion_client,
            )
        case "judge-alpaca-eval":
            completion_client = make_completion_client(args)
            judge = JudgeAlpacaEval(
                completion_client=completion_client,
            )
        case "judge-length":
            judge = JudgeLength()
        case "judge-lm7b":
            import torch
            judge = JudgeLM7B(
                device="cuda" if torch.cuda.is_available() else "cpu",
                # load_8bit=bool(args.bit8),
                # cpu_offloading=bool(args.cpu_offloading),
                # temperature=args.temperature,
                # max_new_token=args.max_pred_len,
            )
        case "judge-pandalm":
            judge = PandaLM7B(
                # temperature=args.temperature,
                # max_new_tokens=args.max_pred_len,
                max_len_prompt=args.max_len_prompt,
            )
        # case "judge-pandalm-single":
        #     judge = PandaLM7BSing(
        #         # temperature=args.temperature,
        #         # max_new_tokens=args.max_pred_len,
        #     )
        # case "judge-pandalm-original":
        #     judge = PandaLM7BOrig(
        #         # temperature=args.temperature,
        #         # max_new_tokens=args.max_pred_len,
        #     )
        case "judge-pandalm-skipper":
            judge = PandaLM7BSkipper(
                # temperature=args.temperature,
                # max_new_tokens=args.max_pred_len,
                max_len_prompt=args.max_len_prompt,
            )
    return judge


def make_completion_client(args: EvaluateFidelityArgs):
    try:
        completion_client = create_client(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            max_pred_len=args.max_pred_len,
            dtype=args.dtype,
        )
    except ImportError as e:
        print(str(e))
        args.model = "Ollama"
        print(
            "cannot import vllm client, assuming you are running on a machine without GPU and using Ollama"
        )
        completion_client = OllamaCompletionClient()
    return completion_client

