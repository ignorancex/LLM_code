import json
import logging
import os
import pprint
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from judgetuning.annotation_dataset import (
    AnnotationDataset,
)
from judgetuning.annotation_dataset import ArenaHardDataset, AlpacaEvalDataset
from judgetuning.experiment_tracker import ExperimentTracker
from judgetuning.join import common_models
from judgetuning.judge.eval_judge import eval_judge
from judgetuning.judge.generate_judge_annotations import generate_judge_annotations
from judgetuning.judge_paths import path_config_judge, path_result_judge, path_judge
from judgetuning.script.utils import make_judge, parse_args

np.random.seed(0)
random.seed(0)


def get_models(annotation_dataset):
    # gets a list of models where the first 27 models are common models between alpaca-eval and arena-hard
    # which are shuffled followed by the remaining models specific of the dataset, also shuffled
    _common_models = common_models()
    _distinct_models = sorted(
        list(set(annotation_dataset.models).difference(_common_models))
    )
    random.shuffle(_common_models)
    random.shuffle(_distinct_models)
    return _common_models + _distinct_models


def most_correlated_instructions(
    df_preference,
    elo_chatbot_arena,
    n_seeds: int | None = 20,
):
    prompt_corr_series = []
    prompts = df_preference.index
    if n_seeds is None:
        dd = df_preference.loc[:, elo_chatbot_arena.index]
        prompts_corr = {}
        for prompt in prompts:
            prompts_corr[prompt] = spearmanr(
                elo_chatbot_arena.loc[dd.columns], dd.loc[[prompt]].values.reshape(-1)
            ).statistic
        prompt_corr_series.append(pd.Series(prompts_corr))
    else:
        for _ in range(n_seeds):
            dd = df_preference.loc[:, elo_chatbot_arena.index].sample(
                axis=1, frac=1.0, replace=True
            )
            # dd = df_preference.loc[:, elo_chatbot_arena.index].sample(
            #     axis=1, n=20, replace=False
            # )
            prompts_corr = {}
            for prompt in prompts:
                prompts_corr[prompt] = spearmanr(
                    elo_chatbot_arena.loc[dd.columns],
                    dd.loc[[prompt]].values.reshape(-1),
                ).statistic
            prompt_corr_series.append(pd.Series(prompts_corr))
    df_prompt_corrs = pd.DataFrame(prompt_corr_series)

    prompt_corr_series = df_prompt_corrs.mean(axis=0).sort_values(ascending=False)
    return prompt_corr_series.index.tolist()

def load_sorted_instruction(
    annotation_dataset: AnnotationDataset, models: list[str]
) -> list[str]:
    # return annotation_dataset.instructions
    return annotation_dataset.instruction_index
    df_preference = annotation_dataset.df_winrate_against_baseline(models=models)
    test_models = common_models()
    val_models = [
        m for m in models if m not in test_models and m in df_preference.columns
    ]
    # not sure why this happens
    val_models = [m for m in models if m in df_preference.columns]

    return most_correlated_instructions(
        df_preference.loc[:, val_models],
        annotation_dataset.chatbot_arena_elo_ratings().loc[val_models],
    )


def generate_sorted_instruction_and_models(annotation_dataset, model_instruction_path):
    models = get_models(annotation_dataset)
    instructions = load_sorted_instruction(annotation_dataset, models)

    with open(model_instruction_path, "w") as f:
        f.write(json.dumps(dict(instructions=instructions, models=models)))


def load_instruction_and_models(annotation_dataset) -> tuple[list[str], list[str]]:
    model_instruction_path = (
        Path(__file__).parent
        / "data"
        / f"models-instructions-{annotation_dataset.name}.json"
    )

    if not model_instruction_path.exists():
        logging.warning(
            f"files containing sorted models or instructions for {annotation_dataset.name} not found, regenerating it"
        )
        generate_sorted_instruction_and_models(
            annotation_dataset, model_instruction_path
        )

    with open(model_instruction_path, "r") as f:
        model_instructions = json.load(f)

    return model_instructions["instructions"], model_instructions["models"]

def main():
    np.random.seed(0)
    random.seed(0)
    args = parse_args()

    print(f"Evaluation args: {args}")
    try:
        import torch

        print(f"GPU memory: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB")
    except Exception as e:
        print(e)
    jobname = os.getenv("SP_JOBNAME", "")
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        jobname += "-" + os.getenv("SLURM_ARRAY_TASK_ID")

    n_models = args.n_models
    n_instructions = args.n_instructions
    expid = args.expid

    match args.dataset:
        case "arena-hard":
            annotation_dataset = ArenaHardDataset(
                keep_only_chatbot_arena=True, rename_chatbot_arena=True
            )
            baseline_model = "gpt-4-0314"
        case "alpaca-eval":
            annotation_dataset = AlpacaEvalDataset(
                keep_only_chatbot_arena=True, rename_chatbot_arena=True
            )
            baseline_model = "gpt-4-0314"
        case _:
            raise ValueError(f"Invalid dataset: {args.dataset}")
    print(f"Loaded dataset: {annotation_dataset.name}.")
    instructions, models = load_instruction_and_models(annotation_dataset)
    if args.dataset == "arena-hard":
        # use model available for claude and GPT4 annotation in arena-hard
        models = [
            "gpt-4-turbo-2024-04-09",
            "gpt-4-0125-preview",
            "claude-2.1",
            "llama-2-70b-chat",
            "gpt-3.5-turbo-1106",
            "gemma-2b-it",
            "yi-34b-chat",
            "dbrx-instruct-preview",
            "qwen1.5-72b-chat",
            "gpt-4-1106-preview",
            "gemma-7b-it",
            "claude-2.0",
            "mistral-medium",
            "claude-3-sonnet-20240229",
            "mixtral-8x7b-instruct-v0.1",
            "gemini-pro",
            "tulu-2-dpo-70b",
            "gpt-4-0613",
            "gpt-3.5-turbo-0613",
            "mistral-large-2402",
        ]

    if baseline_model in models:
        # no need to judge the baseline with itself
        models.remove(baseline_model)

    judge = make_judge(args)
    print(f"Generating annotations for judge {judge}")

    subset_instructions = instructions[:n_instructions]
    subset_models = models[:n_models]

    # judge_name = "_".join([str(x).replace("/", "-") for x in args.__dict__.values()])
    print(
        f"Going to evaluate {n_instructions} instructions on {n_models} models for job {jobname} ({judge}).\n"
    )
    print(f"Models considered: {subset_models}")

    experiment_tracker = ExperimentTracker(
        judge=judge,
        instructions=subset_instructions,
        instruction_dataset=args.dataset,
        models=subset_models,
        jobname=jobname,
        tags=[expid],
    )
    # save judge configuration
    path_config_judge(expid=expid, judge_name=jobname).parent.mkdir(
        parents=True, exist_ok=True
    )
    with open(path_config_judge(expid=expid, judge_name=jobname), "w") as f:
        f.write(judge.to_json())
    with open(path_judge(expid=expid, judge_name=jobname) / "args.json", "w") as f:
        f.write(json.dumps(args.__dict__))

    generate_judge_annotations(
        annotation_dataset=annotation_dataset,
        expid=expid,
        judge=judge,
        judge_name=jobname,
        instructions_index=subset_instructions,
        models=subset_models,
        experiment_tracker=experiment_tracker,
        baseline_model=baseline_model,
    )
    metrics = eval_judge(
        expid=expid,
        judge_name=jobname,
        annotation_dataset=annotation_dataset,
    )

    print(f"Results for {judge}:")
    pprint.pprint(metrics, indent=2)

    # save results into json file
    with open(path_result_judge(expid=expid, judge_name=jobname), "w") as f:
        f.write(json.dumps(metrics.__dict__))

    experiment_tracker.track_results(metrics)


if __name__ == "__main__":
    main()
