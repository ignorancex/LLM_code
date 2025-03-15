import os
import sys
import importlib
import warnings

import torch
from pprint import pprint
from tqdm import tqdm
import hydra
import numpy as np
from dotenv import load_dotenv
from omegaconf import DictConfig
from src.myutils.file import write_json, load_json
sys.path.insert(0, os.path.abspath("./vllm"))

from vllm import LLM, SamplingParams



def warpped_sampling(prompts, llm, sampling_params, enable_test_speed, max_prompt_len):
    if enable_test_speed:
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
    if max_prompt_len > 0:
        tokenizer = llm.get_tokenizer()

    results = {
        "generated": [],
        "total_time": [],
        "num_tokens": [],
        "num_tokens_per_sec": [],
    }

    for prompt in tqdm(prompts, desc="Processing prompts"):
        if max_prompt_len > 0:
            assert (
                max_prompt_len
                < llm.llm_engine.model_config.max_model_len - sampling_params.max_tokens
            ), "max_prompt_len should be less than max_model_len - max_tokens"
            prompt_token_ids = tokenizer.encode(prompt, return_tensors="pt")[
                :, -max_prompt_len:
            ]
            prompt = tokenizer.decode(prompt_token_ids[0], skip_special_tokens=True)
        if enable_test_speed:
            starter.record()
            output = llm.generate(prompt, sampling_params, use_tqdm=False)
            ender.record()
            torch.cuda.synchronize()

            results["num_tokens"].append(len(output[0].outputs[0].token_ids))
            results["total_time"].append(starter.elapsed_time(ender) / 1000)
            results["num_tokens_per_sec"].append(
                results["num_tokens"][-1] / results["total_time"][-1]
            )
        else:
            output = llm.generate(prompt, sampling_params, use_tqdm=False)
            results["num_tokens"].append(len(output[0].outputs[0].token_ids))

        results["generated"].append(output[0].outputs[0].text)

    return results


def warm_up(llm):
    print("warm up...")
    text = "Alice and Bob"
    sampling_params = SamplingParams(max_tokens=1)
    for _ in range(10):
        llm.generate(text, sampling_params, use_tqdm=False)
    torch.cuda.synchronize()


def init_dataset(args):
    MyDataset = importlib.import_module(
        f"src.mydatasets.{args.dataset.name}.mydataset"
    ).MyDataset
    dataset = MyDataset(size=args.dataset.size, use_fewshot=args.dataset.use_fewshot)
    return dataset


def preprocess_cfg(cfg):
    # preprocess max_model_len
    if cfg.method.llm.max_model_len is None:
        cfg.method.llm.max_model_len = 1e6

        def get_max_model_len(model_name):
            model_cfg = load_json(
                f"{os.environ['MODEL_PATH']}/{model_name}/config.json"
            )
            return model_cfg["max_position_embeddings"]

        keys = ["model", "amateur_model", "draft_model"]
        for key in keys:
            if key in cfg.method:
                cfg.method.llm.max_model_len = min(
                    cfg.method.llm.max_model_len, get_max_model_len(cfg.method[key])
                )

    if "extra_model" in cfg.method and cfg.method.extra_model is not None:
        if isinstance(cfg.method.extra_model, str):
            extra_models = [cfg.method.extra_model]
        else:
            extra_models = cfg.method.extra_model
        if len(extra_models) > 1:
            warnings.warn(
                "More than one models are provided, we will override ensemble_fn to (...) / num_total_models"
            )
            if cfg.method.llm.ensemble_target in ["logits", "raw-logits"]:
                cfg.method.llm.ensemble_fn = (
                    f"${{eval:'lambda logits: sum(logits) / {len(extra_models) + 1}'}}"
                )
            elif cfg.method.llm.ensemble_target in ["probs"]:
                cfg.method.llm.ensemble_fn = f"${{eval:'lambda probs, logprobs: (sum(probs) / {len(extra_models) + 1}, logprobs[0])'}}"
            else:
                raise ValueError(
                    f"ensemble_target {cfg.method.llm.ensemble_target} not supported"
                )


def process_result(cfg, results, evaluate_func):
    if cfg.test_speed:
        results_stats = {
            "performance": evaluate_func(results["generated"]),
            "num_tokens_per_sec": np.mean(results["num_tokens_per_sec"]),
            "total_time": np.mean(results["total_time"]),
            "num_tokens": np.mean(results["num_tokens"]),
        }
    else:
        results_stats = {
            "performance": evaluate_func(results["generated"]),
            "num_tokens": np.mean(results["num_tokens"]),
        }

    pprint(results_stats)

    if cfg.save_path:
        write_json(cfg.save_path, results)


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):

    preprocess_cfg(cfg)
    torch.manual_seed(cfg.seed)

    dataset = init_dataset(cfg)
    prompts = dataset.get_prompts()

    llm = LLM(**cfg.method.llm)
    sampling_params = SamplingParams(seed=cfg.seed, **cfg.method.generate)

    results = warpped_sampling(
        prompts,
        llm=llm,
        sampling_params=sampling_params,
        enable_test_speed=cfg.test_speed,
        max_prompt_len=cfg.max_prompt_len,
    )

    process_result(cfg, results, dataset.evaluate)


if __name__ == "__main__":
    load_dotenv(override=True)
    main()
