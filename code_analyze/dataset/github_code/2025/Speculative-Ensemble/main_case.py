import pdb
import os

import torch
from pprint import pprint
from tqdm import tqdm
import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig
from vllm import LLM, SamplingParams


def warpped_sampling(prompt, llm, sampling_params):

    # warm_up(llm)  # warm up (to avoid the first inference time being too long)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )

    results = {}

    starter.record()
    output = llm.generate(prompt, sampling_params, use_tqdm=False)
    ender.record()
    torch.cuda.synchronize()

    results["generated"] = output[0].outputs[0].text
    results["total_time"] = starter.elapsed_time(ender) / 1000
    results["num_tokens"] = len(output[0].outputs[0].token_ids)
    results["num_tokens_per_sec"] = results["num_tokens"] / results["total_time"]

    return results


def warpped_sampling_wo_speed_test(prompts, llm, sampling_params):

    results = {
        "generated": [],
        "total_time": [],
        "num_tokens": [],
        "num_tokens_per_sec": [],
    }

    for prompt in tqdm(prompts, desc="Processing prompts"):
        output = llm.generate(prompt, sampling_params, use_tqdm=False)
        results["generated"].append(output[0].outputs[0].text)
        results["total_time"].append(-1)
        results["num_tokens"].append(len(output[0].outputs[0].token_ids))
        results["num_tokens_per_sec"].append(-1)

    return results


@hydra.main(version_base=None, config_path="configs", config_name="default_case")
def main(cfg: DictConfig):

    torch.manual_seed(cfg.seed)

    llm = LLM(**cfg.method.llm)
    sampling_params = SamplingParams(seed=cfg.seed, **cfg.method.generate)

    prompt = cfg.input
    results = warpped_sampling(prompt, llm=llm, sampling_params=sampling_params)

    pprint(results)


if __name__ == "__main__":
    load_dotenv(override=True)
    main()
