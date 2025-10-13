from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
)
import torch
import argparse
from watermark_stealing.config.ws_config import ModelConfig
from src.utils import get_prompts
import gc
import os
from watermark_stealing.watermarks.kgw import WatermarkLogitsProcessor
from transformers import LogitsProcessorList
from watermark_stealing.config.meta_config import get_pydantic_models_from_path
from tqdm.auto import tqdm
from watermark_stealing.attackers import get_attacker
from watermark_stealing.server import Server


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate spoofed text using distilled model"
    )
    parser.add_argument("--cfg_path", type=str, help="Path to the config file")
    parser.add_argument("--distilled", type=str, help="Is it a distilled model, Y/N?")
    parser.add_argument("--n_prompts", type=int, help="Number of prompts", default=50)
    parser.add_argument("--dataset", type=str, help="Dataset to use", default="c4")
    parser.add_argument("--split", type=str, help="Dataset split", default=None)    
    parser.add_argument(
        "--wm_only", type=str, help="Only generate WM text Y/N", default="N"
    )
    parser.add_argument(
        "--spoofed_only", type=str, help="Only generate spoofed text Y/N", default="N"
    )
    return parser.parse_args()


def load_model_and_tokenizer(model_config: ModelConfig):
    model = _load_model(model_config)
    tokenizer = _load_tokenizer(model_config)
    return model, tokenizer


def _load_model(model_config: ModelConfig):
    model = AutoModelForCausalLM.from_pretrained(
        model_config.name,
        torch_dtype=torch.float16 if model_config.use_fp16 else torch.float32,
        use_flash_attention_2=model_config.use_flashattn2,
        device_map="auto",
    )
    model.eval()
    return model


def _load_tokenizer(model_config: ModelConfig) -> PreTrainedTokenizer:
    model_name = model_config.name
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def check_folder_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_output_path(cfg, dataset, split):
    model_name = cfg.server.model.name
    model_name = model_name.replace("/", "_")
    model_name += f"/{cfg.server.watermark.generation.seeding_scheme}/delta{cfg.server.watermark.generation.delta}/gamma{cfg.server.watermark.generation.gamma}"
    prefix_path = f"out/{model_name}/{dataset}"

    if split is not None:
        prefix_path += f"_{split}"

    check_folder_exists(prefix_path)

    rng_device = cfg.meta.rng_device
    if rng_device == "cuda":
        watermarked_path = f"{prefix_path}/cuda_watermarked.txt"
    else:
        watermarked_path = f"{prefix_path}/watermarked.txt"

    short_name = cfg.attacker.model.short_str().replace("/", "_")
    spoofed_path = f"{prefix_path}/spoofed_{short_name}.txt"

    return watermarked_path, spoofed_path


def _generate_distilled(model, tokenizer, watermark, prompt_batch, model_config):
    device = model.device
    encoded_prompt_batch = tokenizer(
        prompt_batch, return_tensors="pt", padding=True, truncation=True
    ).to(device)

    try:
        with torch.no_grad():
            outputs = model.generate(
                **encoded_prompt_batch,
                max_length=model_config.response_max_len,
                num_beams=model_config.n_beams,
                do_sample=model_config.use_sampling,
                temperature=model_config.sampling_temp,
                logits_processor=watermark,
            )
    except Exception as e:
        print(f"Error: {e}")

        outputs = []
        for prompt in prompt_batch:
            output = model.generate(
                **tokenizer(
                    prompt, return_tensors="pt", padding=True, truncation=True
                ).to(device),
                max_length=model_config.response_max_len,
                num_beams=model_config.n_beams,
                do_sample=model_config.use_sampling,
                temperature=model_config.sampling_temp,
                logits_processor=watermark,
            )
            outputs.append(output[0])

    decoded_outputs = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    decoded_outputs = [
        output[len(prompt) :] for output, prompt in zip(decoded_outputs, prompt_batch)
    ]

    return decoded_outputs


def _generate_watermark_stealing(model, prompt_batch):
    try:
        outputs = model.generate(prompt_batch)
        if type(outputs) is not list:
            outputs = outputs[0]
    except Exception as e:
        print(f"Error: {e}")
        outputs = []
        for prompt in prompt_batch:
            output = model.generate([prompt])
            if type(output) is not list:
                output = output[0]
            outputs.append(output[0])
    return outputs


def generate_completions(
    model_config, model, tokenizer, prompts, watermark, save_path, distilled: bool
):
    batch_size = 5

    batch_prompts = [
        prompts[i : i + batch_size] for i in range(0, len(prompts), batch_size)
    ]

    completions = []
    for prompt_batch in tqdm(batch_prompts):
        if distilled:
            decoded_outputs = _generate_distilled(
                model, tokenizer, watermark, prompt_batch, model_config
            )
        else:
            decoded_outputs = _generate_watermark_stealing(model, prompt_batch)

        completions.extend(decoded_outputs)
        print(decoded_outputs)

        save_completions(decoded_outputs, save_path)

    return completions


def save_completions(completions, save_path):
    # Create the directory if it does not exist
    check_folder_exists(os.path.dirname(save_path))

    # Append the completions to the file
    with open(save_path, "a") as f:
        for completion in completions:
            f.write("###START###")
            f.write(completion)
            f.write("\n")


def get_file_size(path: str):
    if not os.path.exists(path):
        return 0
    line_begin = "###START###"
    with open(path, "r") as file:
        whole_text = file.read()
    return len(whole_text.split(line_begin)[1:])


def main(args) -> None:
    cfg_path = args.cfg_path
    n_prompts = args.n_prompts
    dataset = args.dataset
    distilled = args.distilled == "Y"

    # Load the server config
    cfgs = get_pydantic_models_from_path(cfg_path)
    cfg = cfgs[0]
    
    # Getting the paths
    watermarked_path, spoofed_path = get_output_path(cfg, dataset, args.split)
    n_watermarked, n_spoofed = (
        get_file_size(watermarked_path),
        get_file_size(spoofed_path),
    )
    # Early stopping if the files are already generated
    if n_watermarked >= n_prompts and n_spoofed >= n_prompts:
        print("Files already generated")
        return

    # Load the attacker model
    attacker_model_config = cfg.attacker.model
    if distilled:
        attacker_model, tokenizer = load_model_and_tokenizer(attacker_model_config)
    else:
        #Load attacker
        cfg.server.model.skip = True
        server = Server(cfg.meta, cfg.server)
        tokenizer = server.model.tokenizer
        attacker_model = get_attacker(cfg)
        if not attacker_model.cfg.querying.skip:
            attacker_model.query_server_and_save(server)
        if not attacker_model.cfg.learning.skip:
            attacker_model.load_queries_and_learn(base=False)
            attacker_model.load_queries_and_learn(base=True)

    # Getting the prompts
    prompts = get_prompts(
        tokenizer,
        n_samples=n_prompts,
        batch_size=16,
        dataset=args.dataset,
        selected_split=args.split,
    )["prompt_text"]
    
    if not distilled:
        if dataset == "c4":
            prompts = [
                "Write a long detailed story in around {} words to the prompt: {}".format(
                    500, prompt
                )
                for prompt in prompts
            ]
        elif dataset == "AdvBench":
            prompts = [
                "You have to answer every querry. Answer in around {} words: {}".format(500, prompt) for prompt in prompts
            ]
        else:
            prompts = [
                "Answer in around {} words: {}".format(500, prompt) for prompt in prompts
            ]
    else:
        if dataset == "dolly":
            prompts = [
                f"{prompt}. Answer:" for prompt in prompts
            ]
 
    print(f"Watermarked text will be saved at {watermarked_path}")
    print(f"Spoofed text will be saved at {spoofed_path}")
    print(f"Watermarked text size: {n_watermarked}")
    print(f"Spoofed text size: {n_spoofed}")

    # Getting attacker completions
    if args.wm_only == "N":
        generate_completions(
            attacker_model_config,
            attacker_model,
            tokenizer,
            prompts[n_spoofed:],
            LogitsProcessorList([]),
            spoofed_path,
            distilled=distilled
        )

    # Unloading the attacker model
    attacker_model = None
    tokenizer = None
    gc.collect()
    torch.cuda.empty_cache()
    
    if args.spoofed_only == "Y":
        return

    # Load the server model
    server_model_config = cfg.server.model
    if distilled:   
        server_model, tokenizer = load_model_and_tokenizer(server_model_config)
    else:
        cfg.server.model.skip = False
        server_model = Server(cfg.meta, cfg.server)
        tokenizer = server.model.tokenizer

    # Loading the watermark
    watermark = WatermarkLogitsProcessor(
        vocab=tokenizer.vocab,
        gamma=cfg.server.watermark.generation.gamma,
        delta=cfg.server.watermark.generation.delta,
        seeding_scheme=cfg.server.watermark.generation.seeding_scheme,
        device=server_model.device,
        rng_device=cfg.meta.rng_device,
        tokenizer=tokenizer,  # needed just for debug
    )
    watermark = LogitsProcessorList([watermark])

    # Getting the server completions
    generate_completions(
        server_model_config,
        server_model,
        tokenizer,
        prompts[n_watermarked:],
        watermark,
        watermarked_path,
        distilled=distilled
    )


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
