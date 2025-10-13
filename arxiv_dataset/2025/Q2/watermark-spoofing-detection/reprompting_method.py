from watermark_stealing.config.meta_config import get_pydantic_models_from_path
from watermark_stealing.server import Server
import numpy as np
from tqdm.auto import tqdm
import argparse
import os
import json
from watermark_stealing.config.ws_config import ModelConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
)
from watermark_stealing.watermarks.kgw import WatermarkLogitsProcessor
from transformers import LogitsProcessorList
import torch


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate distribution of statistics")
    parser.add_argument(
        "--cfg_path", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument(
        "--n_samples", type=int, required=True, help="Number of samples to compute"
    )
    parser.add_argument(
        "--distilled_model", type=str, required=True, help="Is it a distilled model, Y/N?"
    )
    parser.add_argument(
        "--dataset", type=str, help="Dataset.", default="c4"
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


def read_text(path):
    line_begin = "###START###"
    with open(path, "r") as file:
        whole_text = file.read()
    return whole_text.split(line_begin)[1:]


def get_model_name(cfg):
    model_name = cfg.server.model.name.replace("/", "_")
    model_name += f"/{cfg.server.watermark.generation.seeding_scheme}/delta{cfg.server.watermark.generation.delta}/gamma{cfg.server.watermark.generation.gamma}"
    return model_name


def load_generated_text(cfg, dataset, spoofed_only: bool):
    attacker_short_name = cfg.attacker.model.short_str().replace("/", "_")
    model_name = get_model_name(cfg)

    if cfg.meta.rng_device == "cpu":
        path = f"out/{model_name}/{dataset}/watermarked.txt"
    else:
        path = f"out/{model_name}/{dataset}/cuda_watermarked.txt"
        print(path)
        
    if not spoofed_only:
        server_text = read_text(path)
    else:
        server_text = []

    path = f"out/{model_name}/{dataset}/spoofed_{attacker_short_name}.txt"
    spoofed_text = read_text(path)

    return server_text, spoofed_text


def load_server(cfg, distilled: bool):
    cfg.server.model.skip = distilled # Skip the model loading for distilled models
    server = Server(cfg.meta, cfg.server)
    tokenizer = server.model.tokenizer
    return server, tokenizer

def _generate_distilled(model, tokenizer, base_prompt, watermark, base_length, max_length, server):
    
    try:
        with torch.no_grad():
            outputs = model.generate(
                base_prompt,
                max_length=base_length + max_length,
                do_sample=server.model.cfg.use_sampling,
                temperature=server.model.cfg.sampling_temp,
                logits_processor=watermark,
            )
            generated_sentences = tokenizer.batch_decode(
                outputs[:, base_length:], skip_special_tokens=True
            )
    except RuntimeError as e:
        print(f"Error: {e}")
        generated_sentences = []
        for i in range(len(base_prompt)):
            output = model.generate(
                base_prompt[i].unsqueeze(0),
                max_length=base_length + max_length,
                do_sample=server.model.cfg.use_sampling,
                temperature=server.model.cfg.sampling_temp,
                logits_processor=watermark,
            )
            generated_sentences.append(
                tokenizer.decode(output[0, base_length:], skip_special_tokens=True)
            )

    return generated_sentences

def _generate_server(server, base_prompt, base_length, max_length):
    
    tokenizer = server.model.tokenizer
    server.model.cfg.response_max_len = base_length + max_length
    server.model.cfg.response_min_len = base_length
    
    prompts = ["Complete the text:" + partial_sentence for partial_sentence in base_prompt]
    try:
        out = server.generate(prompts)
        if type(out) is not list:
            out = out[0]
        generated_sentences = out
    except RuntimeError as e:
        print(f"Error: {e}")
        outputs = []
        for prompt in prompts:
            out = server.generate([prompt]) 
            if type(out) is not list:
                out = out[0] 
            outputs.append(out)
            
        generated_sentences = [output[0] for output in outputs]
        
    generated_sentences =  tokenizer(generated_sentences, return_tensors="pt", padding=True, truncation=True).input_ids
    generated_sentences = generated_sentences[:, base_length:]
    generated_sentences = tokenizer.batch_decode(generated_sentences, skip_special_tokens=True)
            
    return generated_sentences

# Function to process each batch
def process_batch(model, tokenizer, watermark, server, text_batch, distilled: bool):
    base_length = 25
    max_length = 200

    batch_encoded = tokenizer(
        text_batch,
        padding=True,
        truncation=True,
        max_length=base_length + max_length,
        return_tensors="pt",
    )
    input_ids = batch_encoded["input_ids"].to(model.device)

    base_prompt = input_ids[:, :base_length]
    sentences = tokenizer.batch_decode(
        input_ids[:, base_length:], skip_special_tokens=True
    )
    base_prompt_str = tokenizer.batch_decode(base_prompt, skip_special_tokens=True)

    # Reset the seed
    torch.manual_seed(np.random.randint(0, 1000000))

    if distilled:
        generated_sentences = _generate_distilled(
            model, tokenizer, base_prompt, watermark, base_length, max_length, server
        )
    else:
        generated_sentences = _generate_server(server, base_prompt_str, base_length, max_length)
        

    # Print the generated sentences
    for sentence in generated_sentences:
        print(sentence)

    return sentences, generated_sentences


def get_save_text_path(cfg, dataset: str, type: str = "watermarked"):
    model_name = get_model_name(cfg)
    prefix = f"out/reprompting/{model_name}/{dataset}/"

    if type == "watermarked":
        rng_device = cfg.meta.rng_device
        if rng_device == "cpu":
            path = f"{prefix}watermarked.jsonl"
        else:
            path = f"{prefix}cuda_watermarked.jsonl"
    elif type == "spoofed":
        attacker_short_name = cfg.attacker.model.short_str().replace("/", "_")
        path = f"{prefix}spoofed_{attacker_short_name}.jsonl"

    return path


def get_save_text_id(path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "r") as file:
            lines = file.readlines()
            last_line = lines[-1]
            last_data = json.loads(last_line)
            last_id = last_data.get("id", 0)
            return last_id
    return 0


def save_text(watermarked, original, cfg, dataset, type: str = "watermarked"):
    path = get_save_text_path(cfg, dataset=dataset, type=type)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    data = {"watermarked": watermarked, "original": original}

    data["id"] = get_save_text_id(path) + 1

    # Write the new entry to the file
    with open(path, "a") as file:
        json_record = json.dumps(data)
        file.write(json_record + "\n")


def main(args):
    batch_size = 5

    distilled = args.distilled_model == "Y"
    spoofed_only = args.spoofed_only == "Y"
    
    cfg = get_pydantic_models_from_path(args.cfg_path)[0]
    
    save_text_watermarked_path, save_text_spoofed_path = (
        get_save_text_path(cfg, dataset=args.dataset, type="watermarked"),
        get_save_text_path(cfg, dataset=args.dataset, type="spoofed"),
    )
    watermarked_id, spoofed_id = (
        get_save_text_id(save_text_watermarked_path),
        get_save_text_id(save_text_spoofed_path),
    )
    
    # Early stopping
    if watermarked_id >= args.n_samples and spoofed_id >= args.n_samples:
        print("All samples already generated")
        return

    
    server, _ = load_server(cfg, distilled)

    server_text, spoofed_text = load_generated_text(cfg, args.dataset, spoofed_only)


    server_text = server_text[watermarked_id : args.n_samples]
    spoofed_text = spoofed_text[spoofed_id : args.n_samples]

    # Load the server model
    if distilled:
        server_model, tokenizer = load_model_and_tokenizer(cfg.server.model)
    else:
        server_model, tokenizer = server.model, server.model.tokenizer

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

    server_text_batches = [
        server_text[i : i + batch_size] for i in range(0, len(server_text), batch_size)
    ]
    spoofed_text_batches = [
        spoofed_text[i : i + batch_size]
        for i in range(0, len(spoofed_text), batch_size)
    ]

    if not args.spoofed_only == "Y":
        for server_batch in tqdm(server_text_batches):
            server_sentences, server_generated_sentences = process_batch(
                server_model, tokenizer, watermark, server, server_batch, distilled
            )

            for server_sentence, server_generated_sentence in zip(
                server_sentences, server_generated_sentences
            ):
                save_text(
                    server_generated_sentence, server_sentence, cfg, args.dataset, type="watermarked"
                )

    for spoofed_batch in tqdm(spoofed_text_batches):
        spoofed_sentences, spoofed_generated_sentences = process_batch(
            server_model, tokenizer, watermark, server, spoofed_batch, distilled
        )

        for spoofed_sentence, spoofed_generated_sentence in zip(
            spoofed_sentences, spoofed_generated_sentences
        ):
            save_text(spoofed_generated_sentence, spoofed_sentence, cfg, args.dataset, type="spoofed")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
