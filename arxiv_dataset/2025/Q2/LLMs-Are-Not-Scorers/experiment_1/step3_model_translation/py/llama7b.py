# -*- coding: utf-8 -*-
"""
LLaMA-2-7B Translation Tool for WMT Benchmark

This script processes JSONL files containing source texts in the "src" field,
translates them to English using the meta-llama/Llama-2-7b-chat-hf model, and saves
the results with a new "src_translation" field to Google Drive.

Optimized for Google Colab with A100 GPU to minimize compute unit consumption.
Features batch processing for efficient inference and robust output cleaning.
"""

import json
import torch
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
import pandas as pd
from tqdm import tqdm
import logging
from typing import List, Dict, Any, Iterator
import time
from pathlib import Path
import argparse
import os
import glob

# Log configuration
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Utility functions
def load_jsonl_generator(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def save_jsonl_stream(data_iterator: Iterator[Dict[str, Any]], path: str):
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for item in data_iterator:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
    logger.info(f"Saved {count} records to: {path}")

# Prompt construction function (adapted for decoder-only models)
def build_prompt(example):
    prompt = f"""You are a professional translator.

Translate the following sentence into English with exactly one sentence.

Rules:
1. Output exactly one sentence, and end it with a newline character (`\\n`).
2. The translation must be accurate, fluent, and natural.
3. Do not output anything except the translation.

Sentence: {example['text']}

Translation:
"""
    example["prompt"] = prompt
    return example


# Translation result cleaning function
def extract_translation(output: str, prompt: str) -> str:
    output = output.replace(prompt, "").strip()
    return output.split("\n")[0].strip()

# File processing main logic
def process_file_in_chunks(file_path: str, model, tokenizer, output_dir: Path, args):
    path = Path(file_path)
    output_path = output_dir / path.name
    start_time = time.time()
    chunk_size = 1000
    chunk_idx = 0
    temp_outputs = []

    with open(file_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    logger.info(f"Starting to process: {path.name} (total {total_lines} lines)")

    for chunk_start in range(0, total_lines, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_lines)
        logger.info(f"Processing block {chunk_idx+1}/{(total_lines + chunk_size - 1) // chunk_size}: lines {chunk_start+1}-{chunk_end}")

        data_chunk = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= chunk_start and i < chunk_end:
                    data_chunk.append(json.loads(line))

        df_chunk = pd.DataFrame(data_chunk)
        if "src" not in df_chunk.columns:
            logger.warning(f"Skipping file (missing 'src' field): {path.name}")
            return

        dataset_chunk = Dataset.from_pandas(df_chunk[["src"]].rename(columns={"src": "text"}))
        dataset_chunk = dataset_chunk.map(build_prompt)

        translations = batch_translate_optimized(
            dataset_chunk,
            model,
            tokenizer,
            args.translation_batch_size,
            64,  # max_length hardcoded to 64
            args.temperature,  # temperature for do_sample=False
            5     # num_beams hardcoded to 5
        )

        # Add translation results
        for i, item in enumerate(data_chunk):
            item["src_translation"] = translations[i]

        temp_output_path = output_dir / f"{path.stem}_chunk{chunk_idx}{path.suffix}"
        save_jsonl_stream((item for item in data_chunk), str(temp_output_path))
        temp_outputs.append(temp_output_path)

        del data_chunk, df_chunk, dataset_chunk, translations
        gc.collect()
        torch.cuda.empty_cache()
        chunk_idx += 1

    with open(output_path, 'w', encoding='utf-8') as outfile:
        for temp_file in temp_outputs:
            with open(temp_file, 'r', encoding='utf-8') as infile:
                outfile.write(infile.read())
            temp_file.unlink()

    logger.info(f"Completed {path.name}, time elapsed: {time.time() - start_time:.2f} seconds")

# Batch translation function (with translation extraction)
def batch_translate_optimized(dataset, model, tokenizer, batch_size, max_length, temperature, num_beams):
    translations = []
    adaptive_batch_size = batch_size
    progress_bar = tqdm(total=len(dataset), desc=f"Translation progress")
    i = 0
    while i < len(dataset):
        try:
            current_batch_size = min(adaptive_batch_size, len(dataset) - i)
            batch = dataset[i:i + current_batch_size]
            prompts = batch["prompt"]

            with torch.cuda.amp.autocast():
                inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(model.device)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_length,
                        do_sample=(temperature > 0),  
                        temperature=temperature,
                        num_beams=num_beams,
                        pad_token_id=tokenizer.eos_token_id
                    )

                decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                for j in range(len(decoded)):
                    translation = extract_translation(decoded[j], prompts[j])
                    translations.append(translation)

                progress_bar.update(current_batch_size)
                del inputs, outputs, decoded
                torch.cuda.empty_cache()
                i += current_batch_size

        except RuntimeError as e:
            if "CUDA out of memory" in str(e) and adaptive_batch_size > 1:
                adaptive_batch_size = max(1, adaptive_batch_size // 2)
                logger.warning(f"CUDA out of memory, reducing batch size to {adaptive_batch_size} and retrying...")
                torch.cuda.empty_cache()
                gc.collect()
            else:
                progress_bar.close()
                raise e

    progress_bar.close()
    return translations

# Model loading function
def load_model_and_tokenizer(model_name, token):
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=token,
        padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=token,
            load_in_8bit=True,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
    except:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=token,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
    return model, tokenizer

# Main function
def main():
    parser = argparse.ArgumentParser(description="Translate JSONL files using Llama model")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input JSONL files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where output files will be saved")
    parser.add_argument("--hf_token", type=str, required=True, help="Hugging Face authentication token")
    parser.add_argument("--file_batch_size", type=int, default=5, help="Number of files to process per batch")
    parser.add_argument("--translation_batch_size", type=int, default=4, help="Number of sentences to translate per batch")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for sampling. 0.0 for greedy decoding.")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all JSONL files in the input directory
    input_files = glob.glob(os.path.join(args.input_dir, "*.jsonl"))
    
    logger.info(f"Detected {len(input_files)} input files")
    model, tokenizer = load_model_and_tokenizer("meta-llama/Llama-2-7b-chat-hf", args.hf_token)

    total_files = len(input_files)
    for idx, file_path in enumerate(input_files, start=1):
        logger.info(f"[{idx}/{total_files}] Processing file: {os.path.basename(file_path)}")
        process_file_in_chunks(file_path, model, tokenizer, output_dir, args)
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"Completed {idx}/{total_files} files")

# Entry point
if __name__ == "__main__":
    if torch.cuda.is_available():
        logger.info(f"CUDA available, device: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("CUDA not available, model will run on CPU")

    main()
