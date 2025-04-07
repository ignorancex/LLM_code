import argparse
import json
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from str2bool import str2bool
import os
import re


def is_valid_think_format(text):
    # Check basic pattern first
    pattern = r'^<think>(.*?)</think>(.+)$'  # Note the .+ for non-empty after content
    match = re.match(pattern, text, re.DOTALL)

    if not match:
        return False

    # Extract the content inside and after the think tags
    inside_content = match.group(1)
    after_content = match.group(2)

    # Verify inside content is not empty
    if not inside_content.strip():
        return False

    # Verify neither part contains additional think tags
    if '<think>' in inside_content or '</think>' in inside_content:
        return False

    if '<think>' in after_content or '</think>' in after_content:
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Evaluate large language models on critical datasets.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset file.")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to store cached outputs.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model.")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to the pretrained model.")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type to use for the model (e.g., fp16, bf16, etc.).")
    parser.add_argument("--n_gpus", type=int, default=8, help="Number of GPUs to use for tensor parallelism.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for generation.")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p sampling for generation.")
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max_len", type=int, default=32768, help="Maximum number of tokens to generate.")
    parser.add_argument("--use_chat_template", type=str2bool, default=False)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--max_retries", type=int, default=8)

    args = parser.parse_args()

    if args.tokenizer_path is None:
        args.tokenizer_path = args.model_path

    # Load the tokenizer for LLaMA or any model
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # Load inference framework
    model = LLM(
        model=args.model_path,
        tokenizer=args.tokenizer_path,
        tokenizer_mode="slow",
        dtype=args.dtype,
        tensor_parallel_size=args.n_gpus,
        enforce_eager=True,
    )

    items = []
    completions = []
    seed = 0
    for _ in range(args.n):
        prompts = []
        with open(args.data_path, encoding="utf-8") as f:
            for line in f.readlines():
                item = json.loads(line)
                prompt = item["prompt"]
                if args.use_chat_template:
                    messages = [
                        {"role": "user", "content": prompt}
                    ]
                    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                prompts.append(prompt)
                items.append(item)

        with torch.no_grad():
            # Initialize with None for each prompt position
            output_lst = [None] * len(prompts)
            pending_prompts = prompts.copy()
            pending_indices = list(range(len(prompts)))

            retry_count = 0

            while pending_prompts and retry_count < args.max_retries:
                sampling_params = SamplingParams(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_len,
                    repetition_penalty=args.repetition_penalty,
                    seed=seed,
                )
                seed += 1

                # Generate completions for remaining prompts
                batch_outputs = model.generate(pending_prompts, sampling_params)
                batch_texts = [completion.outputs[0].text for completion in batch_outputs]

                # Process current batch results
                still_pending_prompts = []
                still_pending_indices = []

                for i, (idx, text) in enumerate(zip(pending_indices, batch_texts)):
                    if is_valid_think_format(text):
                        # If valid, add to results at original position
                        output_lst[idx] = text
                    else:
                        # If invalid, keep for retry
                        still_pending_prompts.append(pending_prompts[i])
                        still_pending_indices.append(idx)
                        # Store the invalid output in case we reach max retries
                        output_lst[idx] = text

                # Update for next iteration
                pending_prompts = still_pending_prompts
                pending_indices = still_pending_indices
                retry_count += 1

        completions.extend(output_lst)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        for item, completion in zip(items, completions):
            item["completion"] = completion
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    main()
