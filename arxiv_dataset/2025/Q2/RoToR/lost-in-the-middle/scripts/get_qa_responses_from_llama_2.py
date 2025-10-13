#!/usr/bin/env python3
"""Given a data file with questions and retrieval results to use, run Llama-2 to get responses.

Currently supports:

- meta-llama/Llama-2-7b-hf
- meta-llama/Llama-2-13b-hf
- meta-llama/Llama-2-70b-hf
- meta-llama/Llama-2-7b-chat-hf
- meta-llama/Llama-2-13b-chat-hf
- meta-llama/Llama-2-70b-chat-hf

The retrieval results are used in the exact order that they're given.
"""
import argparse
import dataclasses
import json
import logging
import pathlib
import random
import sys
from copy import deepcopy

import torch
from tqdm import tqdm
from xopen import xopen
import sys
sys.path.append('/workspace/jenny/TableInvariance/lost-in-the-middle/src/')
from lost_in_the_middle.prompting import (
    Document,
    get_closedbook_qa_prompt,
    get_qa_prompt,
)

logger = logging.getLogger(__name__)
random.seed(0)


def main(
    input_path,
    model_name,
    temperature,
    top_p,
    closedbook,
    prompt_mention_random_ordering,
    use_random_ordering,
    query_aware_contextualization,
    num_gpus,
    max_new_tokens,
    max_prompt_length,
    hf_cache_path,
    output_path,
    mode
):
    from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig
    sys.path.append('/workspace/jenny/TableInvariance/')
    from PINE.pine import LlamaForCausalLMWithPSNew, LlamaTokenizerFastWithPS
    # Create directory for output path if it doesn't exist.
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    if mode == 'orig':
        logger.info("Loading tokenizer")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
        tokenizer.padding_side = "left"
        #tokenizer.pad_token = tokenizer.eos_token  # to avoid an error
    elif mode == 'ours':
        tokenizer = LlamaTokenizerFastWithPS.from_pretrained(model_name, device='cuda')

    examples = []
    prompts = []
    all_model_documents = []
    did_format_warn = False

    # Fetch all of the prompts
    with xopen(input_path) as fin:
        for line in tqdm(fin):
            input_example = json.loads(line)
            # Get the prediction for the input example
            question = input_example["question"]
            if closedbook:
                documents = []
            else:
                documents = []
                for ctx in deepcopy(input_example["ctxs"]):
                    documents.append(Document.from_dict(ctx))
                if not documents:
                    raise ValueError(f"Did not find any documents for example: {input_example}")

            if use_random_ordering:
                # Randomly order only the distractors (isgold is False), keeping isgold documents
                # at their existing index.
                (original_gold_index,) = [idx for idx, doc in enumerate(documents) if doc.isgold is True]
                original_gold_document = documents[original_gold_index]
                distractors = [doc for doc in documents if doc.isgold is False]
                random.shuffle(distractors)
                distractors.insert(original_gold_index, original_gold_document)
                documents = distractors

            if closedbook:
                prompt = get_closedbook_qa_prompt(question)
            else:
                prompt = get_qa_prompt(
                    question,
                    documents,
                    mention_random_ordering=prompt_mention_random_ordering,
                    query_aware_contextualization=query_aware_contextualization,
                )

            if "chat" in model_name:
                if did_format_warn is False:
                    logger.warning(f"Model {model_name} appears to be an chat model, applying chat formatting")
                    did_format_warn = True
                prompt = format_chat_prompt(prompt)

            #prompt_length = len(tokenizer(prompt)["input_ids"])
            #if max_prompt_length < prompt_length:
            #    logger.info(
            #        f"Skipping prompt {prompt[:100]}... with length {prompt_length}, which "
            #        f"is greater than maximum prompt length {max_prompt_length}"
            #    )
            #    continue

            prompts.append(prompt)
            examples.append(deepcopy(input_example))
            all_model_documents.append(documents)

    logger.info(f"Loaded {len(prompts)} prompts to process")

    # Get responses for all of the prompts
    if not torch.cuda.is_available():
        raise ValueError("Unable to find CUDA device with torch. Please use a CUDA device to run this script.")

    logger.info("Loading model")
    if mode == 'orig':
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map='auto')
    elif mode == 'ours':
        config = LlamaConfig.from_pretrained(model_name)
        config._attn_implementation = 'eager'
        model = LlamaForCausalLMWithPSNew.from_pretrained(model_name, torch_dtype=torch.bfloat16, config=config, device_map='auto')
    model.generation_config.do_sample= False
    model.generation_config.max_new_tokens= 100
    model.generation_config.pad_token_id= tokenizer.eos_token_id
    """
    model = LLM(
        model=model_name,
        tensor_parallel_size=num_gpus,
        trust_remote_code=True,
        download_dir=hf_cache_path,
        load_format="pt",
        max_num_batched_tokens=max_prompt_length,
    )
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens)
    """
    responses = []
    for prompt in tqdm(prompts):
        splitted = prompt.split('\nDocument [')
        prefix = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{splitted[0]}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        end, suffix = splitted[-1].split('\n\nQuestion: ')
        parallel = ['\nDocument [' + x for x in splitted[1:-1] + [end]]
        suffix = '\n\nQuestion: ' + suffix + '<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
        if mode == 'orig':
            new = prefix + ''.join(parallel) + suffix
            inputs = tokenizer(new, return_tensors='pt').to('cuda')
        elif mode == 'ours':
            inputs = tokenizer([prefix, parallel, suffix])
            inputs = {k: torch.tensor(v).to('cuda') for k,v in inputs.items()}
        outputs = model.generate(**inputs)
        raw_out = outputs[0][inputs['input_ids'].shape[-1]:]
        responses.append(tokenizer.decode(raw_out, skip_special_tokens=True).strip())
        #print(f"\nInput: \n{prefix}\n{parallel}\n{suffix}\nOutput: {responses[-1]}\n")
    with xopen(output_path, "w") as f:
        for example, model_documents, prompt, response in zip(examples, all_model_documents, prompts, responses):
            output_example = deepcopy(example)
            # Add some extra metadata to the output example
            output_example["model_prompt"] = prompt
            output_example["model_documents"] = [dataclasses.asdict(document) for document in model_documents]
            output_example["model_answer"] = response
            output_example["model"] = model_name
            output_example["model_temperature"] = temperature
            output_example["model_top_p"] = top_p
            output_example["model_prompt_mention_random_ordering"] = prompt_mention_random_ordering
            output_example["model_use_random_ordering"] = use_random_ordering
            f.write(json.dumps(output_example) + "\n")


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def format_chat_prompt(message: str):
    DEFAULT_SYSTEM_PROMPT = (
        "You are a helpful, respectful and honest assistant. "
        "Always answer as helpfully as possible, while being safe. "
        "Please ensure that your responses are socially unbiased and positive in nature. "
        "If a question does not make any sense, or is not factually coherent, explain "
        "why instead of answering something not correct. If you don't know the answer "
        "to a question, please don't share false information."
    )
    lines = ["<s>[INST] <<SYS>>", DEFAULT_SYSTEM_PROMPT, "<</SYS>>", "", f"{message} [/INST]"]
    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(module)s - %(levelname)s - %(message)s", level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", help="Path to data with questions and documents to use.", required=True)
    parser.add_argument(
        "--model",
        help="Model to use in generating responses",
        required=True,
        choices=[
            "meta-llama/Llama-2-7b-hf",
            "meta-llama/Llama-2-13b-hf",
            "meta-llama/Llama-2-70b-hf",
            "meta-llama/Llama-2-7b-chat-hf",
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "meta-llama/Llama-2-13b-chat-hf",
            "meta-llama/Llama-2-70b-chat-hf",
        ],
    )
    parser.add_argument("--temperature", help="Temperature to use in generation", type=float, default=0.0)
    parser.add_argument("--top-p", help="Top-p to use in generation", type=float, default=1.0)
    parser.add_argument(
        "--closedbook", action="store_true", help="Run the model in closed-book mode (i.e., don't use documents)."
    )
    parser.add_argument(
        "--prompt-mention-random-ordering",
        action="store_true",
        help="Mention that search results are ordered randomly in the prompt",
    )
    parser.add_argument(
        "--use-random-ordering",
        action="store_true",
        help="Randomize the ordering of the distractors, rather than sorting by relevance.",
    )
    parser.add_argument(
        "--query-aware-contextualization",
        action="store_true",
        help="Place the question both before and after the documents.",
    )
    parser.add_argument("--num-gpus", help="Number of GPUs to use", type=int, default=1)
    parser.add_argument("--hf-cache-path", help="Path to huggingface cache to use.")
    parser.add_argument("--output-path", help="Path to write output file of generated responses", required=True)
    parser.add_argument(
        "--max-new-tokens",
        help="Maximum number of new tokens to generate",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--max-prompt-length",
        help="Maximum number of tokens in the prompt. Longer prompts will be skipped.",
        type=int,
        default=8192,
    )
    parser.add_argument('--mode', type=str, default='orig')
    args = parser.parse_args()

    logger.info("running %s", " ".join(sys.argv))
    main(
        args.input_path,
        args.model,
        args.temperature,
        args.top_p,
        args.closedbook,
        args.prompt_mention_random_ordering,
        args.use_random_ordering,
        args.query_aware_contextualization,
        args.num_gpus,
        args.max_new_tokens,
        args.max_prompt_length,
        args.hf_cache_path,
        args.output_path,
        args.mode
    )
    logger.info("finished running %s", sys.argv[0])
