import random
import numpy as np
import torch
import os
import transformers
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import argparse

from datasets_directory.GSM8k.GSM8k_utils import build_prompt, clean_answer, download_url, extract_answer_from_output, is_correct, load_jsonl, seed_everything
from utils.model_utils import load_model

transformers.logging.set_verbosity(40)

N_SHOT = 8
COT_FLAG = True
DEBUG = False
import copy
import os


def load(model_name_or_path):
    print(f"Loading model from {model_name_or_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    model.eval()

    return model, tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoints_dir", 
        type=str,
        help="The directory to load modified model from.",
        required=True
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="The model checkpoint name for weights initialization with init_llama.",
        required=True,
    )
    parser.add_argument(
        "--data_root",
        type=str,
        help="The root folder of the data.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--testing",
        action="store_true",
    )

    parser.add_argument(
        "--brainfloat",
        action="store_true"
    )

    parser.add_argument("--load", type=str, default=None, help="load quantized model")

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    return args


def generate(model, tokenizer, input_text, generate_kwargs):
    input_text = tokenizer(
        input_text,
        padding=False,
        add_special_tokens=True,
        return_tensors="pt",
    )
    input_ids = input_text.input_ids.cuda()
    attention_mask = input_text.attention_mask.cuda()

    output_ids = model.generate(
        input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs
    )
    response = []
    for i in range(output_ids.shape[0]):
        response.append(
            tokenizer.decode(
                output_ids[i][input_ids.shape[1] :],
                skip_special_tokens=True,
                ignore_tokenization_space=True,
            )
        )

    if len(response) > 1:
        return response
    return response[0]


def main():
    args = parse_args()

    seed_everything(args.seed)

    test_filepath = os.path.join(args.data_root, "gsm8k_test.jsonl")
    if not os.path.exists(test_filepath):
        download_url(
            "https://raw.githubusercontent.com/openai/"
            "grade-school-math/2909d34ef28520753df82a2234c357259d254aa8/"
            "grade_school_math/data/test.jsonl",
            args.data_root,
        )
        os.rename(os.path.join(args.data_root, "test.jsonl"), test_filepath)

    list_data_dict = load_jsonl(test_filepath, instruction="question", output="answer")

    model_info = load_model(engine=args.model_name_or_path, checkpoints_dir=args.checkpoints_dir, brainfloat=args.brainfloat)
    model, tokenizer = model_info["model"], model_info["tokenizer"]
    if args.load: #Deprecated
        print("loading...", args.load)
        model_state = torch.load(args.load, map_location="cpu")
        model.load_state_dict(model_state, strict=False)
        model.half().cuda()

    sampled_responses = []
    answers = []
    for idx, sample in enumerate(tqdm(list_data_dict)):
        if args.testing and idx > 20:
            break
        input_text = build_prompt(sample["instruction"], N_SHOT, COT_FLAG)
        generate_kwargs = dict(max_new_tokens=256, top_p=0.95, temperature=0.8)
        model_completion = generate(model, tokenizer, input_text, generate_kwargs)
        model_answer = clean_answer(model_completion)
        is_cor = is_correct(model_answer, sample["output"])
        answers.append(is_cor)
        if DEBUG:
            print(f"Full input_text:\n{input_text}\n\n")
        print(
            f'Question: {sample["instruction"]}\n\n'
            f'Answers: {extract_answer_from_output(sample["output"])}\n\n'
            f"Model Answers: {model_answer}\n\n"
            f"Model Completion: {model_completion}\n\n"
            f"Is correct: {is_cor}\n\n"
        )

        print(
            f"Num of total question: {len(answers)}, "
            f"Correct num: {sum(answers)}, "
            f"Accuracy: {float(sum(answers))/len(answers)}."
        )
        try:
            if idx % 100 == 0:
                sampled_responses.append(
                    (
                        f'--------\n'
                        f'Question: {sample["instruction"]}\n\n'
                        f'Answers: {extract_answer_from_output(sample["output"])}\n\n'
                        f"Model Answers: {model_answer}\n\n"
                        f"Model Completion: {model_completion}\n\n"
                        f"Is correct: {is_cor}\n\n"
                        "\n\n"
                        f"Full input_text:\n{input_text}\n\n\n\n"
                    )
                )
        except:
            pass

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, "results.txt"), "w") as f:
        for answer in answers:
            print(answer, file=f)

    with open(os.path.join(args.output_dir, "scores.txt"), "w") as f:
        print(
            f"Num of total question: {len(answers)}, "
            f"Correct num: {sum(answers)}, "
            f"Accuracy: {float(sum(answers))/len(answers)}.",
            file=f,
        )

    try:
        with open(os.path.join(args.output_dir, "examples.txt"), "w") as f:
            for sample in sampled_responses:
                print(
                    sample,
                    file=f,
                )
    except:
        print("Example capturing failed")


if __name__ == "__main__":
    main()