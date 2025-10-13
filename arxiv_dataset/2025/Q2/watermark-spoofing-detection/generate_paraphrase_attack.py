import argparse
import json
import nltk
import time
import os
import tqdm

from nltk.tokenize import sent_tokenize
from watermark_stealing.config.meta_config import get_pydantic_models_from_path

from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

from generate_text import get_output_path, save_completions
import pandas as pd
from datasets import Dataset
import argparse

nltk.download("punkt")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process Text.")
    parser.add_argument("--cfg_path", type=str, help="Path to the config file.")
    parser.add_argument("--dataset", type=str, help="Dataset to use", default="c4")
    parser.add_argument("--split", type=str, help="Dataset split", default=None)  
    return parser.parse_args()

def generate_dipper_paraphrases(
    data,
    model_name="kalpeshk2011/dipper-paraphraser-xxl",
    no_ctx=True,
    sent_interval=3,
    start_idx=None,
    end_idx=None,
    paraphrase_file=".output/dipper_attacks.jsonl",
    lex=20,
    order=0,
):
    if no_ctx:
        paraphrase_file = paraphrase_file.split(".jsonl")[0] + "_no_ctx" + ".jsonl"

    if sent_interval == 1:
        paraphrase_file = paraphrase_file.split(".jsonl")[0] + "_sent" + ".jsonl"

    output_file = (
        paraphrase_file.split(".jsonl")[0]
        + "_L_"
        + f"{lex}"
        + "_O_"
        + f"{order}"
        + "_pp"
        + ".jsonl"
    )

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            num_output_points = len([json.loads(x) for x in f.read().strip().split("\n")])
    else:
        num_output_points = 0
    print(f"Skipping {num_output_points} points")

    time1 = time.time()
    tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")
    model = T5ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    print("Model loaded in ", time.time() - time1)
    # model.half()
    model.cuda()
    model.eval()

    data = (
        data.select(range(0, len(data)))
        if start_idx is None or end_idx is None
        else data.select(range(start_idx, end_idx))
    )

    # iterate over data and tokenize each instance
    w_wm_output_attacked = []
    dipper_inputs = []
    for idx, dd in tqdm.tqdm(enumerate(data), total=len(data)):
        if idx < num_output_points:
            continue
        # tokenize prefix
        if "w_wm_output_attacked" not in dd:
            # paraphrase_outputs = {}
            if isinstance(dd["w_wm_output"], str):
                input_gen = dd["w_wm_output"].strip()
            else:
                input_gen = dd["w_wm_output"][0].strip()

            # The lexical and order diversity codes used by the actual model correspond to "similarity" rather than "diversity".
            # Thus, for a diversity measure of X, we need to use control code value of 100 - X.
            lex_code = int(100 - lex)
            order_code = int(100 - order)

            # remove spurious newlines
            input_gen = " ".join(input_gen.split())
            sentences = sent_tokenize(input_gen)
            prefix = " ".join(dd["truncated_input"].replace("\n", " ").split())
            output_text = ""
            final_input_text = ""

            for sent_idx in range(0, len(sentences), sent_interval):
                curr_sent_window = " ".join(sentences[sent_idx : sent_idx + sent_interval])
                if no_ctx:
                    final_input_text = f"lexical = {lex_code}, order = {order_code} <sent> {curr_sent_window} </sent>"
                else:
                    final_input_text = f"lexical = {lex_code}, order = {order_code} {prefix} <sent> {curr_sent_window} </sent>"

                if idx == 0 and lex_code == 60 and order_code == 60:
                    print(final_input_text)

                final_input = tokenizer([final_input_text], return_tensors="pt")
                final_input = {k: v.cuda() for k, v in final_input.items()}

                with torch.inference_mode():
                    outputs = model.generate(
                        **final_input, do_sample=True, top_p=0.75, top_k=None, max_length=512
                    )
                outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                prefix += " " + outputs[0]
                output_text += " " + outputs[0]

            # paraphrase_outputs[f"lex_{lex_code}_order_{order_code}"] = {
            #     "final_input": final_input_text,
            #     "output": [output_text.strip()],
            #     "lex": lex_code,
            #     "order": order_code
            # }
            # dd["w_wm_output_attacked"] = paraphrase_outputs
            w_wm_output_attacked.append(output_text.strip())
            dipper_inputs.append(final_input_text)

        # with open(output_file, "a") as f:
        #     f.write(json.dumps(dd) + "\n")
    # add w_wm_output_attacked to hf dataset object as a column
    data = data.add_column("w_wm_output_attacked", w_wm_output_attacked)
    data = data.add_column(f"dipper_inputs_Lex{lex}_Order{order}", dipper_inputs)
    
    del model
    torch.cuda.empty_cache()

    return data

def read_text(path):
    line_begin = "###START###"
    with open(path, "r") as file:
        whole_text = file.read()
    return whole_text.split(line_begin)[1:]




def main(args):
    
    cfg_path = args.cfg_path
    cfg = get_pydantic_models_from_path(cfg_path)[0]
    watermarked_path, _ = get_output_path(cfg, args.dataset, args.split)

    save_path = watermarked_path.replace(".txt","") + "-dipper.text"
    
    text = read_text(watermarked_path)
    df = pd.DataFrame({"w_wm_output": text, "truncated_input": [""]*len(text)})
    hf_dataset = Dataset.from_pandas(df)

    hf_dataset = generate_dipper_paraphrases(hf_dataset, lex=10)
    completions = hf_dataset["w_wm_output_attacked"]
    
    save_completions(completions, save_path)
    
    
    
    