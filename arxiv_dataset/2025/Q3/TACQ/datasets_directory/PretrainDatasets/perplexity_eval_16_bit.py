import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn

from gptq.datautils import get_c4, get_c4_new, get_wikitext2, get_ptb_new
from utils.model_utils import load_model

# make print a partial with flush = True
import functools
print = functools.partial(print, flush=True)

@torch.no_grad()
def perplexity_eval(model: nn.Module, testenc, args, dev):
    print(f"\nEvaluating perplexity for {args.dataset_name} dataset ...")
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen
    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        i % 10 == 0 and print(f"Processing sample {i}")
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev) # Batch size given by get_wikitext2 is one, so we don't have to worry much
        model_outputs = model(batch)
        lm_logits = model_outputs.logits
        shift_logits = lm_logits[:, :-1, :].contiguous()  # We don't have a label in this sample for the last predicted token.
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:] # at i == 0(i * model.seqlen):((i + 1) * model.seqlen) selects the first 0:seqlen tokens. [:, 1:] removes the first token from the labels
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    result_str = f"{args.model} {args.dataset_name} perplexity = {ppl.item()}"
    print(result_str)
    return ppl, result_str

def main(args):
    # Load Model
    model_info = load_model(engine=args.model, checkpoints_dir=args.addition_dir, brainfloat=args.brainfloat)
    model, tokenizer = model_info["model"], model_info["tokenizer"]

    tokenizer_name = tokenizer.name_or_path
    del tokenizer
    model.seqlen = args.seqlen
    print("LLM running:", args.model)

    # Load Dataset
    if args.dataset_name == "wikitext2":
        _, testenc = get_wikitext2(0, args.serial_number, model.seqlen, tokenizer_name)  # The last option is really trying to find the name of the Tokenizer
    elif args.dataset_name == "c4_new":
        _, testenc = get_c4_new(0, args.serial_number, model.seqlen, tokenizer_name)
    elif args.dataset_name == "ptb_new":
        _, testenc = get_ptb_new(0, args.serial_number, model.seqlen, tokenizer_name)
    else:
        raise Exception(f"Dataset not supported: {args.dataset_name}")
    # Perform Evaluation
    result, result_str = perplexity_eval(model, testenc, args, model.device)

    # Save Results
    os.makedirs(args.save_dir, exist_ok=True)

    with open(os.path.join(args.save_dir, "results.txt"), "w") as f:
        f.write(result_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", "-s", type=str, required=True)
    parser.add_argument("--addition_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--serial_number", type=int, default=0, required=True)
    parser.add_argument("--model", default="Meta-Llama-3-8B-Instruct", required=True)
    parser.add_argument("--seqlen", type=int, required=True, default=2048)
    parser.add_argument("--brainfloat", action="store_true")
    parser.add_argument("--testing", action="store_true")
    parser.add_argument("--dataset_name", type=str, required=True)
    args = parser.parse_args()
    random.seed(int(args.serial_number))
    np.random.seed(int(args.serial_number))
    torch.manual_seed(int(args.serial_number))
    torch.cuda.manual_seed(int(args.serial_number))

    print("Logging_started")
    main(args)