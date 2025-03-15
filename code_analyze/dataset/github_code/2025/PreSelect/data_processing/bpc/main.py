import torch
from torch.utils.data import DataLoader
import argparse
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from packed_dataset import EvalDataset
import numpy as np
from tqdm import tqdm
import json
torch.set_printoptions(threshold=10_000)

def cross_entropy(
    logits, targets, attention_mask: torch.Tensor = None
):

    logits = logits.reshape(-1, logits.size(-1))
    targets = targets.reshape(-1)
    if attention_mask is not None:
        attention_mask = attention_mask.reshape(-1)
        targets = targets.masked_fill(~attention_mask, -1)

    return torch.nn.functional.cross_entropy(logits, targets, ignore_index=-1, reduction='none')



@torch.no_grad()
def validate(args, model,val_dataset, val_dataloader: DataLoader, device):
    model.eval()
    #save each example's Total loss
    example_total_losses = []
    #  cumulative index on example
    curr_token_index = 0
    # current stored loss length
    curr_loss_len = 0
    # losses = []
    curr_temp_loss = 0
    for k, (val_data, attention_mask) in enumerate(tqdm(val_dataloader)):
        input_ids = val_data[:, 0: args.block_size].contiguous().to(device)
        targets = val_data[:, 1: args.block_size + 1].contiguous().long().to(device)
        attention_mask = attention_mask[:, 1: args.block_size + 1].contiguous().to(device)
        logits = model(input_ids).logits
        loss = cross_entropy(logits, targets, attention_mask=attention_mask)
        loss = loss.cpu().numpy()
        attention_mask = attention_mask.reshape(-1).cpu()
        loss = [number for number, b in zip(loss, attention_mask) if b == True]
        # New added
        curr_loss_len += len(loss)

        # if cumulative loss index > cumulative token index + next example's length, we are able to calculate a new example's total loss
        while val_dataset.token_lens[len(example_total_losses)] <= curr_loss_len:
            remain_len = curr_loss_len - val_dataset.token_lens[len(example_total_losses)]
            example_total_losses.append(curr_temp_loss + sum(loss[:len(loss)-remain_len]))
            dir_name = args.model_name.split("/")[-1]
            with open(f"/home/bpc_calculation_results/{dir_name}/{args.part}.json", "a+") as f:
                output = {}
                output["id"] = val_dataset.ids[len(example_total_losses) - 1]
                output["Model"] = args.model_name
                output["total_loss"] = example_total_losses[-1]
                output["part"] = args.part
                f.write(json.dumps(output))
                f.write("\n")
            curr_temp_loss = 0
            curr_loss_len = remain_len
            loss = loss[len(loss)-remain_len:]
        
        curr_temp_loss += sum(loss)



    # out = np.array(losses).sum()
    out = np.array(example_total_losses).sum() + curr_temp_loss
    # print(len(example_total_losses))
    # print(val_dataset.char_number_list)
    # print(val_dataset.ids)
    return out, np.array(example_total_losses)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_name",
        type=str,
    )
    parser.add_argument(
        "--model_name",
        type=str
    )
    parser.add_argument(
        '--block_size',
        type=int,
        default=1900,
    )
    parser.add_argument(
        '--stride',
        type=int,
        default=512,
    )
    parser.add_argument(
        '--batch_size',
        type=int
    )
    parser.add_argument(
        '--file_num',
        default=-1,
        type=int
    )
    parser.add_argument(
        '--flash',
        action="store_true",
        help="set this if you want to use flash attention",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--part", type=int)
    args = parser.parse_args()
    print(args)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True if ("llemma" in args.model_name) or ("mpt" in args.model_name) else False,
        cache_dir=args.cache_dir,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
        use_flash_attention_2=True if args.flash and "mpt" not in args.model_name else False
    )

    valdataset = EvalDataset(
        args=args,
        task_name=args.task_name,
        block_size=args.block_size + 1,
        tokenizer=tokenizer,
        stride=args.stride,
        vocab_size=tokenizer.vocab_size,
        file_num=args.file_num,
        part=args.part
    )
    valdataloader = DataLoader(valdataset, batch_size=args.batch_size, shuffle=False)
    total_loss, all_losses = validate(args, model, valdataset,valdataloader, device)
    print("-"*10, "Result", "-"*10)
    print("Total loss:", total_loss)
    print("Character num:", valdataset.character_num)
    print("BPC:", total_loss / (valdataset.character_num * np.log(2)) )
    print(all_losses)
    # ------



if __name__ == "__main__":
    main()
