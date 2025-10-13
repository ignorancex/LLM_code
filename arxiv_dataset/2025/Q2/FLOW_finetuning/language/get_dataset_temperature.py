import os
import numpy as np
import argparse
import torch.distributed
from tqdm import tqdm
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, DataLoader

from models import create_model_tokenizer_it
from utils.data_utils import load_and_preprocess_it, DataCollatorForSupervisedDataset, IGNORE_INDEX

def init_distributed():
    if dist.is_initialized():
        return
    
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        
        dist.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=rank
        )
        torch.cuda.set_device(rank)
        return rank, world_size
    else:
        print("Distributed environment variables not set, running in non-distributed mode")
        return 0, 1

def save_statistics(losses, path, loss_type):
    total_samples = len(losses)

    # Sort losses and calculate statistics
    sorted_losses = np.sort(losses)
    median_loss = np.median(losses)
    mean_loss = np.mean(losses)
    min_loss = np.min(losses)
    max_loss = np.max(losses)

    print("First 10 samples:", sorted_losses[:10])
    print("Last 10 samples:", sorted_losses[-10:])

    # Save results
    with open(os.path.join(path, f"{loss_type}_results.txt"), 'w') as file:
        print(f'\nTotal samples processed: {total_samples}', file=file)
        print(f'Dataset size: {total_samples}', file=file)
        print(f'Median Loss: {median_loss:.4f}', file=file)
        print(f'Mean Loss: {mean_loss:.4f}', file=file)
        print(f'Min Loss: {min_loss:.4f}', file=file)
        print(f'Max Loss: {max_loss:.4f}', file=file)

    np.save(os.path.join(path, f"{loss_type}_losses.npy"), sorted_losses)

    # Print statistics
    print(f'\nTotal samples processed: {total_samples}')
    print(f'Dataset size: {total_samples}')
    print(f'Median Loss: {median_loss:.4f}')
    print(f'Mean Loss: {mean_loss:.4f}')
    print(f'Min Loss: {min_loss:.4f}')
    print(f'Max Loss: {max_loss:.4f}')

def gather(q, rank, ws, dst, device):
    local_size = torch.tensor(q.size(), device=device)
    all_sizes = [torch.zeros_like(local_size) for _ in range(ws)]
    dist.all_gather(all_sizes, local_size)
    max_size = max(all_sizes)

    size_diff = max_size.item() - local_size.item()
    if size_diff:
        padding = torch.zeros(size_diff, device=device, dtype=q.dtype)
        q = torch.cat((q, padding))

    if rank == dst:
        all_qs_padded = [torch.zeros_like(q) for _ in range(ws)]
        dist.gather(q, all_qs_padded, dst=dst)
        all_qs = []
        for q, size in zip(all_qs_padded, all_sizes):
            all_qs.append(q[:size])
        return all_qs
    else:
        dist.gather(q, dst=dst)
        return None

def main(args):
    rank, world_size = init_distributed()
    model, tokenizer = create_model_tokenizer_it(args)

    model = model.to(args.device)
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    model.eval()

    train_dataset = load_and_preprocess_it(tokenizer=tokenizer, args=args)
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.bs,
        sampler=sampler,
        pin_memory=True,
        collate_fn=data_collator
    )

    criterion = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=IGNORE_INDEX)

    print(f"Starting Sample rewighting on rank {rank}...\n")
    all_sequence_losses = []
    all_token_losses = []
    for batch in tqdm(dataloader, disable=rank != 0):

        input_ids = batch["input_ids"].to(args.device)
        attention_mask = batch["attention_mask"].to(args.device)
        labels = batch["labels"].to(args.device)

        # Calculate losses
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            token_losses = criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            ).view(shift_labels.size())
            
            # We need to mask out the labels that we ignore and the padding tokens
            padding_mask = shift_labels.eq(IGNORE_INDEX)
            num_active_elements = (~padding_mask).sum(dim=-1)

            if args.loss_type == "sequence":
                sequence_losses = token_losses.sum(dim=-1) / num_active_elements
                all_sequence_losses.extend(sequence_losses.cpu().tolist())
            elif args.loss_type == "token":
                for weight, mask in zip(token_losses, padding_mask):
                    all_token_losses.extend(weight[~mask].cpu().tolist())
            else: # Assume both:
                sequence_losses = token_losses.sum(dim=-1) / num_active_elements
                all_sequence_losses.extend(sequence_losses.cpu().tolist())
                for weight, mask in zip(token_losses, padding_mask):
                    all_token_losses.extend(weight[~mask].cpu().tolist())
    
    if world_size > 1:
        dist.barrier()
        # Collect all the data to rank 0
        collected_sequence_losses = gather(torch.Tensor(all_sequence_losses).to(args.device), rank, world_size, 0, args.device)
        collected_token_losses = gather(torch.Tensor(all_token_losses).to(device=args.device), rank, world_size, 0, args.device)
    else:
        collected_sequence_losses = [torch.Tensor(all_sequence_losses)]
        collected_token_losses = [torch.Tensor(all_token_losses)]

    if rank == 0:
        collected_sequence_losses = torch.cat(collected_sequence_losses, 0).cpu().numpy()
        collected_token_losses = torch.cat(collected_token_losses, 0).cpu().numpy()

        if args.run_dir is not None:
            if args.save_path is not None:
                print("Both --run-dir and --save-path were passed, defaulting to --run-dir")
            save_path = os.path.join(args.run_dir, "temperature_results")
        elif args.save_path is not None:
            save_path = args.save_path
        
        os.makedirs(save_path, exist_ok=True)

        if len(collected_sequence_losses):
            save_statistics(collected_sequence_losses, path=save_path, loss_type="sequence")
        if len(collected_token_losses):
            save_statistics(collected_token_losses, path=save_path, loss_type="token")


def parse_args():
    parser = argparse.ArgumentParser(description="A script for weighting the samples of a dataset")

    parser.add_argument("--data-path", type=str, help="Location to the dataset")
    parser.add_argument("--dataset-split", type=str, default="train[:50000]", help="Dataset split to use")
    parser.add_argument("--dataset-field", type=str, nargs="+", default=["query", "response"], help="Fields of dataset input and output")
    parser.add_argument("--save-path", type=str, help="The path to save the datset when done")
    parser.add_argument("--run-dir", type=str, default=None, help="The path to a local run dir (Use this to save run info)")
    parser.add_argument("--model", type=str, default=None, help="The model to use as a reweighter")
    parser.add_argument("--tokenizer", type=str, default=None, help="The tokenizer to use as a reweighter")
    parser.add_argument("--loss-type", type=str, help="The sample reweighting scheme [sequence/token]")

    parser.add_argument("--bs", type=int, default=16, help="Batch Size")
    parser.add_argument("--max-seq-length", type=int, default=1024, help="The maximize sequence length to process samples")
    parser.add_argument("--device", type=str, default="cuda", help="The device to use")

    args = parser.parse_args()

    if args.run_dir is None and args.model is None:
        raise RuntimeError("User didn't pass --run-dir or --model, please ensure that one and only one is passed")

    if args.run_dir is not None:
        if args.model is not None:
            print("Both --run-dir and --model were passed, defaulting to --run-dir")
        
        if args.tokenizer is not None:
            print("Both --run-dir and --tokenizer were passed, defaulting to --run-dir")
        
        args.model = os.path.join(args.run_dir, "final_model")
        args.tokenizer = os.path.join(args.run_dir, "tokenizer")

    return args


if __name__ == "__main__":
    args = parse_args()

    main(args=args)