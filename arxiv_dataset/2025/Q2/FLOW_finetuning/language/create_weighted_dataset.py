import os
import torch
import json
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DistributedSampler, DataLoader
import argparse
from tqdm import tqdm
from datetime import datetime
from datasets import Dataset as HuggingFaceDataset

from utils.data_utils import load_and_preprocess_it, IGNORE_INDEX, IndexedDataCollatorForSupervisedDataset, WeightedDataCollatorForSupervisedDataset
from models import create_model_tokenizer_it

class WeightedDataset(Dataset):
    def __init__(self, original_dataset):
        self.dataset = original_dataset
        self.sequence_weights = torch.zeros(len(original_dataset))
        self.token_weights = [[] for _ in range(len(original_dataset))]
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        item['sequence_weights'] = self.sequence_weights[idx]
        item['token_weights'] = self.token_weights[idx]
        item['index'] = idx
        return item
    
    def set_weights(self, indices, sequence_weights=None, token_weights=None):
        for i, idx in enumerate(indices):
            if 0 <= idx < len(self.dataset):
                if sequence_weights is not None:
                    self.sequence_weights[idx] = sequence_weights[i]
                if token_weights is not None:
                    self.token_weights[idx] = token_weights[i]
    
    def to_hf_dataset(self):
        data = {
            "input_ids": [self.dataset[idx]["input_ids"] for idx in range(len(self))],
            "labels": [self.dataset[idx]["labels"] for idx in range(len(self))],
            "sequence_weights": self.sequence_weights.tolist(),
            "token_weights": self.token_weights,
        }
        return HuggingFaceDataset.from_dict(data)

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

def create_save_dir(args):
    base_dir = args.base_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = args.data_path.split('/')[-1]
    run_name = f"{dataset_name}__{args.run_name}__tp{args.temperature}_l_{args.loss_type}"
    run_dir = os.path.join(base_dir, dataset_name, f"{timestamp}_{run_name}")
    
    os.makedirs(run_dir, exist_ok=True)
    
    config_dict = vars(args)
    with open(os.path.join(run_dir, "config.json"), 'w') as f:
        json.dump(config_dict, f, indent=4)
    
    return run_dir

def process_batch(batch, model, tokenizer, args):
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
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=IGNORE_INDEX)
        token_losses = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        ).view(shift_labels.size())

        if args.loss_type == "sequence":
            padding_mask = labels.eq(IGNORE_INDEX)
            num_active_elements = (~padding_mask).sum(dim=-1)
            sequence_losses = token_losses.sum(dim=-1) / num_active_elements
            sequence_weights = torch.exp(-sequence_losses/args.temperature)

            # Set to zero if Nan
            if torch.isnan(sequence_weights).any():
                print("Sequence Weights contain NaNs")
                sequence_weights = torch.nan_to_num(sequence_weights, nan=0.0)

            return sequence_weights.cpu().tolist(), None
        elif args.loss_type == "token":
            token_weights = torch.exp(-token_losses/args.temperature)

            # Set to zero if Nan
            if torch.isnan(token_weights).any():
                print("Token Weights contain NaNs")
                token_weights = torch.nan_to_num(token_weights, nan=0.0)
            
            token_weights_list = []
            for weight, mask in zip(token_weights, attention_mask[:, :-1]):
                token_weights_list.append(weight[mask.bool()].cpu().tolist())
            
            return None, token_weights_list
        else: # Assume both
            padding_mask = labels.eq(IGNORE_INDEX)
            num_active_elements = (~padding_mask).sum(dim=-1)
            sequence_losses = token_losses.sum(dim=-1) / num_active_elements
            sequence_weights = torch.exp(-sequence_losses/args.temperature)
            token_weights = torch.exp(-token_losses/args.temperature)

            # Set to zero if Nan
            if torch.isnan(sequence_weights).any():
                print("Sequence Weights contain NaNs")
                sequence_weights = torch.nan_to_num(sequence_weights, nan=0.0)
            if torch.isnan(token_weights).any():
                print("Token Weights contain NaNs")
                token_weights = torch.nan_to_num(token_weights, nan=0.0)

            sequence_weights_list = sequence_weights.cpu().tolist()
            token_weights_list = []
            for weight, mask in zip(token_weights, attention_mask[:, :-1]):
                token_weights_list.append(weight[mask.bool()].cpu().tolist())

            return sequence_weights_list, token_weights_list

def gather_batch(batch, rank, world_size, dst):
    if rank == dst:
        all_batches = [None for _ in range(world_size)]
        dist.gather_object(batch, all_batches, dst=dst)
        return all_batches
    else:
        dist.gather_object(batch, dst=dst)

def main(args):
    rank, world_size = init_distributed()

    if rank == 0:
        run_dir = create_save_dir(args=args)

    model, tokenizer = create_model_tokenizer_it(args)

    model = model.to(args.device)
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    model.eval()

    
    dataset = load_and_preprocess_it(tokenizer=tokenizer, args=args)
    train_dataset = WeightedDataset(dataset)
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    data_collator = IndexedDataCollatorForSupervisedDataset(tokenizer=tokenizer)
    dataloader = DataLoader(
        train_dataset,
        batch_size=args.bs,
        sampler=sampler,
        pin_memory=True,
        collate_fn=data_collator
    )
    
    print(f"Starting Sample rewighting on rank {rank}...\n")
    for batch in tqdm(dataloader, disable=rank != 0):
        sequence_weights, token_weights = process_batch(batch, model, tokenizer, args)

        batch["sequence_weights"] = sequence_weights
        batch["token_weights"] = token_weights

        if world_size > 1:
            batches = gather_batch(batch, rank, world_size, dst=0)
        else:
            batches = [batch]
        
        if rank == 0:
            for bat in batches:
                if args.loss_type == "sequence":
                    train_dataset.set_weights(bat["index"], sequence_weights=bat["sequence_weights"])
                elif args.loss_type == "token":
                    train_dataset.set_weights(bat["index"], token_weights=bat["token_weights"])
                else: # both
                    train_dataset.set_weights(bat["index"], sequence_weights=bat["sequence_weights"], token_weights=bat["token_weights"])
    
    # Synchronize processes
    if world_size > 1:
        dist.barrier()

    if args.validate_dataset == "true" and rank == 0:
        print("Starting Validation of Dataset...\n")
        data_collator = WeightedDataCollatorForSupervisedDataset(tokenizer=tokenizer, loss_type=args.loss_type)
        dataloader = DataLoader(
            train_dataset,
            batch_size=1,
            pin_memory=True,
            collate_fn=data_collator,
        )
        for sample in tqdm(dataloader):
            if args.loss_type in ["sequence", "both"]:
                assert len(sample["sequence_weights"]) == 1, print(sample["sequence_weights"])
            if args.loss_type in ["token", "both"]:
                assert len(sample["token_weights"][0]) == len(sample["input_ids"][0]) - 1
    
    if rank == 0:
        training_dataset = train_dataset.to_hf_dataset()
        training_dataset.save_to_disk(run_dir)

    return None


def parse_args():
    parser = argparse.ArgumentParser(description="A script for weighting the samples of a dataset")

    parser.add_argument("--base-dir", type=str, help="The base directory to store datasets")
    parser.add_argument("--run-name", type=str, help="An additional tag to for the run name")

    parser.add_argument("--data-path", type=str, help="Location to the dataset")
    parser.add_argument("--dataset-split", type=str, default="train[:50000]", help="Dataset split to use")
    parser.add_argument("--dataset-field", type=str, nargs="+", default=["query", "response"], help="Fields of dataset input and output")
    parser.add_argument("--model", type=str, help="The model to use as a reweighter")
    parser.add_argument("--tokenizer", type=str, help="The tokenizer to use as a reweighter")
    parser.add_argument("--temperature", type=float, default=1.0, help="The temperature scaling for reweighting")
    parser.add_argument("--loss-type", type=str, choices=["sequence", "token", "both"], help="The sample reweighting scheme [sequence/token]")

    parser.add_argument("--bs", type=int, default=16, help="Batch Size")
    parser.add_argument("--max-seq-length", type=int, default=1024, help="The maximize sequence length to process samples")
    parser.add_argument("--device", type=str, default="cuda", help="The device to use")

    parser.add_argument("--validate-dataset", type=str, default="false", help="Wether to validate the dataset.")

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    train_dataset = main(args=args)