import json
import os
import torch
import torch.distributed as dist
from typing import List, Dict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
import pandas as pd

def setup(rank, world_size):
    """
    Initialize the distributed environment.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """
    Clean up the distributed environment.
    """
    dist.destroy_process_group()

class MedicalDataset(Dataset):
    """Dataset for processing medical dialogues"""
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def description_model_batch(model, tokenizer, prompts: List[str], device) -> List[str]:
    """
    Process a batch of prompts for description generation
    """
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(device)
    
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            outputs = model.module.generate(
                **inputs,
                max_new_tokens=3000,
                temperature=0.01,
                top_p=0.9,
                do_sample=True,
                return_dict_in_generate=True,
                output_scores=False
            )
    
    responses = tokenizer.batch_decode(outputs.sequences[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return [r.strip() for r in responses]

def prepare_prompts(samples: List[Dict], tokenizer) -> List[str]:
    """
    Prepare prompts for batch processing
    """
    batch_prompts = []
    for sample in samples:
        self_report = sample['self_report']
        dialogue_history = ""
        for turn in sample['dialogue']:
            if 'patient_response' in turn and turn['patient_response']:
                dialogue_history += "\n".join(turn['patient_response']) + "\n"
        diagnosis = sample['diagnosis']
        recommendation = sample['recommendation']
        
        prompt = (
            "As a medical assistant, expand the patient's symptom description based on:\n"
            f"- Original self-report: {self_report}\n"
            f"- Dialogue: {dialogue_history}\n"
            f"- Diagnosis: {diagnosis}\n"
            f"- Recommendation: {recommendation}\n"
            "\nRules:\n"
            "1. Summarize the patient's information: Combine the 'Original self-report' and all patient responses from 'dialogue' into a single coherent paragraph. "
            "Include only factual patient statements and exclude the doctor's questions. "
            "If a patient response only makes sense in the context of the doctor's question, infer its meaning based on the context. "
            "Note: If the patient does not have a symptom mentioned in the doctor's question, it should also be stated in the factual summary.\n"
            "2. Based on diagnosis and recommendations, add medical evidence to clearly support symptoms\n"
            "3. Never contradict the patient's original statements\n"
            "4. Keep the language natural and clinical\n"
            "5. Return ONLY the enhanced description."
        )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Enhance the description."}
        ]
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        batch_prompts.append(formatted_prompt)
    
    return batch_prompts

def process_samples_in_worker(rank, world_size, model_path, input_file, output_file, batch_size=16):
    """
    Process samples in a distributed worker with batch processing
    """
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map={"": rank}
    )
    
    model = DDP(model, device_ids=[rank], output_device=rank)
    
    if hasattr(torch, 'compile'):
        model = torch.compile(model)
    
    # Load JSON data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Partition data across GPUs
    samples_per_worker = len(data) // world_size
    start_idx = rank * samples_per_worker
    end_idx = start_idx + samples_per_worker if rank < world_size - 1 else len(data)
    worker_data = data[start_idx:end_idx]
    
    # Process in batches
    local_output_data = []
    
    for batch_start in tqdm(range(0, len(worker_data), batch_size), desc=f"GPU {rank} processing"):
        batch_end = min(batch_start + batch_size, len(worker_data))
        batch_samples = worker_data[batch_start:batch_end]
        
        try:
            # Prepare batch prompts
            batch_prompts = prepare_prompts(batch_samples, tokenizer)
            
            # Generate descriptions in batch
            enhanced_descriptions = description_model_batch(model, tokenizer, batch_prompts, device)
            
            # Update samples with enhanced descriptions
            for i, sample in enumerate(batch_samples):
                sample['enhanced_description'] = enhanced_descriptions[i]
                local_output_data.append(sample)
                
        except Exception as e:
            print(f"Warning: GPU {rank} skipping batch {batch_start}-{batch_end} due to error: {e}")
            for sample in batch_samples:
                sample['enhanced_description'] = ""
                local_output_data.append(sample)
    
    # Save final results for this worker
    final_output_file = output_file.replace(".json", f"_rank{rank}_final.json")
    with open(final_output_file, 'w') as f:
        json.dump(local_output_data, f, indent=4)
    
    # Synchronize workers and gather results
    dist.barrier()
    
    # Combine results only on rank 0
    if rank == 0:
        combined_results = []
        combined_results.extend(local_output_data)
        
        for r in range(1, world_size):
            other_rank_file = output_file.replace(".json", f"_rank{r}_final.json")
            with open(other_rank_file, 'r') as f:
                other_data = json.load(f)
            combined_results.extend(other_data)
        
        with open(output_file, 'w') as f:
            json.dump(combined_results, f, indent=4)
        print(f"Rank 0: Saved combined final results to {output_file}")
    
    cleanup()

def process_dataset(input_file: str, output_file: str, batch_size=16, model_path=None):
    """
    Main function to distribute work across GPUs
    """
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No CUDA devices available")
    
    print(f"Starting distributed processing with {world_size} GPUs, batch size {batch_size}")
    
    mp.spawn(
        process_samples_in_worker,
        args=(world_size, model_path, input_file, output_file, batch_size),
        nprocs=world_size,
        join=True
    )
    
    print(f"Processing complete, final output saved to {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Path to input JSON file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save output JSON file")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing")
    parser.add_argument("--world_size", type=int, default=None, help="Number of GPUs to use (default: all available)")
    parser.add_argument("--model_path", type=str, default="Qwen2.5-7B-Instruct", help="Path to the model")
    args = parser.parse_args()
    
    process_dataset(args.input_file, args.output_file, batch_size=args.batch_size, model_path=args.model_path)