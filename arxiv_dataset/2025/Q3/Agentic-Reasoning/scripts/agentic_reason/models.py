import os
from transformers import AutoTokenizer
from vllm import LLM
from utils.remote_llm import RemoteAPILLM
import torch

def get_output_dir(args, dataset_name, max_search_limit=5, top_k=10):
    """Determine output directory based on model and dataset configuration"""
    if args.remote_model:
        output_dir = f'./outputs/runs.remote/{dataset_name}.{args.remote_model}.agentic_reasoning'
    elif args.model_path and 'qwq' in args.model_path.lower():
        if dataset_name in ['aime', 'amc', 'livecode']:
            output_dir = f'./outputs/{dataset_name}.qwq.agentic_reasoning'
            if dataset_name == 'gpqa' and (max_search_limit != 5 or top_k != 10):
                output_dir = f'./outputs/runs.analysis/{dataset_name}.qwq.agentic_reasoning.{max_search_limit}.{top_k}'
        else:
            output_dir = f'./outputs/runs.qa/{dataset_name}.qwq.agentic_reasoning'
    else:
        # For local models without qwq
        model_short_name = args.model_path.split('/')[-1].lower().replace('-instruct', '') if args.model_path else 'default'
        output_dir = f'./outputs/runs.baselines/{dataset_name}.{model_short_name}.agentic_reasoning'
    
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def initialize_model(args):
    """Initialize the model and tokenizer based on arguments"""
    if not args.remote_model:
        if not args.model_path:
            raise ValueError("model_path is required when not using a remote model")
        if not os.path.exists(args.model_path):
            raise ValueError(f"Local model path does not exist: {args.model_path}")
            
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        
        llm = LLM(
            model=args.model_path,
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=0.95,
        )
    else:
        # Remote model initialization
        if args.remote_model == 'gpt-4o':
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
        elif args.remote_model == 'claude-3.5-sonnet':
            tokenizer = AutoTokenizer.from_pretrained("anthropic/claude-3-sonnet")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        
        llm = RemoteAPILLM(model_name=args.remote_model)
    
    return llm, tokenizer