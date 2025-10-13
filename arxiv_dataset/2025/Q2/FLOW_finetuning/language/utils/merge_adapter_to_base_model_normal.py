import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch


def get_dtype(dtype_str):
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map[dtype_str]

def main(args):
    
    print(f"Loading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=get_dtype(args.dtype),
        device_map={"": args.device},
        trust_remote_code=True
    )
    
    print(f"Loading LoRA adapter: {args.adapter}")
    model = PeftModel.from_pretrained(
        model,
        args.adapter,
        device_map={"": args.device},
    )
    
    print("Merging LoRA weights with base model...")
    model = model.merge_and_unload()
    
    print(f"Saving merged model to: {args.output_path}")
    model.save_pretrained(
        args.output_path,
        safe_serialization=True,
    )
    
    # # Save tokenizer if it exists
    # try:
    #     tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    #     tokenizer.save_pretrained(args.output_path)
    #     print("Tokenizer saved successfully")
    # except Exception as e:
    #     print(f"Warning: Could not save tokenizer - {str(e)}")
    
    print("Merge completed successfully!")

def parse_args():
    parser = argparse.ArgumentParser(description="Merge a LoRA model with its base model")
    parser.add_argument("--base_model", type=str, required=True, help="Path to the base model or model identifier from huggingface.co/models")
    parser.add_argument("--adapter", type=str, required=True, help="Path to the LoRA adapter to merge")
    parser.add_argument("--output_path", type=str, required=True, help="Path where to save the merged model")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to load the model on")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float16", "float32", "bfloat16"], help="Data type for model weights")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)