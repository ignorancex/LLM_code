import os
import json
import torch
from torchvision import transforms
from PIL import Image
from diffusers import DiffusionPipeline
import argparse
from utils.utils_eval import *
from utils.utils_model import *
from utils.utils import *
from utils.utils_finetuning import *

def main():
    parser = argparse.ArgumentParser(description="Evalaye the decoder of a latent video generation model.")
    parser.add_argument("--config", type=str, default="configs/SVD_Evaluate_decoder.json", help="Path to JSON config file.")
    args = parser.parse_args()

    # Load config file
    with open(args.config, "r") as f:
        config = json.load(f)

    # Parse config as arguments
    for key, value in config.items():
        setattr(args, key, value)

    # Set random seed for reproducibility
    set_seed(args.seed if hasattr(args, "seed") else 0)

    # Set Hugging Face cache directory
    set_hf_cache(args.hf_cache_dir)

    # Ensure output folder exists
    os.makedirs(args.output_folder, exist_ok=True)

    # Load Video Generation Pipeline
    pipeline = load_pipeline(args.model_name, args.checkpoint_path)

    # Generate keys
    keys = load_keys_eval(args)

    # Get Message Decoder
    msg_decoder = load_msg_decoder(args)

    # Generate and Evualte Videos
    Generate_and_Evaluate(pipeline, keys, msg_decoder, args)
    
    print("Video generation completed.")

if __name__ == "__main__":
    main()
