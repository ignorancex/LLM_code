import json
import argparse
import os
from utils.utils import *
from utils.utils_dataset import *
from utils.utils_finetuning import *
from utils.utils_model import *

def main():
    parser = argparse.ArgumentParser(description="Fine-tune the decoder of any Hugging Face latent video generation model.")
    parser.add_argument("--config", type=str, default="configs/SVD_Finetune_Second_Stage_config.json", help="Path to JSON config file.")
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

    # Create the directories
    creat_directories(args)

    # Load the pre-trained model from Hugging Face
    model, original_vae = load_model(args.model_name, args)

    # Create dataloader for training
    train_loader, val_loader = get_dataloaders(args)
    #train_loader, val_loader = None, None

    # Generate or load keys
    keys = get_keys(args)

    # Get Message Decoder
    msg_decoder = load_msg_decoder(args)

    # Train the decoder
    train_decoder(model, original_vae, msg_decoder, train_loader, keys, args)

    # Evaluate the decoder
    val_decoder(model, original_vae, msg_decoder, val_loader, keys, args)


if __name__ == "__main__":
    main()
