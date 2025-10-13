#!/usr/bin/env python3
"""
Script to extract DINO features offline for all training images.
This should be run once before training to pre-compute features.
"""

import os
import argparse
from utils.dino_utils import extract_dino_features_offline


def main():
    parser = argparse.ArgumentParser(description='Extract DINO features offline')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Directory containing training images')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save extracted features')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_dir):
        print(f"Error: Image directory {args.image_dir} does not exist")
        return
    
    print(f"Extracting DINO features from {args.image_dir} to {args.output_dir}")
    extract_dino_features_offline(args.image_dir, args.output_dir)
    print("DINO feature extraction completed!")


if __name__ == "__main__":
    main() 