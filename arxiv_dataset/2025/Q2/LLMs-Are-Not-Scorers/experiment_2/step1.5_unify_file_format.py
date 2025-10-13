#!/usr/bin/env python3
"""
TXT to JSONL Converter

This script converts a UTF-8 encoded .txt file into a .jsonl file where each line
becomes a JSON object with the format {"src": "<line_content>"}.

Usage:
    python txt_to_json_converter.py input.txt
    
If no argument is provided, it will use "input.txt" as the default input file.
"""

import sys
import os
import json


def convert_txt_to_jsonl(input_filepath):
    """
    Convert a .txt file to a .jsonl file with the specified format.
    
    Args:
        input_filepath: Path to the input .txt file
        
    Returns:
        Path to the output .jsonl file
    """
    # Create output filename by replacing .txt extension with .jsonl
    base_name = os.path.splitext(input_filepath)[0]
    output_filepath = f"{base_name}.jsonl"
    
    # Read the input file with UTF-8 encoding
    with open(input_filepath, 'r', encoding='utf-8') as input_file:
        lines = input_file.readlines()
    
    # Process each line and convert to JSON objects
    json_objects = []
    for line in lines:
        # Remove trailing newline character but preserve all other content
        line = line.rstrip('\n')
        # Create JSON object
        json_objects.append({"src": line})
    
    # Write JSON objects to output file
    with open(output_filepath, 'w', encoding='utf-8') as output_file:
        for obj in json_objects:
            # Write each object as a separate line (JSONL format)
            output_file.write(json.dumps(obj, ensure_ascii=False) + '\n')
    
    return output_filepath


if __name__ == "__main__":
    # Use command-line argument if provided, otherwise use default
    input_file = sys.argv[1] if len(sys.argv) > 1 else "input.txt"
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    
    output_file = convert_txt_to_jsonl(input_file)
    print(f"Conversion complete. Output saved to: {output_file}")