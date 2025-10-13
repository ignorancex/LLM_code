
# -*- coding: utf-8 -*-
"""
GPT-4-Turbo Translation Tool for WMT Benchmark

This script processes JSONL files containing source texts in the "src" field,
translates them to English using the OpenAI GPT-4-Turbo model, and saves
the results with a new "src_translation" field to Google Drive.

Optimized for Google Colab with efficient batch processing and robust output cleaning.
"""

import os
import gc
import json
import argparse
import logging
import time
import re
from pathlib import Path
from tqdm.notebook import tqdm
from google.colab import drive
import openai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Translate text using GPT-4-Turbo model and save to Google Drive'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Directory containing input JSONL files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory in Google Drive'
    )
    parser.add_argument(
        '--openai_api_key',
        type=str,
        required=True,
        help='OpenAI API key'
    )
    parser.add_argument(
        '--file_batch_size',
        type=int,
        default=1,
        help='Number of files to process before clearing memory (default: 1)'
    )
    parser.add_argument(
        '--translation_batch_size',
        type=int,
        default=8,
        help='Number of translations to process at once (default: 8)'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=128,
        help='Maximum output length (default: 128)'
    )
    return parser.parse_args()

def mount_google_drive():
    """Mount Google Drive to access and save files."""
    try:
        drive.mount('/content/drive')
        logger.info("Google Drive mounted successfully")
    except Exception as e:
        logger.error(f"Failed to mount Google Drive: {e}")
        raise

def initialize_openai_client(api_key):
    """
    Initialize the OpenAI client with the provided API key.

    Args:
        api_key (str): OpenAI API key

    Returns:
        openai.OpenAI: OpenAI client
    """
    logger.info("Initializing OpenAI client...")

    try:
        client = openai.OpenAI(api_key=api_key)
        logger.info("OpenAI client initialized successfully")
        return client
    except Exception as e:
        logger.error(f"Error initializing OpenAI client: {e}")
        raise

def build_prompt(source_text):
    """
    Build the translation prompt according to the specified template.

    Args:
        source_text (str): The source text to translate

    Returns:
        list: List of message dictionaries for the OpenAI API
    """
    system_message = "You are a certified WMT benchmark translator."

    user_message = f"""Translate the following sentence from the WMT22 dataset into English.
Your translation will be directly compared to WMT system outputs using the 'all-mpnet-base-v2' semantic similarity model. To ensure accurate benchmarking, provide exactly one clean English sentence—no alternative translations, explanations, or additional text.
Sentence: {source_text}
Translation:"""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    return messages

def clean_output(output_text):
    """
    Clean the model output to extract just the translated text.

    Args:
        output_text (str): Raw model output

    Returns:
        str: Cleaned translation
    """
    # Extract everything after "Translation:" if it exists
    translation_marker = "Translation:"
    if translation_marker in output_text:
        output_text = output_text.split(translation_marker, 1)[1]

    # Remove any quotes that might be present
    output_text = re.sub(r'^[\s"\']+|[\s"\']+$', '', output_text)

    # More robust translation extraction approaches:

    # 1. Try to detect if the model has added commentary after the translation
    # Look for phrases that might indicate commentary after translation
    commentary_indicators = [
        "\n\nNote:", "\n\nI hope", "I've translated", "\n\nThis translates",
        "The translation is", "Alternative translation:", "Alternatively:"
    ]

    for indicator in commentary_indicators:
        if indicator in output_text:
            output_text = output_text.split(indicator, 1)[0]

    # 2. Handle multi-sentence translations more carefully
    # If there are multiple sentences, don't just cut at the first period
    # First, split by double newlines which often indicate a separation between translation and comments
    if "\n\n" in output_text:
        output_text = output_text.split("\n\n")[0]

    # 3. Remove any trailing single newlines and spaces
    output_text = output_text.rstrip('\n ')

    # 4. If the translation ends with a period and quotes, keep them both
    output_text = re.sub(r'([.!?])"$', r'\1"', output_text)

    # Final cleanup of any remaining whitespace issues
    output_text = output_text.strip()

    return output_text

def get_jsonl_files(input_dir):
    """
    Get list of JSONL files from the input directory.

    Args:
        input_dir (str): Path to input directory

    Returns:
        list: List of paths to JSONL files
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    jsonl_files = list(input_path.glob("*.jsonl"))
    if not jsonl_files:
        logger.warning(f"No JSONL files found in {input_dir}")

    return jsonl_files

def ensure_output_dir(output_dir):
    """
    Ensure the output directory exists, creating it if necessary.

    Args:
        output_dir (str): Path to output directory

    Returns:
        Path: Path object for the output directory
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        logger.info(f"Creating output directory: {output_dir}")
        output_path.mkdir(parents=True, exist_ok=True)
    return output_path

def process_file(file_path, output_path, openai_client, max_length=128, batch_size=8):
    """
    Process a single JSONL file, translating the "src" field of each entry with batched inference.

    Args:
        file_path (Path): Path to input JSONL file
        output_path (Path): Path to output directory
        openai_client: OpenAI client
        max_length (int): Maximum output length
        batch_size (int): Number of examples to process in a single batch

    Returns:
        int: Number of entries processed
    """
    output_file = output_path / file_path.name
    processed_count = 0

    try:
        # Load all data from the file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f if line.strip()]

        logger.info(f"Processing {file_path.name} with {len(data)} entries")

        # Filter entries with 'src' field
        valid_entries = [entry for entry in data if 'src' in entry]
        if len(valid_entries) < len(data):
            logger.warning(f"Skipped {len(data) - len(valid_entries)} entries without 'src' field")

        # Process entries in batches
        for i in range(0, len(valid_entries), batch_size):
            batch_entries = valid_entries[i:i+batch_size]
            batch_responses = []

            # Process each entry in the batch
            for entry in batch_entries:
                messages = build_prompt(entry['src'])

                # Call OpenAI API with exponential backoff for rate limits
                max_retries = 5
                retry_delay = 1
                for attempt in range(max_retries):
                    try:
                        response = openai_client.chat.completions.create(
                            model="gpt-4-turbo",
                            messages=messages,
                            max_tokens=max_length,
                            temperature=0.0,  # Equivalent to do_sample=False
                            n=1,  # Generate a single completion
                        )
                        batch_responses.append(response.choices[0].message.content)
                        break
                    except openai.RateLimitError:
                        if attempt < max_retries - 1:
                            sleep_time = retry_delay * (2 ** attempt)
                            logger.warning(f"Rate limit hit. Retrying in {sleep_time} seconds...")
                            time.sleep(sleep_time)
                        else:
                            logger.error("Rate limit exceeded after maximum retries.")
                            raise
                    except Exception as e:
                        logger.error(f"Error calling OpenAI API: {e}")
                        raise

            # Clean outputs
            batch_translations = [clean_output(text) for text in batch_responses]

            # Add translations to entries
            for entry, translation in zip(batch_entries, batch_translations):
                entry['src_translation'] = translation
                processed_count += 1

            # Update progress
            logger.debug(f"Processed batch {i//batch_size + 1}/{(len(valid_entries) + batch_size - 1)//batch_size}")

        # Save the processed data
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        logger.info(f"Saved translated file to {output_file}")
        return processed_count

    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        raise

def process_files_in_batches(jsonl_files, output_path, openai_client, file_batch_size=1, translation_batch_size=8, max_length=128):
    """
    Process JSONL files in batches to manage memory efficiently.

    Args:
        jsonl_files (list): List of JSONL file paths
        output_path (Path): Path to output directory
        openai_client: OpenAI client
        file_batch_size (int): Number of files to process before clearing memory
        translation_batch_size (int): Number of translations to process at once
        max_length (int): Maximum length for model output

    Returns:
        int: Total number of entries processed
    """
    total_processed = 0

    for i in range(0, len(jsonl_files), file_batch_size):
        batch_files = jsonl_files[i:i+file_batch_size]
        logger.info(f"Processing file batch {i//file_batch_size + 1}/{(len(jsonl_files) + file_batch_size - 1)//file_batch_size}")

        for file_path in batch_files:
            processed = process_file(
                file_path,
                output_path,
                openai_client,
                max_length=max_length,
                batch_size=translation_batch_size
            )
            total_processed += processed

        # Clear memory after each batch of files
        logger.info("Clearing memory between file batches")
        gc.collect()

    return total_processed

def main():
    """Main function to run the translation pipeline."""
    # Parse command line arguments
    args = parse_arguments()

    try:
        logger.info("Starting GPT-4-Turbo translation tool...")
        logger.info(f"Arguments: input_dir={args.input_dir}, output_dir={args.output_dir}, "
                   f"file_batch_size={args.file_batch_size}, translation_batch_size={args.translation_batch_size}, "
                   f"max_length={args.max_length}")

        # Mount Google Drive
        mount_google_drive()

        # Get JSONL files
        jsonl_files = get_jsonl_files(args.input_dir)
        logger.info(f"Found {len(jsonl_files)} JSONL files to process")

        if not jsonl_files:
            logger.error("No JSONL files found. Exiting.")
            return 1

        # Ensure output directory exists
        output_path = ensure_output_dir(args.output_dir)

        # Initialize OpenAI client
        openai_client = initialize_openai_client(args.openai_api_key)

        # Process files in batches
        start_time = time.time()

        total_processed = process_files_in_batches(
            jsonl_files,
            output_path,
            openai_client,
            file_batch_size=args.file_batch_size,
            translation_batch_size=args.translation_batch_size,
            max_length=args.max_length
        )

        elapsed_time = time.time() - start_time
        logger.info(f"Total processing time: {elapsed_time:.2f} seconds")

        if total_processed > 0:
            logger.info(f"Average time per translation: {elapsed_time / total_processed:.4f} seconds")

        logger.info(f"Translation complete. Processed {total_processed} entries across {len(jsonl_files)} files.")

        # Clean up resources
        logger.info("Cleaning up resources...")
        gc.collect()

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

    logger.info("Script completed successfully")
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)

