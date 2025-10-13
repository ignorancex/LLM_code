# -*- coding: utf-8 -*-
"""
DeepSeek-R1 Translation Tool for WMT Benchmark

This script processes JSONL files containing source texts in the "src" field,
translates them to English using the DeepSeek model via DeepSeek API, and saves
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
import requests

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
        description='Translate text using DeepSeek model and save to Google Drive'
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
        '--deepseek_api_key',
        type=str,
        required=True,
        help='DeepSeek API key'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='deepseek-chat',
        help='DeepSeek model name to use (default: deepseek-chat)'
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

def initialize_deepseek_client(api_key):
    """
    Initialize the DeepSeek client with the provided API key.

    Args:
        api_key (str): DeepSeek API key

    Returns:
        dict: DeepSeek client configuration
    """
    logger.info("Initializing DeepSeek client...")

    try:
        # DeepSeek client is just the API key and endpoint information
        client = {
            'api_key': api_key,
            'api_base': 'https://api.deepseek.com/v1',  # DeepSeek API endpoint
        }
        logger.info("DeepSeek client initialized successfully")
        return client
    except Exception as e:
        logger.error(f"Error initializing DeepSeek client: {e}")
        raise

def build_prompt(source_text):
    """
    Build the translation prompt according to the specified template.

    Args:
        source_text (str): The source text to translate

    Returns:
        str: Formatted prompt for DeepSeek API
    """
    prompt = f"""You are a certified WMT benchmark translator. Translate the following sentence from the WMT22 dataset into English.
Your translation will be directly compared to WMT system outputs using the 'all-mpnet-base-v2' semantic similarity model. To ensure accurate benchmarking, provide exactly one clean English sentence—no alternative translations, explanations, or additional text.
Sentence:
{source_text}
Translation:"""

    return prompt

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

def process_file(file_path, output_path, deepseek_client, model_name="deepseek-chat", max_length=128, batch_size=8):
    """
    Process a single JSONL file, translating the "src" field of each entry with batched inference.

    Args:
        file_path (Path): Path to input JSONL file
        output_path (Path): Path to output directory
        deepseek_client: DeepSeek client configuration
        model_name (str): DeepSeek model name to use
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
                prompt = build_prompt(entry['src'])

                # Call DeepSeek API with exponential backoff for rate limits
                max_retries = 5
                retry_delay = 1
                for attempt in range(max_retries):
                    try:
                        # DeepSeek API request
                        headers = {
                            "Authorization": f"Bearer {deepseek_client['api_key']}",
                            "Content-Type": "application/json"
                        }
                        
                        # Fixed payload with correct model name format
                        payload = {
                            "model": model_name,  # Using the model name parameter
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": max_length,
                            "temperature": 0.0,  # Equivalent to do_sample=False
                            "n": 1,  # Generate a single completion
                        }
                        
                        # Optional parameters for specific models if supported
                        if model_name == "deepseek-chat":
                            payload["stream"] = False
                        
                        # Debug information
                        logger.debug(f"Sending request with model: {model_name}")

                        response = requests.post(
                            f"{deepseek_client['api_base']}/chat/completions",
                            headers=headers,
                            json=payload
                        )

                        # Check response status
                        if response.status_code == 200:
                            response_json = response.json()
                            # Extract content based on API response structure
                            if 'choices' in response_json and len(response_json['choices']) > 0 and 'message' in response_json['choices'][0]:
                                batch_responses.append(response_json['choices'][0]['message']['content'])
                            else:
                                logger.warning(f"Unexpected response format: {response_json}")
                                batch_responses.append("")  # Empty response as fallback
                            break
                        elif response.status_code == 429:  # Rate limit
                            if attempt < max_retries - 1:
                                sleep_time = retry_delay * (2 ** attempt)
                                logger.warning(f"Rate limit hit. Retrying in {sleep_time} seconds...")
                                time.sleep(sleep_time)
                            else:
                                logger.error("Rate limit exceeded after maximum retries.")
                                raise Exception(f"Rate limit error: {response.text}")
                        else:
                            # Log detailed error for debugging
                            logger.error(f"API error: {response.status_code} - {response.text}")
                            
                            # Check for specific error types and provide more information
                            if response.status_code == 400:
                                try:
                                    error_json = response.json()
                                    if 'error' in error_json and 'message' in error_json['error']:
                                        error_msg = error_json['error']['message']
                                        if "Model Not Exist" in error_msg:
                                            available_models = "Available models may include: 'deepseek-chat', 'deepseek-coder', etc."
                                            logger.error(f"Model '{model_name}' does not exist. {available_models}")
                                except:
                                    pass
                                
                            if attempt < max_retries - 1:
                                sleep_time = retry_delay * (2 ** attempt)
                                logger.warning(f"API error. Retrying in {sleep_time} seconds...")
                                time.sleep(sleep_time)
                            else:
                                raise Exception(f"API error: {response.status_code} - {response.text}")
                    except Exception as e:
                        logger.error(f"Error calling DeepSeek API: {e}")
                        if attempt < max_retries - 1:
                            sleep_time = retry_delay * (2 ** attempt)
                            logger.warning(f"Retrying in {sleep_time} seconds...")
                            time.sleep(sleep_time)
                        else:
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

def process_files_in_batches(jsonl_files, output_path, deepseek_client, model_name="deepseek-chat", 
                            file_batch_size=1, translation_batch_size=8, max_length=128):
    """
    Process JSONL files in batches to manage memory efficiently.

    Args:
        jsonl_files (list): List of JSONL file paths
        output_path (Path): Path to output directory
        deepseek_client: DeepSeek client configuration
        model_name (str): DeepSeek model name to use
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
                deepseek_client,
                model_name=model_name,
                max_length=max_length,
                batch_size=translation_batch_size
            )
            total_processed += processed

        # Clear memory after each batch of files
        logger.info("Clearing memory between file batches")
        gc.collect()

    return total_processed

def test_model_availability(deepseek_client, model_name):
    """
    Test if the specified model is available through the DeepSeek API.
    
    Args:
        deepseek_client: DeepSeek client configuration
        model_name (str): Model name to test
        
    Returns:
        bool: True if model is available, False otherwise
    """
    logger.info(f"Testing availability of model: {model_name}")
    
    try:
        headers = {
            "Authorization": f"Bearer {deepseek_client['api_key']}",
            "Content-Type": "application/json"
        }
        
        # Simple test payload
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10,
            "temperature": 0.0,
        }
        
        response = requests.post(
            f"{deepseek_client['api_base']}/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            logger.info(f"✓ Model '{model_name}' is available")
            return True
        else:
            logger.warning(f"✗ Model '{model_name}' is not available: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logger.warning(f"Error testing model availability: {e}")
        return False

def list_available_deepseek_models(deepseek_client):
    """
    Attempt to list available models from DeepSeek API.
    
    Args:
        deepseek_client: DeepSeek client configuration
        
    Returns:
        list: List of available model names, or empty list if unavailable
    """
    try:
        headers = {"Authorization": f"Bearer {deepseek_client['api_key']}"}
        
        response = requests.get(
            f"{deepseek_client['api_base']}/models",
            headers=headers
        )
        
        if response.status_code == 200:
            response_json = response.json()
            if 'data' in response_json:
                models = [model['id'] for model in response_json['data']]
                logger.info(f"Available models: {', '.join(models)}")
                return models
        
        logger.warning(f"Could not list models: {response.status_code} - {response.text}")
        return []
            
    except Exception as e:
        logger.warning(f"Error listing models: {e}")
        return []

def check_and_find_model(deepseek_client, preferred_model="deepseek-chat"):
    """
    Check if preferred model is available, and if not, try to find an alternative.
    
    Args:
        deepseek_client: DeepSeek client configuration
        preferred_model (str): Preferred model name
        
    Returns:
        str: Available model name to use
    """
    # Try to list all available models
    all_models = list_available_deepseek_models(deepseek_client)
    
    # Check if preferred model is in the list
    if all_models and preferred_model in all_models:
        return preferred_model
    
    # If we couldn't get a list, or preferred model isn't in it,
    # try testing the preferred model directly
    if test_model_availability(deepseek_client, preferred_model):
        return preferred_model
    
    # If preferred model is not available, try common alternatives
    potential_models = [
        "deepseek-chat",
        "deepseek-llm",
        "deepseek-coder"
    ]
    
    # If we have a list of all models, use those instead
    if all_models:
        potential_models = all_models
    
    # Try each model
    for model in potential_models:
        if model != preferred_model and test_model_availability(deepseek_client, model):
            logger.info(f"Using alternative model: {model}")
            return model
    
    # If no model is available, raise an error
    raise Exception("No available DeepSeek models found. Please check your API key and try again.")

def main():
    """Main function to run the translation pipeline."""
    # Parse command line arguments
    args = parse_arguments()

    try:
        logger.info("Starting DeepSeek Translation Tool...")
        logger.info(f"Arguments: input_dir={args.input_dir}, output_dir={args.output_dir}, "
                   f"model_name={args.model_name}, file_batch_size={args.file_batch_size}, "
                   f"translation_batch_size={args.translation_batch_size}, max_length={args.max_length}")

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

        # Initialize DeepSeek client
        deepseek_client = initialize_deepseek_client(args.deepseek_api_key)
        
        # Check and find available model
        model_to_use = check_and_find_model(deepseek_client, args.model_name)
        logger.info(f"Using model: {model_to_use}")

        # Process files in batches
        start_time = time.time()

        total_processed = process_files_in_batches(
            jsonl_files,
            output_path,
            deepseek_client,
            model_name=model_to_use,
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