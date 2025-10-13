import argparse
import sys


def parse_arguments():
    """Parse command line arguments with strict model-specific validation."""
    parser = argparse.ArgumentParser(
        description='Text translation tool with model-specific parameters',
        add_help=False  # We'll handle help manually to show model-specific usage
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['gemma7b', 'llama7b', 'llama8b', 'openchat3_5',
                 'qwen8b', 'qwen14b', 'deepseekr1', 'gpt4'],
        help='Name of the model to use'
    )

    parser.add_argument('--input_dir', type=str, help='Input directory path')
    parser.add_argument('--output_dir', type=str, help='Output directory path')
    parser.add_argument('--hf_token', type=str,
                        help='Hugging Face access token')
    parser.add_argument('--deepseek_api_key', type=str,
                        help='DeepSeek API key')
    parser.add_argument('--openai_api_key', type=str, help='OpenAI API key')
    parser.add_argument('--file_batch_size', type=int, help='Files per batch')
    parser.add_argument('--translation_batch_size',
                        type=int, help='Translations per batch')
    parser.add_argument('--max_length', type=int, help='Maximum output length')
    parser.add_argument('--temperature', type=float,
                        help='Temperature for sampling')

    # Custom help message handling
    parser.add_argument('-h', '--help', action='store_true',
                        help='Show help message')

    args, unknown_args = parser.parse_known_args()

    # Handle help requests
    if args.help:
        show_model_specific_help(args.model)
        sys.exit(0)

    # Check for unknown parameters
    if unknown_args:
        print(f"Error: Unknown parameters {unknown_args}")
        show_model_specific_help(args.model)
        sys.exit(1)

    # Model-specific parameter validation
    validate_model_args(args)

    return args


def validate_model_args(args):
    """Validate arguments based on the selected model."""
    required = {
        'gemma7b': ['input_dir', 'output_dir', 'hf_token'],
        'llama7b': ['input_dir', 'output_dir', 'hf_token'],
        'llama8b': ['input_dir', 'output_dir', 'hf_token'],
        'openchat3_5': ['input_dir', 'output_dir', 'hf_token'],
        'qwen8b': ['input_dir', 'output_dir', 'hf_token'],
        'qwen14b': ['input_dir', 'output_dir', 'hf_token'],
        'deepseekr1': ['input_dir', 'output_dir', 'deepseek_api_key'],
        'gpt4': ['input_dir', 'output_dir', 'openai_api_key']
    }

    allowed = {
        'gemma7b': ['input_dir', 'output_dir', 'hf_token', 'file_batch_size',
                    'translation_batch_size', 'max_length'],
        'llama7b': ['input_dir', 'output_dir', 'hf_token', 'file_batch_size',
                    'translation_batch_size', 'temperature'],
        'llama8b': ['input_dir', 'output_dir', 'hf_token', 'file_batch_size',
                    'translation_batch_size', 'max_length'],
        'openchat3_5': ['input_dir', 'output_dir', 'hf_token', 'file_batch_size',
                        'translation_batch_size', 'max_length'],
        'qwen8b': ['input_dir', 'output_dir', 'hf_token', 'file_batch_size',
                   'translation_batch_size', 'max_length'],
        'qwen14b': ['input_dir', 'output_dir', 'hf_token', 'file_batch_size',
                    'translation_batch_size', 'max_length'],
        'deepseekr1': ['input_dir', 'output_dir', 'deepseek_api_key', 'file_batch_size',
                       'translation_batch_size', 'max_length'],
        'gpt4': ['input_dir', 'output_dir', 'openai_api_key', 'file_batch_size',
                 'translation_batch_size', 'max_length']
    }

    # Check required parameters
    for param in required[args.model]:
        if getattr(args, param) is None:
            print(f"Error: --{param} is a required parameter")
            show_model_specific_help(args.model)
            sys.exit(1)

    # Check for invalid parameters
    allowed_params = allowed[args.model] + ['model']
    for param in vars(args):
        if param in ['help', '_unknown_args']:
            continue
        if getattr(args, param) is not None and param not in allowed_params:
            print(f"Error: Model {args.model} does not support parameter --{param}")
            show_model_specific_help(args.model)
            sys.exit(1)


def show_model_specific_help(model):
    """Display help message specific to the selected model."""
    help_msgs = {
        'gemma7b': """
Usage: python model_translation.py --model gemma7b \\
    --input_dir INPUT_DIR \\
    --output_dir OUTPUT_DIR \\
    --hf_token HF_TOKEN \\
    [--file_batch_size FILE_BATCH_SIZE] \\
    [--translation_batch_size TRANSLATION_BATCH_SIZE] \\
    [--max_length MAX_LENGTH]
""",
        'llama7b': """
Usage: python model_translation.py --model llama7b \\
    --input_dir INPUT_DIR \\
    --output_dir OUTPUT_DIR \\
    --hf_token HF_TOKEN \\
    [--file_batch_size FILE_BATCH_SIZE] \\
    [--translation_batch_size TRANSLATION_BATCH_SIZE] \\
    [--temperature TEMPERATURE]
""",
        'llama8b': """
Usage: python model_translation.py --model llama8b \\
    --input_dir INPUT_DIR \\
    --output_dir OUTPUT_DIR \\
    --hf_token HF_TOKEN \\
    [--file_batch_size FILE_BATCH_SIZE] \\
    [--translation_batch_size TRANSLATION_BATCH_SIZE] \\
    [--max_length MAX_LENGTH]
""",
        'openchat3_5': """
Usage: python model_translation.py --model openchat3_5 \\
    --input_dir INPUT_DIR \\
    --output_dir OUTPUT_DIR \\
    --hf_token HF_TOKEN \\
    [--file_batch_size FILE_BATCH_SIZE] \\
    [--translation_batch_size TRANSLATION_BATCH_SIZE] \\
    [--max_length MAX_LENGTH]
""",
        'qwen8b': """
Usage: python model_translation.py --model qwen8b \\
    --input_dir INPUT_DIR \\
    --output_dir OUTPUT_DIR \\
    --hf_token HF_TOKEN \\
    [--file_batch_size FILE_BATCH_SIZE] \\
    [--translation_batch_size TRANSLATION_BATCH_SIZE] \\
    [--max_length MAX_LENGTH]
""",
        'qwen14b': """
Usage: python model_translation.py --model qwen14b \\
    --input_dir INPUT_DIR \\
    --output_dir OUTPUT_DIR \\
    --hf_token HF_TOKEN \\
    [--file_batch_size FILE_BATCH_SIZE] \\
    [--translation_batch_size TRANSLATION_BATCH_SIZE] \\
    [--max_length MAX_LENGTH]
""",
        'deepseekr1': """
Usage: python model_translation.py --model deepseekr1 \\
    --input_dir INPUT_DIR \\
    --output_dir OUTPUT_DIR \\
    --hf_token HF_TOKEN \\
    [--file_batch_size FILE_BATCH_SIZE] \\
    [--translation_batch_size TRANSLATION_BATCH_SIZE] \\
    [--max_length MAX_LENGTH]
""",
    }

    print(help_msgs.get(model, "Invalid model name"))


def main():
    import gemma7b as gemma7b
    import llama7b as llama7b
    import llama8b as llama8b
    import openchat3_5 as openchat3_5
    import qwen14b as qwen14b
    try:
        args = parse_arguments()
        print("Parameter validation passed:")
        print(vars(args))
        # Set default values
        if args.file_batch_size is None:
            args.file_batch_size = 1
        if args.translation_batch_size is None:
            args.translation_batch_size = 8
        if args.max_length is None:
            args.max_length = 128
        if args.temperature is None:
            args.temperature = 0.0
        # Call the corresponding model function
        if args.model == 'gemma7b':
            gemma7b.main(args.input_dir, args.output_dir, args.hf_token,
                         args.file_batch_size, args.translation_batch_size, args.max_length)
        elif args.model == 'llama7b':
            llama7b.main(args.input_dir, args.output_dir, args.hf_token,
                         args.file_batch_size, args.translation_batch_size, args.max_length)
        elif args.model == 'llama8b':
            llama8b.main(args.input_dir, args.output_dir, args.hf_token,
                         args.file_batch_size, args.translation_batch_size, args.max_length)
        elif args.model == 'openchat3_5':
            openchat3_5.main(args.input_dir, args.output_dir, args.hf_token,
                             args.file_batch_size, args.translation_batch_size, args.max_length)
        elif args.model == 'qwen14b':
            qwen14b.main(args.input_dir, args.output_dir, args.hf_token,
                         args.file_batch_size, args.translation_batch_size, args.max_length)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
