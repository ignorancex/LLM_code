import argparse
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import torch
from datasets import load_dataset  # Import the datasets library
from tqdm import tqdm  # Import tqdm for progress visualization
from torch.utils.data import DataLoader  # Import DataLoader for batching

# Enable CuDNN benchmarking for optimized performance
torch.backends.cudnn.benchmark = True

SCRATCH_DIR = os.path.join("/scratch-shared", os.getenv("USER"))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Set the GPU number. If inference with multiple GPUs, set multiple GPU numbers
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the system prompt for text-to-video (t2v) conversion
sys_prompt = """
You are part of a team of bots with two tasks:
1. Obfuscating sensitive information in text while preserving its overall structure and meaning
2. Creating alternative versions by replacing sensitive information with similar but different entities

For each input text, you should provide TWO outputs separated by [SEP]:
- First output: Masked version using placeholders (e.g., <NAME>, <LOCATION>)
- Second output: Alternative version replacing sensitive entities with different but contextually similar entities

Rules for masking:
1. Replace sensitive entities (names, locations, organizations) with appropriate placeholders
2. Use format like <NAME>, <LOCATION>, <YEAR>, <ENTITY>
3. Maintain grammatical correctness and original meaning
4. Use consistent placeholders for identical entities
5. If no sensitive entities exist, output unchanged text

Rules for alternative version:
1. Replace sensitive entities with different but plausible alternatives
2. Maintain the same grammatical structure and coherence
3. Ensure replacements are of the same category (e.g., replace person with person, city with city)
4. The alternative should be semantically valid but change the meaning

Example 1:
Input: "Barack Obama visited the United Nations headquarters in New York"
Masked version: <PERSON> visited the United Nations headquarters in <LOCATION>[SEP]Alternative version: Lebron James visited the United Nations headquarters in Los Angeles

Example 2:
Input: "Elon Musk is the CEO of SpaceX, based in Hawthorne, California"
Masked version: <PERSON> is the CEO of <ORGANIZATION>, based in <LOCATION>[SEP]Alternative version: Sam Altman is the CEO of OpenAI, based in San Francisco

Example 3:
Input: "The CEO of Tesla, Elon Musk, met with the President of the United States"
Masked version: The CEO of [[ORGANIZATION]], [[PERSON]], met with the President of the United States[SEP]Alternative version: The CEO of Meta, Mark Zuckerberg, met with the Prime Minister of the United Kingdom

Please respond with: Masked version[SEP]Alternative version.
"""

def mask_passage_batch(prompts: list, model, tokenizer, do_sample: bool, max_length: int, temperature: float = 0.7, top_p: float = 0.9):
    # Prepare batched inputs
    messages = [
        [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f'Please provide both masked and alternative versions for the following text: "{prompt}"'}
        ] for prompt in prompts
    ]
    
    # Vectorized tokenization
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors=None,  # Avoid returning PyTorch tensors for compatibility
        return_dict=True
    )
    
    inputs_list = inputs.to(device)
    
    # Pad the inputs to ensure consistent dimensions
    padded_inputs = tokenizer.pad(
        {"input_ids": inputs_list["input_ids"], "attention_mask": inputs_list["attention_mask"]},
        padding=True,  # Add padding to the shorter sequences
        return_tensors="pt"  # Convert the padded sequences to PyTorch tensors
    ).to(device)
    
    # Set generation parameters
    gen_kwargs = {
        "max_length": max_length,
        "do_sample": do_sample,
        "top_k": 1,
    }
    with torch.no_grad():
        outputs = model.generate(
            input_ids=padded_inputs['input_ids'],
            attention_mask=padded_inputs['attention_mask'],
            **gen_kwargs
        )
        
        generated_txts = tokenizer.batch_decode(outputs[:, padded_inputs['input_ids'].shape[1]+1:], skip_special_tokens=True)
        
        # Split the outputs into masked and alternative versions
        processed_outputs = []
        for txt in generated_txts:
            versions = txt.split('[SEP]')
            if len(versions) == 2:
                processed_outputs.append({
                    'masked': versions[0].strip().replace('Masked version:', '').strip().replace('\n', ''),
                    'alternative': versions[1].strip().replace('Alternative version:', '').strip().replace('\n', '')
                })
            else:
                # Handle case where [SEP] is not present but "Alternative version:" exists
                versions = txt.split('Alternative version:')
                if len(versions) == 2:
                    processed_outputs.append({
                        'masked': versions[0].strip().replace('Masked version:', '').strip().replace('\n', ''),
                        'alternative': versions[1].strip().replace('\n', ' ')
                    })
                else:
                    # If neither delimiter is found, add the entire text as masked version
                    processed_outputs.append({
                        'masked': txt.strip().replace('\n', ''),
                        'alternative': ''  # Empty string for alternative if not found
                    })
            
            
    return processed_outputs

if __name__ == "__main__":
    # Argument parser for user input
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='sentence-transformers/altlex', help="Name of the dataset to use")
    parser.add_argument("--model_path", type=str, default='GLM-4/glm-4-9b-chat', help="Path to the pre-trained Llama-3 model")
    parser.add_argument("--max_length", type=int, default=2500, help="Maximum length of the generated description")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for processing")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling parameter")
    parser.add_argument("--num_iterations", type=int, default=40, help="Number of iterations to process")
    args = parser.parse_args()

    # Load the dataset
    dataset = load_dataset(args.dataset_name)  # Load the altlex dataset

    # Load the model and tokenizer
    model_path = os.path.join(SCRATCH_DIR, args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto"
    ).eval()
    
    # Create a DataLoader for batching with increased workers and prefetching
    batch_size = args.batch_size  # Adjust batch size as needed
    data_loader = DataLoader(
        dataset['train'], 
        batch_size=batch_size, 
        shuffle=False, 
        pin_memory=True, 
        num_workers=4,
    )

    # Open the log file for writing outside the loop
    log_file = f'{batch_size}_{args.num_iterations}_second_half_iterations.jsonl'
    with open(log_file, 'w', encoding='utf-8') as f:
        # Iterate over the dataset in batches
        for idx, batch in tqdm(enumerate(data_loader), desc="Processing batches"):
            if idx > args.num_iterations and idx < 2 * args.num_iterations:
                if idx % 5 == 0 and idx != 0: 
                    torch.cuda.empty_cache()  # Clear cache every five iterations
                
                prompts = [sentence for sentence in batch['simplified']]  # Get the original texts
                processed_outputs = mask_passage_batch(
                    prompts, 
                    model, 
                    tokenizer, 
                    do_sample=True, 
                    max_length=args.max_length, 
                    temperature=args.temperature, 
                    top_p=args.top_p
                )

                # Log the original and generated texts in JSON format
                for original_txt, processed in zip(batch['simplified'], processed_outputs):
                    log_entry = {
                        "original": original_txt,
                        "masked": processed['masked'],
                        "alternative": processed['alternative']
                    }
                    json.dump(log_entry, f, ensure_ascii=False)
                    f.write('\n')  # Write a newline for each entry to separate them
            if idx >= 2 * args.num_iterations:
                break

# Example usage
# python LLM_expert_masking.py