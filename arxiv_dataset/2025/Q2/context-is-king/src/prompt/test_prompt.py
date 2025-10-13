# %% Imports

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from glob import glob
from tqdm import tqdm
import traceback

import numpy as np
import pandas as pd
from scipy import stats

import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


CLASS_NAMES = ['4ASK', '4PAM', '8ASK', '16PAM', 'CPFSK', 'DQPSK', 'GFSK', 'GMSK', 'OQPSK', 'OOK']

def load_npy_file(file_path):
    """Load the npy file from the given path."""
    return np.load(file_path,)

def get_symbol_index(data):
    """Generate a symbol index array based on data length."""
    return np.arange(data.shape[0])

def extract_signal_features(data):
    """
    Extract I and Q components along with magnitude and phase.
    
    Returns I, Q, magnitude, phase as float16 arrays.
    """
    I = data.real.astype(np.float16)
    Q = data.imag.astype(np.float16)
    magnitude = np.abs(data).astype(np.float16)
    phase = np.angle(data, deg=True).astype(np.float16)
    return I, Q, magnitude, phase

def create_structured_table(symbol_idx, I, Q, magnitude, phase):
    """
    Combine signal data into a structured numpy array table.
    
    Returns a structured array with fields:
      - 'Symbol Index'
      - 'I (Real)'
      - 'Q (Imaginary)'
      - 'Magnitude'
      - 'Phase (°)'
    """
    table = np.zeros(symbol_idx.shape[0], dtype=[
        ('Symbol Index', np.int32),
        ('I (Real)', np.float16),
        ('Q (Imaginary)', np.float16),
        ('Magnitude', np.float16),
        ('Phase (°)', np.float16)
    ])
    table['Symbol Index'] = symbol_idx
    table['I (Real)'] = I
    table['Q (Imaginary)'] = Q
    table['Magnitude'] = magnitude
    table['Phase (°)'] = phase
    return table

def create_dataframe(table):
    """Convert a structured numpy array into a pandas DataFrame."""
    return pd.DataFrame(table)

def compute_overall_summary(df_table):
    """
    Compute overall min, max, mean, and standard deviation for
    I (Real), Q (Imaginary), Magnitude, and Phase.
    """
    summary = (
        df_table[['I (Real)', 'Q (Imaginary)', 'Magnitude', 'Phase (°)']]
        .agg(['min', 'max', 'mean', 'std'])
        .T
        .reset_index()
    )
    summary.columns = ['Parameter', 'Min', 'Max', 'Mean', 'Std Dev']
    return summary

def compute_group_summary(df_table, k=100):
    """
    Compute group statistics (min, max, mean, std) for every k indices.
    
    Returns the summary in long format.
    """
    df_table['Group'] = df_table['Symbol Index'] // k
    group_sum = (
        df_table.groupby('Group')[['I (Real)', 'Q (Imaginary)', 'Magnitude', 'Phase (°)']]
        .agg(['min', 'max', 'mean', 'std'])
    )
    # Fix the stacking and column names
    group_sum = group_sum.stack(level=0, future_stack=True)
    group_sum = group_sum.reset_index()
    group_sum.columns = ['Group', 'Parameter', 'min', 'max', 'mean', 'std']
    group_sum = group_sum.rename(columns={
        'min': 'Min',
        'max': 'Max',
        'mean': 'Mean',
        'std': 'Std Dev'
    })
    return group_sum

def table_to_dict_list(table):
    """
    Convert a structured numpy array table or a pandas DataFrame into a list of dictionaries,
    where each dictionary represents a row with keys as column names and values as cell data.
    """
    if hasattr(table, 'dtype') and table.dtype.names is not None:
        col_names = table.dtype.names
        return [{col: row[col] for col in col_names} for row in table]
    elif isinstance(table, pd.DataFrame):
        return table.to_dict(orient="records")
    else:
        raise ValueError("Unsupported table format")
    
def clean_response(response):
    """
    Clean the response string by removing unwanted characters.
    """
    response = re.sub(r'<[^>]*>', '', response)
    # response = re.sub(r'\s+', '', response)
    return response.replace("'", "").replace('"', '')

def get_signal_summary(file_path, div=10):
    data = load_npy_file(file_path).astype(np.complex128)

    label = file_path.split('/')[-1].split('_')[0]

    k = data.shape[0] // div
    # print(k<data.shape[0]*div+1)

    symbol_idx = get_symbol_index(data)
    I, Q, magnitude, phase = extract_signal_features(data)
    table = create_structured_table(symbol_idx, I, Q, magnitude, phase)
    df_table = create_dataframe(table)

    overall = compute_overall_summary(df_table)
    group = compute_group_summary(df_table, k=k)

    return overall, group, data.shape, k, label

def load_model_and_tokenizer(model_id, cache_dir="../../models", local_files_only=True):
    """Load model and tokenizer with error handling."""
    try:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            local_files_only=local_files_only
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            cache_dir=cache_dir,
            local_files_only=local_files_only
        )
        model.eval()
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def process_batch(model, tokenizer, inputs, file_batch, results):
    """Process a batch of prompts and update results."""
    try:
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
        
        decoded = []
        for out in outputs:
            full_text = tokenizer.decode(out, max_new_tokens=8, pad_token_id=tokenizer.eos_token_id, temperature=0.2)
            generated = full_text[full_text.find('### Response: ### Answer:') + len('### Response: ### Answer:'):].strip()
            # print(generated)
            decoded.append(clean_response(generated))
            
        for fpath, text in zip(file_batch, decoded):
            results['file_path'].append(fpath)
            results['modulation_type'].append(fpath.split('/')[-1].split('_')[0])
            results['response'].append(text)
            
    except Exception as e:
        print(f"Error processing batch: {str(e)}")
        # Add failed files to results with error indicator
        for fpath in file_batch:
            results['file_path'].append(fpath)
            results['modulation_type'].append(fpath.split('/')[-1].split('_')[0])
            results['response'].append("ERROR")
    
    finally:
        torch.cuda.empty_cache()

# Get the summary statistics
def get_stats_context(file_path):
    data = load_npy_file(file_path)
    label = file_path.split('/')[-1].split('_')[0]
    des_stats = stats.describe(data)
    stats_summary = {
        'nobs': f'{des_stats.nobs}',
        'min': f"{des_stats.minmax[0]:.5f}",
        'max': f"{des_stats.minmax[1]:.5f}",
        'mean': f"{des_stats.mean:.5f}",
        'variance': f"{des_stats.variance:.5f}",
        'skewness': f"{des_stats.skewness:.5f}",
        'kurtosis': f"{des_stats.kurtosis:.5f}",
    }
    for i in range(10):
        stats_summary[f'moment_{i}'] = f'{stats.moment(data, order=i):.5f}'

    for i in range(1, 5):
        stats_summary[f'kstat_{i}'] = f'{stats.kstat(data, i):.5f}'

    for i in range(1, 3):
        stats_summary[f'kstatvar_{i}'] = f'{stats.kstatvar(data, i):.5f}'


    return str(stats_summary).replace("'", ''), label

def get_example_paths(noise_type='noisySignal'):
    example_paths = {
        '4ASK': f'../../data/unlabeled_10k/test/{noise_type}/4ASK_-0.17dB__081_20250127_164342.npy',
        '4PAM': f'../../data/unlabeled_10k/test/{noise_type}/4PAM_-0.00dB__031_20250127_164618.npy',
        '8ASK': f'../../data/unlabeled_10k/test/{noise_type}/8ASK_-0.11dB__016_20250127_164352.npy',
        '16PAM': f'../../data/unlabeled_10k/test/{noise_type}/16PAM_-0.08dB__058_20250127_145951.npy',
        'CPFSK': f'../../data/unlabeled_10k/test/{noise_type}/CPFSK_-0.03dB__088_20250127_164523.npy',
        'DQPSK': f'../../data/unlabeled_10k/test/{noise_type}/DQPSK_-0.01dB__036_20250127_164655.npy',
        'GFSK': f'../../data/unlabeled_10k/test/{noise_type}/GFSK_-0.05dB__042_20250127_164545.npy', 
        'GMSK': f'../../data/unlabeled_10k/test/{noise_type}/GMSK_-0.12dB__059_20250127_164925.npy',
        'OQPSK': f'../../data/unlabeled_10k/test/{noise_type}/OQPSK_-0.24dB__006_20250127_145655.npy',
        'OOK': f'../../data/unlabeled_10k/test/{noise_type}/OOK_-0.17dB__091_20250127_164311.npy'
    }
    # Ensure the example paths match the class names
    assert set(example_paths.keys()) == set(CLASS_NAMES), "Example paths do not match class names"

    return example_paths

def create_stats_prompt(file_path, noise_type='noisySignal'):
    stats_summary, label = get_stats_context(file_path)
    example_paths = get_example_paths(noise_type=noise_type)


    input_text = (
    f"Overall Signal Information: {stats_summary}"
    )
    
    prompt_prefix = f"""### Instructions
    You are an expert quantitative analyst in wireless communication modulation.
    Based on your knowledge in wireless communication modulation and the detailed signal statistics provided below, determine the modulation type."""

    prompt_examples = ""
    for idx, ex_path in enumerate(example_paths.values()):
        ex_summary, ex_label = get_stats_context(ex_path)
        prompt_examples += f"""
    ### Example {idx+1}: Overall Signal Information: {ex_summary} ### Answer {idx+1}: {ex_label}"""

    prompt_suffix = f"""
    YOUR ANSWER MUST BE STRICTLY ONE AND ONLY ONE OF {", ".join(CLASS_NAMES)} MODULATION TYPES AND NOTHING ELSE. DO NOT PROVIDE ANY ADDITIONAL INFORMATION OR CONTEXT. No OTHER TEXT, NO BLABBER. DO NOT PROVIDE ANY ADDITIONAL INFORMATION OR CONTEXT. No OTHER TEXT, NO BLABBER. What is the modulation type of this signal information: {input_text}
    ### Response: ### Answer:"""

    prompt = prompt_prefix + prompt_examples + prompt_suffix

    return prompt, label

def create_no_context_prompt(file_path):
    stats_summary, label = get_stats_context(file_path)

    input_text = (
    f"Overall Signal Information: {stats_summary}"
    )
    
    prompt_prefix = f"""### Instructions
    You are an expert quantitative analyst in wireless communication modulation.
    Based on your knowledge in wireless communication modulation and the detailed signal statistics provided below, determine the modulation type.
    Your answer must be strictly ONE and ONLY ONE of these modulation types (e.g., 4ASK, 4PAM, 8ASK, 16PAM, CPFSK, DQPSK, GFSK, GMSK, OQPSK, OOK) and nothing else. DO NOT PROVIDE ANY ADDITIONAL INFORMATION OR CONTEXT. No OTHER TEXT, NO BLABBER. What is the modulation type of this signal information: {input_text}
    ### Response: ### Answer:"""

    prompt = prompt_prefix

    return prompt, label

def create_in_context_prompt(file_path, div, noise_type='noisySignal'):
    input_overall, input_group, input_shape, input_k, input_label = get_signal_summary(file_path, div=div)

    example_paths = get_example_paths(noise_type=noise_type)

    input_text = (
        f"Overall Signal Information: {table_to_dict_list(input_overall)}\n"
        f"Every {input_k}-step Signal Summary Information: {table_to_dict_list(input_group)}\n"
        f"Shape: {input_shape}, "
    )

    prompt_prefix = f"""
    ### Instructions
    You are an expert in wireless communication signal analysis.
    Based solely on the detailed signal summaries provided below, determine the modulation type.
    Focus exclusively on the quantitative information—both the overall signal statistics and the summary for every k-step.
    Your answer must be strictly one of these modulation types (e.g., 4ASK, 4PAM, 8ASK, 16PAM, CPFSK, DQPSK, GFSK, GMSK, QPSK, OQPSK, OOK) and nothing else.

    ### Examples:
    """
    prompt_examples = ""
    for idx, ex_path in enumerate(example_paths.values()):
        ex_overall, ex_group, ex_shape, ex_k, ex_label = get_signal_summary(ex_path, div=div)
        prompt_examples += f"""
    Example {idx+1}:
    Overall Signal Information: {table_to_dict_list(ex_overall)}
    Every {ex_k}-step Signal Summary Information: {table_to_dict_list(ex_group)}
    Shape: {ex_shape}
    Modulation type: {ex_label}
    """
    prompt_suffix = f"""

    What is the modulation type of this signal information: {input_text}
    No other text, no blabber.
    ### Response:The modulation type is 
    """

    prompt = prompt_prefix + prompt_examples + prompt_suffix

    return prompt, input_label

def extract_valid_modulation(response_text, valid_modulations=None):
    """Extract valid modulation type from response text or return None."""
    if valid_modulations is None:
        valid_modulations = CLASS_NAMES
    
    # Convert response to uppercase for case-insensitive matching
    response_upper = response_text.upper()
    
    # Check if any valid modulation type exists in the response
    for mod in valid_modulations:
        if mod in response_upper:
            return mod
    return None

def test_model(model_name, model, tokenizer, test_type="no_context", noise_type='noisySignal'):
    """
    Test model with different prompt types and noise settings.
    
    Args:
        model_name: Name of the model being tested
        model: The loaded model 
        tokenizer: The tokenizer
        test_type: Type of test - "no_context", "in_context", or "stats_context"
        noise_type: Type of noise - "noisySignal" or "noiselessSignal"
    """
    try:
        # Set up paths and parameters
        example_paths = get_example_paths(noise_type=noise_type)
        if test_type == "no_context":
            module_signals = glob(f"../../data/unlabeled_10k/test/{noise_type}/*.npy")
            create_prompt_fn = lambda x: create_no_context_prompt(x)[0]
        else:
            module_signals = [x for x in glob(f"../../data/unlabeled_10k/test/{noise_type}/*.npy") 
                            if x not in example_paths.values()]
            create_prompt_fn = (lambda x: create_stats_prompt(x, noise_type)[0] 
                              if test_type == "stats_context" 
                              else lambda x: create_in_context_prompt(x, 3, noise_type)[0])

        results = {'file_path': [], 'modulation_type': [], 'response': []}
        batch_size = 4 if test_type != "in_context" else 1

        # Create and process prompts
        print(f"Creating prompts for {noise_type}...")
        prompts = [create_prompt_fn(file_path) for file_path in tqdm(module_signals[:1000])]
        
        print("Encoding prompts...")
        prompts_encodings = tokenizer(prompts, return_tensors="pt", padding=True)

        print("Generating responses...")
        for i in tqdm(range(0, len(prompts_encodings['input_ids']), batch_size)):
            inputs = {
                'input_ids': prompts_encodings['input_ids'][i:i+batch_size].cuda(),
                'attention_mask': prompts_encodings['attention_mask'][i:i+batch_size].cuda()
            }
        
            file_batch = module_signals[i:i+batch_size]
            process_batch(model, tokenizer, inputs, file_batch, results)
            torch.cuda.empty_cache()

        # Process results
        results_df = pd.DataFrame(results)
        results_df['pred'] = results_df['response'].apply(extract_valid_modulation)
        results_df['is_true'] = results_df['pred'] == results_df['modulation_type']
            
        # Save results
        filename = f'{model_name.replace("/","_")}_{test_type}_{noise_type}_results.csv'
        results_df.to_csv(filename, index=False)
        return results_df

    except Exception as e:
        print(f"Fatal error in test_{test_type}: {str(e)}\n{traceback.format_exc()}")
        results_df = pd.DataFrame(results)
        results_df['pred'] = results_df['response'].apply(extract_valid_modulation)
        results_df['is_true'] = results_df['pred'] == results_df['modulation_type']
        filename = f'{model_name.replace("/","_")}_{test_type}_{noise_type}_results.csv'
        results_df.to_csv(filename, index=False)
        raise

if __name__ == "__main__":
    try:
        print("loading model...")
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        model_name = 'model_traning_outputs/checkpoint-2147'

        for model_name in [
            'model_traning_outputs/checkpoint-2147',
            # "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            #"deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", 
                           ]:
            print(f"Loading model: {model_name}")

            model, tokenizer = load_model_and_tokenizer(model_name)

            for noise_type in ['noisySignal', 'noiselessSignal']:
                print(f"\nProcessing No-Context with {noise_type}...")
                test_model(model_name, model, tokenizer, test_type="no_context", noise_type=noise_type)
                
                # print(f"\nProcessing In-Context with {noise_type}...")
                # test_in_context(model, tokenizer, noise_type)

                print(f"\nProcessing Stats-Context with {noise_type}...")
                test_model(model_name, model, tokenizer, test_type="stats_context", noise_type=noise_type)

    except Exception as e:
        print(f"Program terminated with error: {str(e)}")


