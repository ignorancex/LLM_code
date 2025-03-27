import transformers
import torch
import re
from transformers import pipeline
from transformers import AutoTokenizer
import pandas as pd
from secrets import SECRET_KEY

print(f"Your secret key is: {SECRET_KEY}")



def setup_model(model_name: str, key: str):
    """
    Sets up the text generation model and tokenizer with the given model name and token.
    
    Parameters:
        model_name (str): The model name to be used.
        key (str): The token for authentication.
    
    Returns:
        pipeline: The text generation pipeline.
        AutoTokenizer: The tokenizer used for the model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=key)
    llama_pipeline = transformers.pipeline(
        "text-generation",
        model=model_name,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",
        token=key
    )
    return llama_pipeline, tokenizer

def get_llama_response(model_pipeline, neg_word, pos_word):
    """
    Generate a response from the Llama model.

    Parameters:
        model_pipeline
        negative word
        positive word

    Returns:
        Tuple: negated and original sentence
    """

    prompt = (
    "You are a helpful assistant to generate sentence. "
    "Example follow this structure:\n"
    "Unattainable: The company's goals seemed unattainable given the current market conditions.\n"
    "Attainable: The company's goals seemed attainable given the favorable market conditions.\n\n"
    f"generate the sentence using the word '{neg_word}'. "
    f"Then change the sentence to use the word '{pos_word}', keeping minimal changes.\n\n"
    )

    
    # Generate text using the pipeline with custom parameters
    sequences = model_pipeline(
        prompt,
        max_length=200,  # Adjust as needed
        temperature=0.2,
        top_p=0.9,
        repetition_penalty=1,
        num_return_sequences=1  # Number of responses to generate
    )

    generated_text = sequences[0]['generated_text']

    # Extract positive and negative sentence from the model output
    neg_sentence, pos_sentence = extract_sentences_with_labels(generated_text, neg_word, pos_word)

    return (neg_sentence, pos_sentence)

def extract_sentences_with_labels(generated_text: str, neg_label: str, pos_label: str) -> tuple:
    """
    Extract sentences labeled with provided negative and positive labels from the generated text.

    Parameters:
        generated_text (str): The generated text containing sentences with various labels.
        neg_label (str): The label used for the negative sentiment.
        pos_label (str): The label used for the positive sentiment.

    Returns:
        tuple: Extracted sentences for the provided negative and positive labels.
    """
    # Create patterns to match each label and extract the subsequent sentence
    pattern = re.compile(
        rf'{neg_label}:\s*(.*?)(?=\n[A-Za-z]+:|$)|{pos_label}:\s*(.*?)(?=\n[A-Za-z]+:|$)',
        re.IGNORECASE | re.DOTALL
    )

    # Initialize sentences
    neg_sentence = None
    pos_sentence = None

    # Find all matches
    for match in pattern.finditer(generated_text):
        if match.group(1):
            neg_sentence = match.group(1).strip()
        if match.group(2):
            pos_sentence = match.group(2).strip()

    return (neg_sentence, pos_sentence)

def save_sentences_to_file(data: pd.DataFrame, model_pipeline, file_path: str):
    """
    Generate sentences using the provided DataFrame and save them to a .txt file.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing 'Positive', 'Negative', and 'Affix Used' columns.
        pipeline (pipeline): The text generation pipeline.
        num_rows (int): Number of rows to process from the DataFrame.
        file_path (str): Path to the .txt file where results will be saved.
    """
    # Extract the specified number of rows from the DataFrame
    df_subset = data

    # Open a file to write the results
    with open(file_path, 'w') as file:
        # Write column headers
        file.write('negated\toriginal\n')
        
        # Iterate through the rows and generate sentences
        for index, row in df_subset.iterrows():
            neg_word = row['Negative']
            pos_word = row['Positive']
            neg_sentence, pos_sentence = get_llama_response(model_pipeline, neg_word, pos_word)
            
            if neg_sentence and pos_sentence:
                # Write sentences to the file
                file.write(f"{neg_sentence}\t{pos_sentence}\n")


KEY = SECRET_KEY # Enter your key
model_name = "meta-llama/Llama-2-7b-chat-hf"

# Setup model and pipeline
llama_pipeline, tokenizer = setup_model(model_name, KEY)

# File path to the text file
file_path = '../data/affixal/affixal_list.txt'

# Read the text file into a DataFrame
affixal_pairs = pd.read_csv(file_path, sep='\t')

# Generate sentences and save to 'generated_sentences.txt'
save_sentences_to_file(affixal_pairs, llama_pipeline, file_path='../data/affixal/generated_sentences.txt')
