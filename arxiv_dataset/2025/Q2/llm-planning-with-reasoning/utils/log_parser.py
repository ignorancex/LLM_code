import json
import re
from copy import deepcopy
import numpy as np
import nltk
from utils.functions import check_nltk_resource
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


fixed_keys = {'Mask', 'Accepted', 'Corrct Scores', 'Eval fail rate', 'Scores', 'Responses'}
prommpt_divider = "<think>\nOkay,"
div = "\n=======\n" # divider for responses
eos_string = "<\uff5cend\u2581of\u2581sentence\uff5c>"
# 

def jprint(s):
    print(json.dumps(s, indent=4))

# parse log into hashmap
def extract_eval_log(text, fixed_keys, div, prommpt_divider):
    # clean up eos

    text = text.replace(eos_string,"")
    # Initialize hashmap
    hashmap = {key: None for key in fixed_keys}
    lines = text.split("\n")

    key = None
    capture_responses = False
    responses_text = []

    for line in lines:
        match = re.match(r"([\w\s]+):\s*(.*)", line)
        
        if match:
            key = match.group(1).strip()
            value = match.group(2).strip()

            if key in fixed_keys:
                if key == "Mask":  # Boolean list
                    value = [v == "True" for v in value.strip("[]").replace(",", "").split()]
                elif key == "Responses":  # Start capturing responses
                    capture_responses = True
                    continue  # Skip this line since actual content starts from next line
                else:  # Convert numeric values
                    value = value.replace("%", "").strip("[]").split()
                    value = [float(v) for v in value] if len(value) > 1 else float(value[0])

                hashmap[key] = value
        elif capture_responses:  # Capture multi-line responses
            responses_text.append(line)

    # Store collected responses
    hashmap["Responses"] = "\n".join(responses_text).strip()  # Join and remove trailing spaces
    hashmap["Responses"] =  hashmap["Responses"].split(div) 

    # divide responses into prompt and reply
    prompts = []
    prompt_responses = []
    for r in hashmap["Responses"]:
        responses_parts = r.split(prommpt_divider)
        prompt = responses_parts[0] + prommpt_divider
        prompt_response = responses_parts[-1]
        prompts.append(prompt)
        prompt_responses.append(prompt_response)

    hashmap["Prompts"] = prompts
    hashmap["PromptResponses"] = prompt_responses
    return hashmap


def log_parser(fname):
    """
    outputs hashmap with three keys.
    "Evaluation"
    "Parameters"
    "Data"
    """
    fdata = json.load(open(fname))

    result = {}

    result["Evaluation"] = fdata[-2]
    result["Parameters"] = fdata[-1]

    data = []
    for i in range(0,len(fdata)-2):
        curr = fdata[i]
        # parse log into hashmap
        extracted_hashmap = extract_eval_log(curr['evaluation_log'] , fixed_keys, div, prommpt_divider)
        # merge hashmap
        merged_curr = curr.copy()
        merged_curr.update(extracted_hashmap) 
        data.append( merged_curr)

    result["Data"] = data

    return result


def fail_rate_calc(result):
    eval_fail_rate= []
    for d in result["Data"]:
       eval_fail_rate.append(d['Eval fail rate']) 
    eval_fail_rate=np.array(eval_fail_rate)
    return np.mean(eval_fail_rate)


def extract_sentences(text):
    # Download the necessary tokenizer
    check_nltk_resource('punkt',verbose=False)

    # Tokenize text into sentences
    sentences = nltk.tokenize.sent_tokenize(text)

    return sentences

def extraction_process(result):
    all_sentences = []
    all_text = []
    for d in result["Data"]:
        responses = d['PromptResponses']
        for r in responses:
            sentences = extract_sentences(r)
            all_sentences.append(sentences)
            all_text.append(r)
    return all_sentences, all_text

def count_repeated_sentences(sentences):
    
    # Count occurrences of each sentence
    sentence_counts = Counter(sentences)
    
    # Filter only repeated sentences
    repeated_sentences = {sentence: count for sentence, count in sentence_counts.items() if count > 1}
    
    repetition_ratio = sum(repeated_sentences.values()) / len(sentences) if sentences else 0

    # handle empty case
    if len(repeated_sentences)==0:
       repeated_sentences = {"" : 0} 
    
    return repeated_sentences, repetition_ratio


def compute_entropy(sentences):
    """  Entropy-Based Metric """
    total_sentences = len(sentences)
    if total_sentences == 0:
        return 0  # No entropy if there are no sentences

    sentence_counts = Counter(sentences)
    probabilities = np.array(list(sentence_counts.values())) / total_sentences
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    return entropy

def compute_ttr(sentences):
    """ Type-Token Ratio (TTR) """
    unique_sentences = set(sentences)
    return len(unique_sentences) / len(sentences) if sentences else 0

def repeated_ngrams(text, n=3):
    """Find repeated n-grams in a text and calculate repetition ratio."""
    words = text.split()
    ngrams = [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]
    
    # Count occurrences
    ngram_counts = Counter(ngrams)
    
    # Identify repeated n-grams (occurring more than once)
    repeated = {k: v for k, v in ngram_counts.items() if v > 1}
    
    # Compute repetition ratio
    repetition_ratio = sum(repeated.values()) / len(ngrams) if ngrams else 0

    return repeated, repetition_ratio


def compute_diversity(sentences, model):
    if len(sentences) < 2:
        return 1  # Max diversity if only one sentence

    embeddings = model.encode(sentences)
    sim_matrix = cosine_similarity(embeddings)
    
    # Ignore diagonal (self-similarity)
    avg_similarity = (np.sum(sim_matrix) - np.trace(sim_matrix)) / (len(sentences)**2 - len(sentences))
    
    return 1 - avg_similarity  # Higher means more diversity


"""
Markdown Table Generator

This script generates a Markdown table from a given set of numerical values.
It ensures that all floating-point numbers are formatted to two decimal places.

Usage:
- Provide a list `x` for the first column (e.g., `max_new_tokens`).
- Provide a list of lists `y_values` for the subsequent columns (each sublist corresponds to a metric).
- Provide a list `titles` for the column headers.
- Provide a string `xtitle` for the first column's title.

Example:
    x = [10, 20, 30]
    y_values = [
        [0.81, 0.85, 0.87],  # Accuracy
        [0.75, 0.78, 0.80],  # Macro F1
    ]
    titles = ["Accuracy", "Macro F1"]
    xtitle = "max_new_tokens"

    markdown_table = generate_markdown_table(x, y_values, titles, xtitle)
    print(markdown_table)

"""

def generate_markdown_table(x, y_values, titles, xtitle):
    """
        Generates a Markdown table from numerical data.

        Parameters:
        - x (list): Values for the first column (e.g., max_new_tokens).
        - y_values (list of lists): Metrics data; each inner list corresponds to a column.
        - titles (list): Column headers for the metrics.
        - xtitle (str): Title for the first column.

        Returns:
        - str: A formatted Markdown table as a string.
    """
    # Define column headers
    headers = [xtitle] + titles
    # Create table header
    markdown_table = "| " + " | ".join(headers) + " |\n"
    markdown_table += "|-" + "-|-".join(["-" * len(h) for h in headers]) + "-|\n"
    
    # Populate rows with formatted values
    for i in range(len(x)):
        row_values = [f"{x[i]}"] + [f"{y_values[j][i]:.2f}" for j in range(len(y_values))]
        markdown_table += "| " + " | ".join(row_values) + " |\n"

    return markdown_table
