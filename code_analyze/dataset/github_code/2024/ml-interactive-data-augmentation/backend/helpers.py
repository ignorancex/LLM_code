"""
For licensing see accompanying LICENSE file.
Copyright (C) 2024 Apple Inc. All Rights Reserved.
"""

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
import vec2text
from openai import OpenAI
import json
import sys

# model paths + settings

embedding_model_name = "sentence-transformers/gtr-t5-base"
corrector_model_name = "gtr-base"
OPENAI_MODEL = "gpt-4o-mini"


# load the api keys from the secrets file
# outputs: open_ai_key (str)
def load_api_keys():
    try:
        with open("../secrets.json") as f:
            secrets = json.load(f)
        open_ai_key = secrets["openai"]
        print("API keys loaded.")
        return open_ai_key
    except FileNotFoundError:
        print("Secrets file not found. YOU NEED THEM TO RUN THIS.")
        return None

# perform mean pooling on hidden states while considering attention mask
# inputs: hidden_states (torch.Tensor), attention_mask (torch.Tensor)
# outputs: pooled_outputs (torch.Tensor)


def mean_pool(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    B, S, D = hidden_states.shape
    unmasked_outputs = hidden_states * attention_mask[..., None]
    pooled_outputs = unmasked_outputs.sum(
        dim=1) / attention_mask.sum(dim=1)[:, None]
    assert pooled_outputs.shape == (B, D)
    return pooled_outputs

# format the new sentences and umap points to return as a list of dict objects
# inputs: new_sentences (list), new_umap_points (np.ndarray)
# outputs: new_points (list)


def format_new_points(new_sentences, new_umap_points):
    new_points = []
    for i in range(len(new_sentences)):
        new_points.append({
            "sentence": new_sentences[i],
            "umap_x": new_umap_points[i][0],
            "umap_y": new_umap_points[i][1]
        })
    return new_points

# format the new umap points to return as a list of dict objects
# inputs: new_umap_points (np.ndarray)
# outputs: new_points (list)


def format_new_points_umap(new_umap_points):
    new_points = []
    for i in range(len(new_umap_points)):
        new_points.append({
            "umap_x": new_umap_points[i][0],
            "umap_y": new_umap_points[i][1]
        })
    return new_points

# format the new sentences, umap points, and weights to return as a list of dict objects
# inputs: new_sentences (list), new_umap_points (np.ndarray), weights (list)
# outputs: new_points (list)


def format_new_points_interpolate(new_sentences, new_umap_points, weights):
    new_points = []
    for i in range(len(new_sentences)):
        new_points.append({
            "sentence": new_sentences[i],
            "umap_x": new_umap_points[i][0],
            "umap_y": new_umap_points[i][1],
            "weight": weights[i]
        })
    return new_points

 # Convert float32 obj to regular float
 # inputs: obj (np.number)
    # outputs: float(obj)


def convert_to_serializable(obj):
    if isinstance(obj, np.number):
        return float(obj)
    return obj

# convert dict of points to serializable format
# inputs: points (list)
# outputs: serializable_points (list)


def convert_points_to_serializable(points):
    # Convert the generated points to serializable format
    serializable_points = [{k: convert_to_serializable(v) for k, v in point.items()}
                           for point in points]
    return serializable_points

# Format the response from the llm model if it is a multiline response
# inputs: response (str)
# outputs: sentence_list (list)


def format_llm_multiline_response(response):
    # format the response
    response_formatted = str(response).split("\n")

    # remove numbers at the beginning of each line
    sentence_list = [line.split(". ", 1)[1] if len(line) > 0 and line[0].isdigit(
    ) and len(line.split(". ", 1)) > 1 else line for line in response_formatted]

    # strip whitespace for each line
    sentence_list = [line.strip() for line in sentence_list]

    # remove any empty lines
    sentence_list = [line for line in sentence_list if line]

    return sentence_list

# [OPENAI] generate variations of the input sentence using the llm model
# inputs: llm (OpenAI), num_sentences (int), input_sentence (str),
#         temperature (float)
# outputs: sentence_list (list)


def generate_sentence_variations(llm, num_sentences, input_sentence, temperature=0.5):
    variation_prompt = f"""You are an expert prompt writer. You are in charge of writing diverse prompts to send to a large language model.
                        Generate exactly {num_sentences} different variations of Sentence A.
                        A placeholder is a word in square brackets (e.g., [name]). 
                        If Sentence A contains any placeholders, make sure all newly generated sentences have the same placeholders.
                        Maintain the length and style of the original sentence, but vary the sentence structure and wording.
                        Make sure each new sentence is the same type of sentence as Sentence A (e.g., statement, question, command).
                        Remember you are not responding to Sentence A, but generating variations of it.
                        Format your response with one new sentence per line (separated with a \\n character).
                        Just generate the {num_sentences} new sentence without explanations:"""

    user_prompt = f"Sentence A: {input_sentence}"

    completion = llm.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": variation_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    response = completion.choices[0].message.content

    # format the response
    sentence_list = format_llm_multiline_response(response)

    return sentence_list


# [MISTRAL] generate variations of the input sentence using the llm model
# inputs: llm (Llama), num_sentences (int), input_sentence (str)
#         max_tokens (int), temperature (float), echo (bool), stop (list)
# outputs: sentence_list (list)


def generate_sentence_variations_mistral(llm, num_sentences, input_sentence,
                                         max_tokens=200,
                                         temperature=0.1,
                                         echo=False,
                                         stop=["Q"]):

    if num_sentences < 1:
        return []

    formatted_prompt = f"[INST]Generate exactly {num_sentences} different variations of Prompt A: '{input_sentence}'" + \
        "A placeholder is a word in square brackets (e.g., [name]). If Prompt A contains any placeholders, make sure all newly generated prompts have the same placeholders." + \
        "Format your response with one new prompt per line. An example output for the prompt 'Write a short story about flying cupcakes' would be:" + \
        "Create a brief tale about airborne cupcakes.\n" + \
        "Write a concise story featuring cupcakes that can fly.\n" + \
        "Pen a short narrative involving flying cupcakes.\n" + \
        f"Just generate the {num_sentences} new prompts without explanations:[/INST]Model answer:</s>"
    # Define the parameters
    model_output = llm(
        formatted_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        echo=echo,
        stop=stop,
    )

    sentence_list = model_output["choices"][0]["text"].strip().strip(
        '"\'').split("\n")
    sentence_list = [line.split(". ", 1)[1] if line[0].isdigit(
    ) else line for line in sentence_list]

    return sentence_list

# [OPENAI] general method for prompting the llm model for sentence variations
# inputs: llm (OpenAI), num_sentences (int), input_sentence (str),
#         instruction (str), temperature (float)
# outputs: sentence_list (list)


def prompt_for_sentence_variations_llm(llm, num_sentences, input_sentence, instruction, temperature=0.5):
    system_prompt = f"""You are an expert prompt writer. You are in charge of writing diverse prompts to send to a large language model.
                Generate exactly {num_sentences} different variations of Sentence A by appling this instruction: {instruction}.
                Remember you are not responding to Sentence A, but generating variations of it.
                Each new sentence should be different from the original and other generated sentences.
                Format your response with one new sentence per line (separated with a \\n character).
                Just generate the {num_sentences} new sentence without explanations:
                """
    user_prompt = f"Sentence A: {input_sentence}"

    completion = llm.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    response = completion.choices[0].message.content

    # format the response
    sentence_list = format_llm_multiline_response(response)

    return sentence_list

# [OPENAI] correct the input sentence using the llm model
# inputs: llm (OpenAI), example_sentence (str), input_sentence (str),
#         temperature (float)
# outputs: response_formatted (str)


def correct_sentence(llm, example_sentence, input_sentence, temperature=0.1):
    correct_prompt = """You are an expert prompt writer. You are in charge of writing diverse prompts to send to a large language model.
                    Follow these steps to modify Sentence A. A placeholder is a word in square brackets (e.g., [name]). 
                    1) If Sentence A contains any placeholders, remove them.
                    2) If Sentence B has any placeholders, add them to Sentence A.
                    3) Using Sentence B as an example of a well-written sentence, edit Sentence A to be complete and grammatically correct, while maintaining its length and meaning.
                    4) Make sure Sentence A is the same type of sentence as Sentence B (e.g., statement, question, command).
                    5) Make sure Sentence A is not identical to Sentence B, and ends with a period, question mark, or quotation mark.
                    Just generate the modified version of Sentence A without explanations:"""
    user_prompt = f"Sentence A: {input_sentence}\nSentence B: {example_sentence}"

    completion = llm.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": correct_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    response = completion.choices[0].message.content

    # format the response
    response_formatted = response.strip().strip('"\'')
    return response_formatted


# [MISTRAL] correct the input sentence using the llm model
# inputs: llm (Llama), example_sentence (str), input_sentence (str)
#         max_tokens (int), temperature (float), echo (bool), stop (list)
# outputs: model_output (str)


def correct_sentence_mistral(llm, example_sentence, input_sentence,
                             max_tokens=100,
                             temperature=0.1,
                             echo=False,
                             stop=["Q"]):

    formatted_prompt = "[INST]You are an expert prompt writer. Imagine you are writing prompts to send to a large language model." + \
        f"Follow these steps to modify Prompt A = '{input_sentence}'" + \
        "A placeholder is a word in square brackets (e.g., [name]). 1) if Prompt A contains any placeholders, remove them." + \
        f"2) Using Prompt B = '{example_sentence}' as an example, edit Prompt A to be a complete and grammatically correct sentence, while maintaining its length." + \
        "Just generate the modified version of Prompt A without explanations:[/INST]Model answer:</s>"
    # Define the parameters
    model_output = llm(
        formatted_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        echo=echo,
        stop=stop,
    )

    formatted_output = model_output["choices"][0]["text"]

    # remove anything before "Prompt A:"
    if ("Prompt A:" in formatted_output):
        formatted_output = formatted_output.split("Prompt A:")[1]

    formatted_output = formatted_output.strip().strip('"\'')

    return formatted_output

# [OPENAI] correct (multiple) interpolation sentences using the llm model
# inputs: llm (OpenAI), sentences (list), sent1 (str), sent2 (str), temperature (float)
# outputs: sentence_list (list)


def correct_multiple_sentences(llm, sentences, sent1, sent2, temperature=0.1):
    num_sentences = len(sentences)
    system_prompt = f"""You are an expert prompt writer. You are in charge of writing diverse prompts to send to a large language model.
                    Follow these steps to modify each of the {num_sentences} sentences in the list.
                    These sentences were generated by interpolating between two sentences, sentence A: '{sent1}' and sentence B: '{sent2}'.
                    1) Edit each sentence to be complete and grammatically correct, while maintaining its length and meaning.
                    2) Remember that each sentence should be a blend of sentence A and sentence B. 
                    This means all sentences should have aspects of both sentence A and sentence B, but with earlier sentences leaning more towards {sent1} and later sentences leaning more towards {sent2}.
                    This progression should be smooth and gradual, like a story that starts with sentence A and ends with sentence B.
                    3) A placeholder is a word in square brackets (e.g., [name]). If sentence A or sentence B contains any placeholders, add them accordingly to the interpolated sentences.
                    4) Make sure there are no duplicate sentences, and no sentences end in commas.
                    5) Format your response with one new sentence per line (separated with a \\n character).
                    Just generate the {num_sentences} new sentences without explanations:"""
    user_prompt = "Sentences:\n" + str(sentences)

    completion = llm.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    response = completion.choices[0].message.content

    # format the response
    sentence_list = format_llm_multiline_response(response)

    return sentence_list


# [OPENAI] generate prompt ideas to augment a dataset of sentences
# inputs: llm (OpenAI), num_prompts (int), sentence (str), temperature (float)
# outputs: prompt_list (list)
def generate_prompt_ideas(llm, num_prompts, sentence, temperature=0.5):
    system_prompt = f"""You are an expert prompt writer in charge of augmenting a dataset of sentences to increase the data diversity.
                    Please come up with {num_prompts} prompt ideas you could send to a large language model to generate diverse variations of the provided sentence. 
                    Remember you're not rewriting the sentence but coming up with ideas that would be useful to modify the sentence in various ways. 
                    Don't include the sentence or summarize details about it in the prompt. Each prompt should be concise and unique.
                    Format your response with one prompt per line (separated with a \\n character), without numbers or bullet points.
                    Just generate the {num_prompts} prompts without explanations:"""
    user_prompt = f"Sentence: {sentence}"

    completion = llm.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    response = completion.choices[0].message.content

    # format the response
    prompt_list = format_llm_multiline_response(response)

    return prompt_list

# load dataset-agnostic models needed to run backend
# outputs: model_dict (dict)


def load_models():
    # see what version of python is being used
    python_version = f"{sys.version_info[0]}.{sys.version_info[1]}"
    print('Python version:', python_version)

    print('Loading models...')
    device = torch.device('cuda' if torch.cuda.is_available(
    ) else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print('model device:', device)

    # Load the embedding model and tokenizer
    embedding_model = AutoModel.from_pretrained(
        embedding_model_name, device_map="auto", torch_dtype=torch.float16
    ).encoder
    # move the model to the device
    embedding_model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
    print('\nembedding model loaded:', embedding_model_name)
    corrector = vec2text.load_pretrained_corrector(
        corrector_model_name)
    print('corrector model loaded:', corrector_model_name)

    # Load OpenAI API
    api_key = load_api_keys()
    llm = OpenAI(api_key=api_key)

    # return a dictionary with all the models
    model_dict = {
        'python_version': python_version,
        'device': device,
        'embedding_model': embedding_model,
        'tokenizer': tokenizer,
        'corrector': corrector,
        'llm': llm
    }
    return model_dict
