from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import torch
import os
import sys
from PIL import Image
import argparse
from tqdm import tqdm
import pandas as pd
import pickle
import json
from huggingface_hub import login

# this is a restricted repo
# to use the model, go to https://huggingface.co/settings/tokens
# Create a new fine-grained token
# Add repositories (in this case google/paligemma-3b-mix-224) to 
# repositories permissions

HF_TOKEN = "ADD YOUR TOKEN HERE"
login(token=HF_TOKEN)

torch.manual_seed(1234)

def write_pickle(data, filename):
    with open(filename, 'wb') as fid:
        pickle.dump(data, fid, protocol=pickle.HIGHEST_PROTOCOL)

def run_paligemma(model_name='paligemma-3b-mix-224', data_path=None, input_path=None, results_path=None):
    print('Loading', model_name)
    os.makedirs('cache', exist_ok=True)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = PaliGemmaForConditionalGeneration.from_pretrained(f'google/{model_name}', token=HF_TOKEN).eval().to(device)
    processor = AutoProcessor.from_pretrained(f'google/{model_name}', token=HF_TOKEN)


    data_path = os.environ['SENSOR_BIAS_DATA']
    cache_file = os.path.join('cache', os.path.basename(results_path).replace('.xlsx', '.pkl'))
    
    results = []

    with open(input_path, 'r') as fid:
        data_dict = json.load(fid)

    if os.path.isfile(cache_file):
        with open(cache_file, 'rb') as fid:
            results = pickle.load(fid)

    processed_images = {x['Path']: 0 for x in results}

    num_processed = 0
    num_images = len(data_dict)

    for path, question_dict in tqdm(data_dict.items(), total=num_images):

        image_path = os.path.join(data_path, path)
        image = Image.open(image_path)
        
        response_dict = {'Path': path}

        if path not in processed_images:
            for i in range(1,8):

                question = question_dict[f'Q{i}']
                # Prompt formatting as explained in 
                # https://github.com/huggingface/blog/blob/main/paligemma.md
                # add <image> for each image in the prompt
                # add beginning of sentence <bos> token after the images
                # add newline after the prompt
                model_inputs = processor(text=f'<image><bos>{question}\n', images=image, return_tensors='pt').to(device)
                input_len = model_inputs['input_ids'].shape[-1]

                with torch.inference_mode():
                    generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
                    generation = generation[0][input_len:]
                    response = processor.decode(generation, skip_special_tokens=True)

                    response_dict[f'A{i}'] = response

            results.append(response_dict)
            num_processed += 1

        if (num_processed > 0) and (num_processed % 1000 == 0):
            write_pickle(results, cache_file)

    write_pickle(results, cache_file)
    print(f'Saved results to {cache_file}')
    pd.DataFrame.from_dict(results).to_excel(results_path)

#paligemma-3b-mix-224
#paligemma-3b-mix-448


if __name__ == '__main__':
    model_name = sys.argv[1]

    input_path = '../../../../annotations/questions_data_v5.json'
    data_path = '../../../../data_v5'
    results_path = f'../../../../raw_results/vqa/{model_name}.xlsx'
    run_paligemma(model_name=model_name, 
                 data_path=data_path, 
                 input_path=input_path, 
                 results_path=results_path)