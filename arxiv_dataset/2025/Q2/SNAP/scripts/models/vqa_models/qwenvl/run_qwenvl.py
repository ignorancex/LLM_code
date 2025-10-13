from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from transformers.generation import GenerationConfig
import torch
import os
from PIL import Image
import argparse
from tqdm import tqdm
import pandas as pd
import pickle
import json

torch.manual_seed(1234)

def write_pickle(data, filename):
    with open(filename, 'wb') as fid:
        pickle.dump(data, fid, protocol=pickle.HIGHEST_PROTOCOL)

def run_qwenvl(model_name=None, data_path=None, input_path=None, results_path=None):

    model = Qwen2VLForConditionalGeneration.from_pretrained(f'Qwen/{model_name}', torch_dtype="auto", device_map="auto").eval()
    min_pixels = 256 * 28 * 28
    max_pixels = 1280 * 28 * 28
    processor = AutoProcessor.from_pretrained(f'Qwen/{model_name}', min_pixels=min_pixels, max_pixels=max_pixels)

    # Specify hyperparameters for generation (No need to do this if you are using transformers>=4.32.0)
    # model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True)
    #data_df = pd.read_excel(input_list, engine='openpyxl')

    data_path = os.environ['SENSOR_BIAS_DATA']
    cache_file = os.path.join('cache', os.path.basename(results_path).replace('.xlsx', '.pkl'))
    

    with open(input_path, 'r') as fid:
        data_dict = json.load(fid)

    results = []
    if os.path.isfile(cache_file):
        with open(cache_file, 'rb') as fid:
            results = pickle.load(fid)

    processed_images = {x['Path']: 0 for x in results}


    num_processed = 0
    num_images = len(data_dict)
    with torch.no_grad():

        for path, question_dict in tqdm(data_dict.items(), total=num_images):
            if path not in processed_images:
                img_path = os.path.join(data_path, path)
                
                response_dict = {'Path': path}

                for i in range(1, 8):
                    question = question_dict[f'Q{i}']
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": f'file://{img_path}',
                                },
                                {"type": "text", "text": question},
                            ],
                        }
                    ]
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = processor(text=[text],
                                       images=image_inputs,
                                       videos=video_inputs,
                                       padding=True,
                                       return_tensors="pt",
                                       )
                    inputs = inputs.to(model.device)
                    generated_ids = model.generate(**inputs, max_new_tokens=1024)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    response = output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                
                    response_dict[f'A{i}'] = response
                
                results.append(response_dict)
                num_processed += 1

            if (num_processed > 0) and (num_processed % 1000 == 0):
                write_pickle(results, cache_file)

    write_pickle(results, cache_file)
    print(f'Saved results to {cache_file}')
    pd.DataFrame.from_dict(results).to_excel(results_path)

#Qwen2-VL-7B-Instruct
#Qwen2-VL-2B-Instruct

if __name__ == '__main__':
    model_name = sys.argv[1]

    input_path = '../../../../annotations/questions_data_v5.json'
    data_path = '../../../../data_v5'
    results_path = f'../../../../raw_results/vqa/{model_name}.xlsx'
    run_qwenvl(model_name=model_name, 
                 data_path=data_path, 
                 input_path=input_path, 
                 results_path=results_path)