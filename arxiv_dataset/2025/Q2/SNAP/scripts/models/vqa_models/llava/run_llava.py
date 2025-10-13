from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

import argparse
import torch
import os
import sys
import json
import pandas as pd
import pickle

from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
import re
from tqdm import tqdm

#from scripts.models.vqa_models.phi3vision.run_phi3vision import response


#from scripts.models.vqa_models.phi3vision.run_phi3vision import model_id

def write_pickle(data, filename):
    with open(filename, 'wb') as fid:
        pickle.dump(data, fid, protocol=pickle.HIGHEST_PROTOCOL)


class LLAVA():
    def __init__(self, model_id):
        self.model_id = model_id
        self.model_path = f'liuhaotian/{model_id}'
        self.model_base = None
        self.conv_mode = None
        self.temperature = 0.2
        self.top_p = None
        self.num_beams = 1
        self.max_new_tokens = 512


    def setup_model(self):

        model_name = get_model_name_from_path(self.model_path)
        if "llama-2" in model_name.lower():
            model_conv_mode = "llava_llama_2"
        elif "mistral" in model_name.lower():
            model_conv_mode = "mistral_instruct"
            # model_conv_mode = "mistral_direct"
        elif "v1.6-34b" in model_name.lower():
            model_conv_mode = "chatml_direct"
        elif "v1" in model_name.lower():
            model_conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            model_conv_mode = "mpt"
        else:
            model_conv_mode = "llava_v0"

        if self.conv_mode is not None and self.conv_mode != model_conv_mode:
            print(
                "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                    self.conv_mode, model_conv_mode, self.conv_mode
                )
            )
        else:
            self.conv_mode = model_conv_mode

        disable_torch_init()

        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            self.model_path, self.model_base, model_name
        )

    def format_prompt(self, qs):
        #qs = prompt
        image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        if IMAGE_PLACEHOLDER in qs:
            if self.model.config.mm_use_im_start_end:
                qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
            else:
                qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
        else:
            if self.model.config.mm_use_im_start_end:
                qs = image_token_se + "\n" + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        return prompt


    def run_model(self):

        data_path = os.environ['SENSOR_BIAS_DATA']
        input_path = '../../../../annotations/SNAP_VQA.json'
        results_path = f'../../../../raw_results/vqa/{self.model_id}.xlsx'


        os.makedirs('../../../../raw_results/vqa/', exist_ok=True)
        os.makedirs('cache', exist_ok=True)

        with open(input_path, 'r') as fid:
            data_dict = json.load(fid)

        cache_file = f"cache/{os.path.basename(results_path).replace('xlsx', 'pkl')}"

        results = []
        if os.path.isfile(cache_file):
            with open(cache_file, 'rb') as fid:
                results = pickle.load(fid)

        processed_images = {x['Path']: 0 for x in results}

        num_images = len(data_dict)
        with torch.no_grad():
            num_processed = 0
            for path, question_dict in tqdm(data_dict.items(), total=num_images):
                if path not in processed_images:
                    img_path = os.path.join(data_path, path)

                    images = [Image.open(img_path).convert("RGB")]

                    image_sizes = [x.size for x in images]
                    images_tensor = process_images(
                        images,
                        self.image_processor,
                        self.model.config
                    ).to(self.model.device, dtype=torch.float16)

                    response_dict = {'Path': path}
                    for i in range(1, 8):
                        question = question_dict[f'Q{i}']
                        prompt = self.format_prompt(question)

                        input_ids = (
                            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                            .unsqueeze(0)
                            .cuda()
                        )


                        with torch.inference_mode():
                            output_ids = self.model.generate(
                                input_ids,
                                images=images_tensor,
                                image_sizes=image_sizes,
                                do_sample=True if self.temperature > 0 else False,
                                temperature=self.temperature,
                                top_p=self.top_p,
                                num_beams=self.num_beams,
                                max_new_tokens=self.max_new_tokens,
                                use_cache=True
                            )

                        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                        #print(f'{img_path}\nQuery: {qs}\n Answer: {outputs}')
                        response_dict[f'A{i}'] = outputs
                    results.append(response_dict)
                    num_processed += 1

                    # save results periodically if the script is interrupted
                if (num_processed > 0) and (num_processed % 100 == 0):
                    write_pickle(results, cache_file)
                # print(f'Saved results to {cache_file}')

        write_pickle(results, cache_file)
        print(f'Saved results to {cache_file}')

        pd.DataFrame.from_dict(results).to_excel(results_path)


# python3 run_llava.py liuhaotian/llava-v1.6-vicuna-7b

if __name__ == '__main__':
    model_name = sys.argv[1]
    llavaVL = LLAVA(model_name)
    llavaVL.setup_model()
    llavaVL.run_model()