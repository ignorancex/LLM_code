import torch
import sys
import os
import pickle
import json
from tqdm import tqdm
import pandas as pd
from transformers import AutoModelForCausalLM

from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images

def write_pickle(data, filename):
    with open(filename, 'wb') as fid:
        pickle.dump(data, fid, protocol=pickle.HIGHEST_PROTOCOL)

class DeepSeekVL():

    def __init__(self, model_id, data_part=None):
        self.model_id = model_id
        self.model_path = f'deepseek-ai/{model_id}'
        self.data_part = data_part

    def setup_model(self):

        self.vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(self.model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer

        self.vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True)
        #self.vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    def format_prompt(self, img_path, q):
        ## single image conversation example
        prompt = [
            {
                "role": "User",
                "content": f"<image_placeholder>{q}",
                "images": [img_path],
            },
            {"role": "Assistant", "content": ""},
        ]
        return prompt


    def run_model(self):

        data_path = os.environ['SNAP_DATA_PATH']
        input_path = '../../../../../annotations/SNAP_VQA.json'
        results_path = f'../../../../../raw_results/vqa/{self.model_id}.xlsx'

        os.makedirs('../../../../raw_results/vqa/', exist_ok=True)
        os.makedirs('cache', exist_ok=True)

        with open(input_path, 'r') as fid:
            data_dict = json.load(fid)

        cache_file = f"cache/{os.path.basename(results_path).replace('xlsx', 'pkl')}"
        print(cache_file)
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

                    response_dict = {'Path': path}
                    for i in range(1, 8):
                        question = question_dict[f'Q{i}']
                        prompt = self.format_prompt(img_path, question)

                        # load images and prepare for inputs
                        pil_images = load_pil_images(prompt)
                        prepare_inputs = self.vl_chat_processor(
                            conversations=prompt,
                            images=pil_images,
                            force_batchify=True
                        ).to(self.vl_gpt.device)

                        # run image encoder to get the image embeddings
                        inputs_embeds = self.vl_gpt.prepare_inputs_embeds(**prepare_inputs)

                        with torch.inference_mode():
                            # run the model to get the response
                            outputs = self.vl_gpt.language_model.generate(
                                inputs_embeds=inputs_embeds,
                                attention_mask=prepare_inputs.attention_mask,
                                pad_token_id=self.tokenizer.eos_token_id,
                                bos_token_id=self.tokenizer.bos_token_id,
                                eos_token_id=self.tokenizer.eos_token_id,
                                max_new_tokens=512,
                                do_sample=False,
                                use_cache=True
                            )
                        answer = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
                        #print(f'{img_path}\nQuery: {question}\n Answer: {answer}')
                        response_dict[f'A{i}'] = answer
                    results.append(response_dict)
                    num_processed += 1

                    # save results periodically if the script is interrupted
                if (num_processed > 0) and (num_processed % 100 == 0):
                    write_pickle(results, cache_file)
                # print(f'Saved results to {cache_file}')

        write_pickle(results, cache_file)
        print(f'Saved results to {cache_file}')

        pd.DataFrame.from_dict(results).to_excel(results_path)


# model names
# DeepSeek-VL-1.3B-chat
# DeepSeek-VL-7B-chat

# git clone https://github.com/deepseek-ai/DeepSeek-VL

# python3 run_deepseek-vl.py DeepSeek-VL-7B-chat
# python3 run_deepseek-vl.py DeepSeek-VL-1.3B-chat


if __name__ == '__main__':
    model_name = sys.argv[1]
    data_part = None
    if len(sys.argv) == 3:
        data_part = sys.argv[2]
    deepseekvl = DeepSeekVL(model_name, data_part)
    deepseekvl.setup_model()
    deepseekvl.run_model()
