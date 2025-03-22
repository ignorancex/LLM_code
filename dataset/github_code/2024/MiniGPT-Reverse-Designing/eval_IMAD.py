import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from transformers import StoppingCriteriaList

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

from PIL import Image
import json
import pandas as pd
from tqdm import tqdm
import pdb


def parse_args():
    parser = argparse.ArgumentParser(description="Eval")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0,
             'pretrain_llama2': CONV_VISION_LLama2}

print('\n\n Model Initialization \n\n')

args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

CONV_VISION = conv_dict[model_config.model_type]

vis_processor_cfg = cfg.datasets_cfg.IMAD.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

stop_words_ids = [[835], [2277, 29937], [2]]
stop_words_ids = [torch.tensor(ids).to(device='cuda:{}'.format(args.gpu_id)) for ids in stop_words_ids]
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

print('Initialization Finished')

# ========================================
#             Eval
# ========================================

f = open(cfg.run_cfg.data)
test_data = json.load(f)["annotations"]
print("\n\nNum of test data: ", len(test_data))

all_prompts = model.prompt_list_cmd + model.prompt_list
all_prompts = ["<s> " + item for item in all_prompts]
print("\nLen all prompts: ", len(all_prompts))
print(all_prompts[0])

image_ids = []
prompts = []
outputs = []
preds = []
commands = []

# old_data = pd.read_csv("old_data.csv")["prompts"].tolist()

for idx in tqdm(range(len(test_data[:10]))):  
    
    item = test_data[idx]
    
    im_id = item["image_id"]
    pred = item["prediction"]
    cmd = item["command"]
    # print(im_id)
    
    image1_path = "data/IMAD/input_images/" + im_id + ".jpg"
    image1 = Image.open(image1_path).convert("RGB")
    
    image2_path = "data/IMAD/output_images/" + im_id + ".jpg"
    image2 = Image.open(image2_path).convert("RGB")

    image_1 = vis_processor(image1).unsqueeze(0).to(args.gpu_id)
    image_2 = vis_processor(image2).unsqueeze(0).to(args.gpu_id)
    
    image_1_emb, _ = model.encode_img(image_1)
    image_2_emb, _ = model.encode_img(image_2)

    emb_lists = []
    
    text_input = random.choices(all_prompts, k=1)[0]
    # text_input = old_data[idx]
    
    if "Command:" in text_input:
        cur_segs = text_input.split("[/INST]")
        if cmd[-1] == ".":
            cmd = cmd[:-1]
        text_input = cur_segs[0][:-1] + cmd + ". [/INST] "
    
    p_segs = text_input.split('<ImageHere1>')
    interleave_emb = []
    for idx, seg in enumerate(p_segs[:-1]):
        p_tokens = model.llama_tokenizer(
            seg, return_tensors="pt", add_special_tokens=False).to(args.gpu_id)
        p_embed = model.embed_tokens(p_tokens.input_ids)
        interleave_emb.append(torch.cat([p_embed, image_1_emb], dim=1))
    wrapped_emb1 = torch.cat(interleave_emb, dim=1)

    p_segs = p_segs[-1].split('<ImageHere2>')
    interleave_emb = []
    for idx, seg in enumerate(p_segs[:-1]):
        p_tokens = model.llama_tokenizer(
            seg, return_tensors="pt", add_special_tokens=False).to(args.gpu_id)
        p_embed = model.embed_tokens(p_tokens.input_ids)
        interleave_emb.append(torch.cat([p_embed, image_2_emb], dim=1))
    wrapped_emb2 = torch.cat(interleave_emb, dim=1)

    # End of the prompt
    p_tokens = model.llama_tokenizer(
        p_segs[-1], return_tensors="pt", add_special_tokens=False).to(args.gpu_id)
    p_embed = model.embed_tokens(p_tokens.input_ids)
    wrapped_emb = torch.cat([wrapped_emb1, wrapped_emb2, p_embed], dim=1)
    emb_lists.append(wrapped_emb)
    
    max_new_tokens=256
    num_beams=1
    min_length=1
    top_p=0.9
    repetition_penalty=1.05
    length_penalty=1
    temperature=0.3
    max_length=2000
    
    generation_kwargs = dict(
        inputs_embeds=emb_lists[0],
        max_new_tokens=max_new_tokens,
        # stopping_criteria=stopping_criteria,
        # num_beams=num_beams,
        do_sample=False,
        # min_length=min_length,
        # top_p=top_p,
        # repetition_penalty=repetition_penalty,
        # length_penalty=length_penalty,
        # temperature=float(temperature),
    )
    
    with model.maybe_autocast():
        output_token = model.llama_model.generate(**generation_kwargs)[0]
    output_text = model.llama_tokenizer.decode(output_token, skip_special_tokens=True)
    
    image_ids.append(im_id)
    prompts.append(text_input)
    outputs.append(output_text)
    preds.append(pred)
    commands.append(cmd)
    
df = pd.DataFrame({'image_ids':image_ids,
                   'commands':commands,
                   'groundtruth':preds,
                   'prompts':prompts,
                   'model_output':outputs})

df.to_csv(cfg.run_cfg.resFileName, index=False)
