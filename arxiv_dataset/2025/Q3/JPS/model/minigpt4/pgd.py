import json
import torch
import time
import sys
sys.path.append('/workshop/crm/project/IJTS')
sys.path.append('/workshop/crm/project/IJTS/model')

import torch.nn as nn

import csv

from PIL import Image

from tqdm import tqdm
import argparse


import random
import torch.backends.cudnn as cudnn

import numpy as np
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry

from minigpt4.minigpt_utils import prompt_wrapper, generator
from torchvision.utils import save_image

def normalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
    images = images - mean[None, :, None, None]
    images = images / std[None, :, None, None]
    return images

def denormalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
    images = images * std[None, :, None, None]
    images = images + mean[None, :, None, None]
    return images

def parse_args():

    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg_path", default="eval_configs/minigpt4_eval.yaml", help="path to configuration file.")
    parser.add_argument("--gpu_id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--n_iters", type=int, default=500, help="specify the number of iterations for attack.")
    parser.add_argument('--eps', type=int, default=32, help="epsilon of the attack budget")
    parser.add_argument('--alpha', type=int, default=1, help="step_size of the attack")


    parser.add_argument("--save_dir", type=str, default='output',
                        help="save directory")

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

def attack_loss(model, prompts, targets):
    device = model.llama_model.device

    context_embs = prompts.context_embs

    if len(context_embs) == 1:
        context_embs = context_embs * len(targets) # expand to fit the batch_size

    assert len(context_embs) == len(targets), f"Unmathced batch size of prompts and targets {len(context_embs)} != {len(targets)}"

    batch_size = len(targets)
    model.llama_tokenizer.padding_side = "right"

    to_regress_tokens = model.llama_tokenizer(
        targets,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=model.max_txt_len,
        add_special_tokens=False
    ).to(device)
    to_regress_embs = model.llama_model.model.embed_tokens(to_regress_tokens.input_ids)

    bos = torch.ones([1, 1],
                        dtype=to_regress_tokens.input_ids.dtype,
                        device=to_regress_tokens.input_ids.device) * model.llama_tokenizer.bos_token_id
    bos_embs = model.llama_model.model.embed_tokens(bos)

    pad = torch.ones([1, 1],
                        dtype=to_regress_tokens.input_ids.dtype,
                        device=to_regress_tokens.input_ids.device) * model.llama_tokenizer.pad_token_id
    pad_embs = model.llama_model.model.embed_tokens(pad)


    T = to_regress_tokens.input_ids.masked_fill(
        to_regress_tokens.input_ids == model.llama_tokenizer.pad_token_id, -100
    )


    pos_padding = torch.argmin(T, dim=1) # a simple trick to find the start position of padding

    input_embs = []
    targets_mask = []

    target_tokens_length = []
    context_tokens_length = []
    seq_tokens_length = []

    for i in range(batch_size):

        pos = int(pos_padding[i])
        if T[i][pos] == -100:
            target_length = pos
        else:
            target_length = T.shape[1]

        targets_mask.append(T[i:i+1, :target_length])
        input_embs.append(to_regress_embs[i:i+1, :target_length]) # omit the padding tokens

        context_length = context_embs[i].shape[1]
        seq_length = target_length + context_length

        target_tokens_length.append(target_length)
        context_tokens_length.append(context_length)
        seq_tokens_length.append(seq_length)

    max_length = max(seq_tokens_length)

    attention_mask = []

    for i in range(batch_size):

        # masked out the context from loss computation
        context_mask =(
            torch.ones([1, context_tokens_length[i] + 1],
                    dtype=torch.long).to(device).fill_(-100)  # plus one for bos
        )

        # padding to align the length
        num_to_pad = max_length - seq_tokens_length[i]
        padding_mask = (
            torch.ones([1, num_to_pad],
                    dtype=torch.long).to(device).fill_(-100)
        )

        targets_mask[i] = torch.cat( [context_mask, targets_mask[i], padding_mask], dim=1 )
        input_embs[i] = torch.cat( [bos_embs, context_embs[i], input_embs[i],
                                    pad_embs.repeat(1, num_to_pad, 1)], dim=1 )
        attention_mask.append( torch.LongTensor( [[1]* (1+seq_tokens_length[i]) + [0]*num_to_pad ] ) )

    targets = torch.cat( targets_mask, dim=0 ).to(device)
    inputs_embs = torch.cat( input_embs, dim=0 ).to(device)
    attention_mask = torch.cat(attention_mask, dim=0).to(device)


    outputs = model.llama_model(
            inputs_embeds=inputs_embs,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
    loss = outputs.loss

    return loss

# ========================================
#             Model Initialization
# ========================================


print('>>> Initializing Models')

args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
model.eval()

vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

print('[Initialization Finished]\n')

question_file = '/workshop/crm/data/advbench/harmful_behaviors.csv'
result_file = "/data/chenrenmiao/project/Llama-guard2/advbench/advbench_withsuffix_result.json"

result = {}

question_list = []
target_list = []

with open(question_file, newline='') as csvfile:
    reader = csv.reader(csvfile)
    
    next(reader)
    
    for row in reader:
        question_list.append(row[0])
        target_list.append(row[1])

# Take the first 5 and attack
attack_list = [0,1,2,3,4]
attack_question_num = len(attack_list)
attack_question_list = [question_list[i] for i in attack_list]
attack_target_list = [target_list[i] for i in attack_list]

# attack_question_num = 3
# attack_question_list = question_list[:attack_question_num]
# attack_target_list = target_list[:attack_question_num]

image = Image.new('RGB', (224, 224), (0, 0, 0))
image = vis_processor(image).unsqueeze(0).to(model.device)

text_prompt_template = prompt_wrapper.minigpt4_chatbot_prompt

my_generator = generator.Generator(model=model)
my_generator.max_new_tokens = 64

num_steps = 2000
epsilon=32/255
alpha=1/255

adv_noise = torch.rand_like(image).to(model.llama_model.device) * 2 * epsilon - epsilon
x = denormalize(image).clone().to(model.llama_model.device)
adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data

adv_noise.requires_grad_(True)
adv_noise.retain_grad()

for i in range(num_steps):
    epoch_start_time = time.time()
    
    text_prompts = [text_prompt_template % attack_question for attack_question in attack_question_list]

    x_adv = x + adv_noise
    x_adv = normalize(x_adv)

    prompt = prompt_wrapper.Prompt(model=model, text_prompts=text_prompts, img_prompts=[[x_adv]])

    prompt.img_embs = prompt.img_embs * len(attack_question_list)
    prompt.update_context_embs()
    
    target_loss = attack_loss(model, prompt, attack_target_list)
    target_loss.backward()
    
    adv_noise.data = (adv_noise.data - alpha * adv_noise.grad.detach().sign()).clamp(-epsilon, epsilon)
    adv_noise.data = (adv_noise.data + x.data).clamp(0, 1) - x.data
    adv_noise.grad.zero_()
    model.zero_grad()
    
    success_num = 0
    if i % 10 == 0:
        x_adv = x + adv_noise
        x_adv = normalize(x_adv)
        prompt.update_img_prompts([[x_adv]])
        prompt.img_embs = prompt.img_embs * len(attack_question_list)
        prompt.update_context_embs()

        with torch.no_grad():
            responses, _ = my_generator.generates(prompt)

        for id, response in enumerate(responses):
            if response[:len(attack_target_list[id])] == attack_target_list[id]:
                is_success=True
            else:
                is_success=False
            success_num+=is_success
            print("**********************")
            print(f"Current Response:\n{response}\n")
            print("**********************")

    epoch_end_time = time.time()
    epoch_cost_time = round(epoch_end_time - epoch_start_time, 2)
    print(
        "################################\n"
        # f"Current Data: {i}/{len(harmful_data.goal[args.start:])}\n"
        f"Current Epoch: {i}/{num_steps}\n"
        f"Passed:{success_num}/{attack_question_num}\n"
        f"Loss:{target_loss.item()}\n"
        f"Epoch Cost:{epoch_cost_time}\n"
        # f"Current Suffix:\n{best_new_adv_suffix}\n"
        
        "################################\n")

    if success_num == attack_question_num:
        adv_img_prompt = denormalize(x_adv).detach().cpu()
        adv_img_prompt = adv_img_prompt.squeeze(0)
        save_image(adv_img_prompt, 'minigpt4_pgd.png')
        import pdb;pdb.set_trace()
        break
noise = images_tensor - ori_images

