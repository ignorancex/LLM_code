import argparse
import os
import torch
from PIL import Image
from utils import *
from sklearn.metrics import accuracy_score
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images



result_map = defaultdict(dict)

model_path_map = {
    "deepseek-vl": {
        "deepseek-vl-7b": "Deepseek/deepseek-vl-7b-chat",
        "deepseek-vl-1.3b": "Deepseek/deepseek-vl-1.3b-chat"
    },
    "deepseek-vl2": {
        "deepseek-vl2-small": "Deepseek/deepseek-vl2-small",
        "deepseek-vl2-tiny": "Deepseek/deepseek-vl2-tiny"
    }
}


def load_deepseek_vl(model_name):
    model_path = model_path_map["deepseek-vl"][model_name]
    processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    )
    model = model.to(torch.bfloat16).cuda().eval()
    return model, tokenizer, processor



model_example_map = {
    "deepseek-vl": load_deepseek_vl,
}


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type',
                        type=str,
                        default="deepseek-vl",
                        choices=model_example_map.keys())
    parser.add_argument('--model_used',
                        type=str,
                        default='all',
                        help='Number of prompts to run.')
    parser.add_argument('--cuda',
                        type=str,
                        default='2')
    parser.add_argument('--entity_task',
                        type=bool,
                        default=True)
    parser.add_argument('--relation_task',
                        type=bool,
                        default=True)
    return parser




def run_deepseek_vl(model_type, model_name, args):
    model, tokenizer, processor = model_example_map[model_type](model_name)
    evaluation_data_entity, evaluation_data_relation = load_task1_dataset()
    choices = ["2", "3", "4", "5", "6", "7", "8", "9"]
    choice_ids = [tokenizer.encode(choice)[-1] for choice in choices] 
    print(choice_ids)
    model = model.eval()
    if args.entity_task:
        answers = []
        predicts = []
        for data_instance in tqdm(evaluation_data_entity):
            try:
                image = data_instance["image_file"]
                load_temp = Image.open(data_instance["image_file"]).convert("RGB")
            except:
                continue
            question = data_instance["input"]
            conversation = [
                {
                    "role": "User",
                    "content": "<image_placeholder>{}".format(question),
                    "images": [image],
                },
                {"role": "Assistant", "content": ""},
            ]
            pil_images = load_pil_images(conversation)
            prepare_inputs = processor(
                conversations=conversation, images=pil_images, force_batchify=True
            ).to(model.device)
            # run image encoder to get the image embeddings
            inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
            logits = model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=4,
                do_sample=False,
                use_cache=True,
            ).logits[0, -1, :]
            choice_logits = [logits[idx].item() for idx in choice_ids]
            choice_probs = torch.softmax(torch.tensor(choice_logits), dim=-1)
            predict_answer = int(choices[torch.argmax(choice_probs)])
            answers.append(data_instance["answer"])
            predicts.append(predict_answer)
            # print(choice_logits)
        ent_acc = accuracy_score(y_true=answers, y_pred=predicts)
        print("entity", ent_acc)
        result_map[model_name]["entity"] = ent_acc
    if args.relation_task:
        answers = []
        predicts = []
        for data_instance in tqdm(evaluation_data_relation):
            try:
                image = data_instance["image_file"]
                load_temp = Image.open(data_instance["image_file"]).convert("RGB")
            except:
                continue
            question = data_instance["input"]
            conversation = [
                {
                    "role": "User",
                    "content": "<image_placeholder>{}".format(question),
                    "images": [image],
                },
                {"role": "Assistant", "content": ""},
            ]
            pil_images = load_pil_images(conversation)
            prepare_inputs = processor(
                conversations=conversation, images=pil_images, force_batchify=True
            ).to(model.device)
            # run image encoder to get the image embeddings
            inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
            logits = model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=4,
                do_sample=False,
                use_cache=True,
            ).logits[0, -1, :]
            choice_logits = [logits[idx].item() for idx in choice_ids]
            choice_probs = torch.softmax(torch.tensor(choice_logits), dim=-1)
            predict_answer = int(choices[torch.argmax(choice_probs)])
            answers.append(data_instance["answer"])
            predicts.append(predict_answer)
        ent_acc = accuracy_score(y_true=answers, y_pred=predicts)
        print("relation", ent_acc)
        result_map[model_name]["relation"] = ent_acc


def model_dispatch(model_type, model_name, args):
    if model_type == "deepseek-vl":
        run_deepseek_vl(model_type, model_name, args)
    else:
        raise NotImplementedError


def run_inference(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    model = args.model_type
    if model not in model_example_map:
        raise ValueError(f"Model type {model} is not supported.")
    if args.model_used == "all":
        all_model_names = model_path_map[model].keys()
        for model_name in all_model_names:
            model_dispatch(model, model_name, args)
    else:
        model_dispatch(model, model_name, args)
    for model in result_map.keys():
        print(model, result_map[model])
    


if __name__ == "__main__":
    parser = load_args()
    args = parser.parse_args()
    run_inference(args=args)
