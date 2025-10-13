import argparse
import os
import torch
from PIL import Image
from utils import *
from sklearn.metrics import accuracy_score
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images



result_map = defaultdict(dict)

model_path_map = {
    "deepseek-vl2": {
        "deepseek-vl2-small": "Deepseek/deepseek-vl2-small",
        "deepseek-vl2-tiny": "Deepseek/deepseek-vl2-tiny"
    }
}


def load_deepseek_vl(model_name):
    model_path = model_path_map["deepseek-vl2"][model_name]
    processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
    model: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    )
    model = model.to(torch.bfloat16).cuda().eval()
    return model, tokenizer, processor



model_example_map = {
    "deepseek-vl2": load_deepseek_vl,
}


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type',
                        type=str,
                        default="deepseek-vl2",
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
    dataset_dict = load_task3_dataset()
    choices = ["A", "B", "C", "D", "E"]
    choice_ids = [tokenizer.encode(choice)[-1] for choice in choices] 
    print(choice_ids)
    choice_id_map = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
        "E": 4
    }
    model = model.eval()
    with torch.no_grad():
        for dataset_type in dataset_dict:
            dataset = dataset_dict[dataset_type]
            answers = []
            predicts = []
            for data_instance in tqdm(dataset):
                try:
                    image = data_instance["image_file"]
                    load_temp = Image.open(data_instance["image_file"]).convert("RGB")
                except:
                    continue
                question = data_instance["input"]
                conversation = [
                    {
                        "role": "<|User|>",
                        "content": "<image>\n<|ref|>{}<|/ref|>.".format(question),
                        "images": [image],
                    },
                    {"role": "<|Assistant|>", "content": ""},
                ]
                pil_images = load_pil_images(conversation)
                prepare_inputs = processor(
                    conversations=conversation, images=pil_images, force_batchify=True
                ).to(model.device)
                # run image encoder to get the image embeddings
                inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
                logits = model.language(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask
                ).logits[0, -1, :].cpu()
                choice_logits = [logits[idx].item() for idx in choice_ids]
                choice_probs = torch.softmax(torch.tensor(choice_logits), dim=-1)
                predict_answer = int(torch.argmax(choice_probs).item())
                answers.append(choice_id_map[data_instance["answer"]])
                predicts.append(predict_answer)
            accuracy = accuracy_score(y_true=answers, y_pred=predicts)
            print(model_name, dataset_type, accuracy)
            result_map[model_name][dataset_type] = accuracy
            result_map[model_name][dataset_type] = {
                "result": accuracy,
                "answer": answers,
                "predict": predicts
            }



def model_dispatch(model_type, model_name, args):
    if model_type == "deepseek-vl2":
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
