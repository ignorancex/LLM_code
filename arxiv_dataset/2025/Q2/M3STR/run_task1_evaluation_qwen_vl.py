import argparse
import os
import json
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from PIL import Image
from utils import *
from sklearn.metrics import accuracy_score
from collections import defaultdict
from tqdm import tqdm
from load_models import *
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info


result_map = defaultdict(dict)

model_path_map = {
    "qwen2-vl": {
        "qwen2-vl-2b": "Qwen/Qwen2-VL-2B-Instruct",
        "qwen2-vl-7b": "Qwen/Qwen2-VL-7B-Instruct",
        "qwen2-vl-72b": "Qwen/Qwen2-VL-72B-Instruct-AWQ"
    },
    "qwen2.5-vl": {
        "qwen2.5-vl-72b": "Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
        "qwen2.5-vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct",
        "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    }
}


def load_qwen_vl(model_name):
    model_type = "qwen2.5-vl" if "2.5" in model_name else "qwen2-vl"
    model_path = model_path_map[model_type][model_name]
    llm = LLM(
        model=model_path,
        limit_mm_per_prompt={"image": 10, "video": 10},
        gpu_memory_utilization=0.9,
        max_model_len=8192
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return llm, processor


def get_qwen_prompt(question, image, processor):
    messages = [
        {
            "role": "system", 
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                    "min_pixels": 224 * 224,
                    "max_pixels": 1280 * 28 * 28,
                },
                {"type": "text", "text": question},
            ],
        },
    ]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, _, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
        # FPS will be returned in video_kwargs
        "mm_processor_kwargs": video_kwargs,
    }
    return llm_inputs



def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type',
                        type=str,
                        default="llava",
                        choices=["qwen2-vl", "qwen2.5-vl"])
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



def run_inference_with_vllm_on_single_model(model, model_name, args):
    llm, processor = load_qwen_vl(model_name)
    evaluation_data_entity, evaluation_data_relation = load_task1_dataset()
    guided_decode_params = GuidedDecodingParams(
        choice=["2", "3", "4", "5", "6", "7", "8", "9"]
    )
    # the temperature should be set to 0 for fair evaluation
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=4,
        stop_token_ids=[],
        guided_decoding=guided_decode_params
    )
    # Batch inference for Entity Counting
    if args.entity_task:
        answers = []
        predicts = []
        for data_instance in tqdm(evaluation_data_entity):
            try:
                image = data_instance["image_file"]
                load_temp = Image.open(data_instance["image_file"])
            except:
                continue
            question = data_instance["input"]
            input_data = get_qwen_prompt(question, image, processor)
            answers.append(data_instance["answer"])
            outputs = llm.generate([input_data], sampling_params=sampling_params, use_tqdm=False)
            for o in outputs:
                generated_text = o.outputs[0].text
                predicts.append(int(generated_text))
        ent_acc = accuracy_score(y_true=answers, y_pred=predicts)
        print("entity", ent_acc)
        result_map[model_name]["entity"] = ent_acc
        result_map[model_name]["entity_results"] = {
            "answer": answers,
            "predict": predicts
        }
    if args.relation_task:
        # Batch inference for Relation Counting
        answers = []
        predicts = []
        for data_instance in tqdm(evaluation_data_relation):
            try:
                # print(data_instance["image_file"])
                image = data_instance["image_file"]
                load_temp = Image.open(data_instance["image_file"])
            except:
                continue
            question = data_instance["input"]
            input_data = get_qwen_prompt(question, image, processor)
            answers.append(data_instance["answer"])
            outputs = llm.generate([input_data], sampling_params=sampling_params, use_tqdm=False)
            for o in outputs:
                generated_text = o.outputs[0].text
                predicts.append(int(generated_text))
        rel_acc = accuracy_score(y_true=answers, y_pred=predicts)
        print("relation", rel_acc)
        result_map[model_name]["relation"] = rel_acc
        result_map[model_name]["relation_results"] = {
            "answer": answers,
            "predict": predicts
        }




def model_dispatch(model_type, model_name, args):
    run_inference_with_vllm_on_single_model(model_type, model_name, args)


def run_inference(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    model = args.model_type
    if "qwen" not in model:
        raise ValueError(f"Model type {model} is not supported.")
    if args.model_used == "all":
        all_model_names = model_path_map[model].keys()
        for model_name in all_model_names:
            model_dispatch(model, model_name, args)
    else:
        model_dispatch(model, model_name, args)
    for model in result_map.keys():
        print(model, "Entity Count:", result_map[model]["entity"])
        print(model, "Relation Count:", result_map[model]["relation"])
    json.dump(result_map, open("{}_result.json".format(model), "w"), ensure_ascii=False)
    


if __name__ == "__main__":
    parser = load_args()
    args = parser.parse_args()
    run_inference(args=args)
