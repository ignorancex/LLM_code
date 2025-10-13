import argparse
import os
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from PIL import Image
from utils import *
from sklearn.metrics import accuracy_score
from collections import defaultdict
from tqdm import tqdm
from load_models import *

result_map = defaultdict(dict)


model_example_map = {
    "llava": load_llava,
    # "llava-next": run_llava_next,
    "phi": load_phi3v,
    "chameleon": load_chameleon,
    "minicpm": run_minicpm,
    "instructblip": load_instruct_blip,
    "internvl": load_internvl,
    # "qwen_vl": run_qwen_vl,
}


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type',
                        type=str,
                        default="llava",
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


def run_inference_with_vllm_on_single_model(model, model_name, args):
    llm, prompt_template, stop_token_ids = model_example_map[model](model_name)
    dataset_dict = load_task2_dataset()
    guided_decode_params = GuidedDecodingParams(
        choice=["A", "B", "C", "D", "E"]
    )
    choice_id_map = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
        "E": 4
    }
    # the temperature should be set to 0 for fair evaluation
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=4,
        stop_token_ids=stop_token_ids,
        guided_decoding=guided_decode_params
    )
    # Batch inference
    for dataset_type in dataset_dict:
        dataset = dataset_dict[dataset_type]
        answers = []
        predicts = []
        for data_instance in tqdm(dataset):
            try:
                # print(data_instance["image_file"])
                image = Image.open(data_instance["image_file"])
            except:
                continue
            question = data_instance["input"]
            input_data = {
                "prompt": prompt_template.format(question),
                "multi_modal_data": {
                    "image": image
                },
            }
            answers.append(choice_id_map[data_instance["answer"]])
            outputs = llm.generate([input_data], sampling_params=sampling_params, use_tqdm=False)
            for o in outputs:
                generated_text = o.outputs[0].text
                predicts.append(choice_id_map[generated_text])
        accuracy = accuracy_score(y_true=answers, y_pred=predicts)
        print(model_name, dataset_type, accuracy)
        result_map[model_name][dataset_type] = accuracy
        result_map[model_name][dataset_type] = {
            "result": accuracy,
            "answer": answers,
            "predict": predicts
        }


def run_inference_with_instructblip(model_type, model_name, args):
    model, prompt_template, processor = model_example_map[model_type](model_name)
    dataset_dict = load_task3_dataset()
    # No -> 0, Yes -> 1
    tokenizer = processor.tokenizer
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
    for dataset_type in dataset_dict:
        print(dataset_type)
        dataset = dataset_dict[dataset_type]
        answers = []
        predicts = []
        for data_instance in tqdm(dataset):
            try:
                # print(data_instance["image_file"])
                image = Image.open(data_instance["image_file"]).convert("RGB")
            except:
                continue
            question = data_instance["input"]
            inputs = processor(image, question, return_tensors="pt").to("cuda")
            logits = model(**inputs).logits[0, -1, :]
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


def run_inference_with_internvl(model_type, model_name, args):
    llm, tokenizer, stop_token_ids = model_example_map[model_type](model_name)
    dataset_dict = load_task3_dataset()
    guided_decode_params = GuidedDecodingParams(
        choice=["A", "B", "C", "D", "E"]
    )
    choice_id_map = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
        "E": 4
    }
    # the temperature should be set to 0 for fair evaluation
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=4,
        stop_token_ids=stop_token_ids,
        guided_decoding=guided_decode_params
    )
    for dataset_type in dataset_dict:
        print(dataset_type)
        dataset = dataset_dict[dataset_type]
        answers = []
        predicts = []
        for data_instance in tqdm(dataset):
            try:
                # print(data_instance["image_file"])
                image = Image.open(data_instance["image_file"])
            except:
                continue
            question = data_instance["input"]
            messages = [{'role': 'user', 'content': "<image>\n{}".format(question)}]
            prompt = tokenizer.apply_chat_template(messages,
                        tokenize=False,
                        add_generation_prompt=True)
            input_data = {
                "prompt": prompt,
                "multi_modal_data": {
                    "image": image
                },
            }
            answers.append(choice_id_map[data_instance["answer"]])
            outputs = llm.generate([input_data], sampling_params=sampling_params, use_tqdm=False)
            for o in outputs:
                generated_text = o.outputs[0].text
                predicts.append(choice_id_map[generated_text])
        accuracy = accuracy_score(y_true=answers, y_pred=predicts)
        print(dataset_type, accuracy)
        result_map[model_name][dataset_type] = accuracy
        result_map[model_name][dataset_type] = {
            "result": accuracy,
            "answer": answers,
            "predict": predicts
        }



def run_inference_with_minicpm(model_type, model_name, args):
    llm, tokenizer, stop_token_ids = model_example_map[model_type](model_name)
    dataset_dict = load_task3_dataset()
    guided_decode_params = GuidedDecodingParams(
        choice=["A", "B", "C", "D", "E"]
    )
    choice_id_map = {
        "A": 0,
        "B": 1,
        "C": 2,
        "D": 3,
        "E": 4
    }
    # the temperature should be set to 0 for fair evaluation
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=4,
        stop_token_ids=stop_token_ids,
        guided_decoding=guided_decode_params
    )
    for dataset_type in dataset_dict:
        print(dataset_type)
        dataset = dataset_dict[dataset_type]
        answers = []
        predicts = []
        for data_instance in tqdm(dataset):
            try:
                # print(data_instance["image_file"])
                image = Image.open(data_instance["image_file"])
            except:
                continue
            question = data_instance["input"]
            messages = [{'role': 'user', 'content': "(<image>./</image>)\n{}".format(question)}]
            prompt = tokenizer.apply_chat_template(messages,
                        tokenize=False,
                        add_generation_prompt=True)
            input_data = {
                "prompt": prompt,
                "multi_modal_data": {
                    "image": image
                },
            }
            answers.append(choice_id_map[data_instance["answer"]])
            outputs = llm.generate([input_data], sampling_params=sampling_params, use_tqdm=False)
            for o in outputs:
                generated_text = o.outputs[0].text
                predicts.append(choice_id_map[generated_text])
        accuracy = accuracy_score(y_true=answers, y_pred=predicts)
        print(dataset_type, accuracy)
        result_map[model_name][dataset_type] = accuracy
        result_map[model_name][dataset_type] = {
            "result": accuracy,
            "answer": answers,
            "predict": predicts
        }



def model_dispatch(model_type, model_name, args):
    if model_type == "instructblip":
        run_inference_with_instructblip(model_type, model_name, args)
    elif model_type == "internvl":
        run_inference_with_internvl(model_type, model_name, args)
    elif model_type == "minicpm":
        run_inference_with_minicpm(model_type, model_name, args)
    else:
        run_inference_with_vllm_on_single_model(model_type, model_name, args)


def run_inference(args):
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
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
    json.dump(result_map, open("{}_task3_result.json".format(model), "w"), ensure_ascii=False)
    


if __name__ == "__main__":
    parser = load_args()
    args = parser.parse_args()
    run_inference(args=args)
