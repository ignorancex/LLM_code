import json
import multiprocessing
import os
from argparse import ArgumentParser
from multiprocessing import Manager, Pool, Queue

import torch
from mplug_docowl.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from mplug_docowl.conversation import conv_templates
from mplug_docowl.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from mplug_docowl.model.builder import load_pretrained_model
from mplug_docowl.processor import DocProcessor
from tqdm import tqdm
from transformers import TextStreamer


# https://github.com/X-PLUG/mPLUG-DocOwl/blob/main/DocOwl1.5/docowl_infer.py
def split_list(lst, n):
    length = len(lst)
    avg = length // n  # 每份的大小
    result = []  # 存储分割后的子列表
    for i in range(n - 1):
        result.append(lst[i * avg : (i + 1) * avg])
    result.append(lst[(n - 1) * avg :])
    return result


def save_json(json_list, save_path):
    with open(save_path, "w", encoding="utf-8") as file:
        json.dump(json_list, file, indent=4)


def _get_args():
    parser = ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="./OCRBench_Images")
    parser.add_argument("--output_folder", type=str, default="./results")
    parser.add_argument("--OCRBench_file", type=str, default="./OCRBench/OCRBench.json")
    parser.add_argument("--model_path", type=str, default="mPLUG/DocOwl1.5")
    parser.add_argument("--save_name", type=str, default="mplug-DocOwl1.5")
    parser.add_argument("--conv_mode", type=str, default="mplug_owl2")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    return args


OCRBench_score = {
    "Regular Text Recognition": 0,
    "Irregular Text Recognition": 0,
    "Artistic Text Recognition": 0,
    "Handwriting Recognition": 0,
    "Digit String Recognition": 0,
    "Non-Semantic Text Recognition": 0,
    "Scene Text-centric VQA": 0,
    "Doc-oriented VQA": 0,
    "Key Information Extraction": 0,
    "Handwritten Mathematical Expression Recognition": 0,
}
AllDataset_score = {
    "IIIT5K": 0,
    "svt": 0,
    "IC13_857": 0,
    "IC15_1811": 0,
    "svtp": 0,
    "ct80": 0,
    "cocotext": 0,
    "ctw": 0,
    "totaltext": 0,
    "HOST": 0,
    "WOST": 0,
    "WordArt": 0,
    "IAM": 0,
    "ReCTS": 0,
    "ORAND": 0,
    "NonSemanticText": 0,
    "SemanticText": 0,
    "STVQA": 0,
    "textVQA": 0,
    "ocrVQA": 0,
    "ESTVQA": 0,
    "ESTVQA_cn": 0,
    "docVQA": 0,
    "infographicVQA": 0,
    "ChartQA": 0,
    "ChartQA_Human": 0,
    "FUNSD": 0,
    "SROIE": 0,
    "POIE": 0,
    "HME100k": 0,
}
num_all = {
    "IIIT5K": 0,
    "svt": 0,
    "IC13_857": 0,
    "IC15_1811": 0,
    "svtp": 0,
    "ct80": 0,
    "cocotext": 0,
    "ctw": 0,
    "totaltext": 0,
    "HOST": 0,
    "WOST": 0,
    "WordArt": 0,
    "IAM": 0,
    "ReCTS": 0,
    "ORAND": 0,
    "NonSemanticText": 0,
    "SemanticText": 0,
    "STVQA": 0,
    "textVQA": 0,
    "ocrVQA": 0,
    "ESTVQA": 0,
    "ESTVQA_cn": 0,
    "docVQA": 0,
    "infographicVQA": 0,
    "ChartQA": 0,
    "ChartQA_Human": 0,
    "FUNSD": 0,
    "SROIE": 0,
    "POIE": 0,
    "HME100k": 0,
}


def eval_worker(args, data, eval_id, output_queue):
    print(f"Process {eval_id} start.")
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, _, _ = load_pretrained_model(
        args.model_path,
        None,
        model_name,
        load_8bit=False,
        load_4bit=False,
        device=f"cuda:{eval_id}",
    )

    doc_image_processor = DocProcessor(
        image_size=448,
        anchors="grid_9",
        add_global_img=True,
        add_textual_crop_indicator=True,
    )

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    for i in tqdm(range(len(data))):
        img_path = os.path.join(args.image_folder, data[i]["image_path"])
        qs = data[i]["question"]
        if data[i].get("predict", 0) != 0:
            print(f"{img_path} predict exist, continue.")
            continue

        image_tensor, patch_positions, text = doc_image_processor(
            images=img_path, query="<|image|>" + qs
        )
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        patch_positions = patch_positions.to(model.device)

        conv = conv_templates["mplug_owl2"].copy()
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(model.device)
        )

        stop_str = conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                patch_positions=patch_positions,
                do_sample=False,
                temperature=1.0,
                max_new_tokens=512,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1] :]).strip()
        data[i]["predict"] = outputs
    output_queue.put({eval_id: data})
    print(f"Process {eval_id} has completed.")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    args = _get_args()

    if os.path.exists(os.path.join(args.output_folder, f"{args.save_name}.json")):
        data_path = os.path.join(args.output_folder, f"{args.save_name}.json")
        print(
            f"output_path:{data_path} exist! Only generate the results that were not generated in {data_path}."
        )
    else:
        data_path = args.OCRBench_file

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data_list = split_list(data, args.num_workers)
    output_queue = Manager().Queue()

    pool = Pool(processes=args.num_workers)
    for i in range(len(data_list)):
        # pool.apply(eval_worker, args=(args, data_list[i], i, output_queue))
        pool.apply_async(eval_worker, args=(args, data_list[i], i, output_queue))
    pool.close()
    pool.join()

    results = {}
    while not output_queue.empty():
        result = output_queue.get()
        results.update(result)
    data = []
    for i in range(len(data_list)):
        data.extend(results[i])

    for i in range(len(data)):
        data_type = data[i]["type"]
        dataset_name = data[i]["dataset_name"]
        answers = data[i]["answers"]
        if data[i].get("predict", 0) == 0:
            continue
        predict = data[i]["predict"]
        data[i]["result"] = 0
        if dataset_name == "HME100k":
            if type(answers) == list:
                for j in range(len(answers)):
                    answer = answers[j].strip().replace("\n", " ").replace(" ", "")
                    predict = predict.strip().replace("\n", " ").replace(" ", "")
                    if answer in predict:
                        data[i]["result"] = 1
            else:
                answers = answers.strip().replace("\n", " ").replace(" ", "")
                predict = predict.strip().replace("\n", " ").replace(" ", "")
                if answers in predict:
                    data[i]["result"] = 1
        else:
            if type(answers) == list:
                for j in range(len(answers)):
                    answer = answers[j].lower().strip().replace("\n", " ")
                    predict = predict.lower().strip().replace("\n", " ")
                    if answer in predict:
                        data[i]["result"] = 1
            else:
                answers = answers.lower().strip().replace("\n", " ")
                predict = predict.lower().strip().replace("\n", " ")
                if answers in predict:
                    data[i]["result"] = 1
    save_json(data, os.path.join(args.output_folder, f"{args.save_name}.json"))
    if len(data) == 1000:
        for i in range(len(data)):
            if data[i].get("result", 100) == 100:
                continue
            OCRBench_score[data[i]["type"]] += data[i]["result"]
        recognition_score = (
            OCRBench_score["Regular Text Recognition"]
            + OCRBench_score["Irregular Text Recognition"]
            + OCRBench_score["Artistic Text Recognition"]
            + OCRBench_score["Handwriting Recognition"]
            + OCRBench_score["Digit String Recognition"]
            + OCRBench_score["Non-Semantic Text Recognition"]
        )
        Final_score = (
            recognition_score
            + OCRBench_score["Scene Text-centric VQA"]
            + OCRBench_score["Doc-oriented VQA"]
            + OCRBench_score["Key Information Extraction"]
            + OCRBench_score["Handwritten Mathematical Expression Recognition"]
        )
        print("###########################OCRBench##############################")
        print(f"Text Recognition(Total 300):{recognition_score}")
        print("------------------Details of Recognition Score-------------------")
        print(
            f"Regular Text Recognition(Total 50): {OCRBench_score['Regular Text Recognition']}"
        )
        print(
            f"Irregular Text Recognition(Total 50): {OCRBench_score['Irregular Text Recognition']}"
        )
        print(
            f"Artistic Text Recognition(Total 50): {OCRBench_score['Artistic Text Recognition']}"
        )
        print(
            f"Handwriting Recognition(Total 50): {OCRBench_score['Handwriting Recognition']}"
        )
        print(
            f"Digit String Recognition(Total 50): {OCRBench_score['Digit String Recognition']}"
        )
        print(
            f"Non-Semantic Text Recognition(Total 50): {OCRBench_score['Non-Semantic Text Recognition']}"
        )
        print("----------------------------------------------------------------")
        print(
            f"Scene Text-centric VQA(Total 200): {OCRBench_score['Scene Text-centric VQA']}"
        )
        print("----------------------------------------------------------------")
        print(f"Doc-oriented VQA(Total 200): {OCRBench_score['Doc-oriented VQA']}")
        print("----------------------------------------------------------------")
        print(
            f"Key Information Extraction(Total 200): {OCRBench_score['Key Information Extraction']}"
        )
        print("----------------------------------------------------------------")
        print(
            f"Handwritten Mathematical Expression Recognition(Total 100): {OCRBench_score['Handwritten Mathematical Expression Recognition']}"
        )
        print("----------------------Final Score-------------------------------")
        print(f"Final Score(Total 1000): {Final_score}")
    else:
        for i in range(len(data)):
            num_all[data[i]["dataset_name"]] += 1
            if data[i].get("result", 100) == 100:
                continue
            AllDataset_score[data[i]["dataset_name"]] += data[i]["result"]
        for key in AllDataset_score.keys():
            print(f"{key}: {AllDataset_score[key]/float(num_all[key])}")
