import torch
import transformers
import json
import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from trace.conversation import conv_templates, SeparatorStyle
from trace.constants import DEFAULT_MMODAL_TOKEN, MMODAL_TOKEN_INDEX
from trace.mm_utils import get_model_name_from_path, tokenizer_MMODAL_token_all, process_video, process_image, KeywordsStoppingCriteria
from trace.model.builder import load_pretrained_model


def inference(args):
    # Video Inference
    paths = args.video_paths
    questions = args.questions
    modal_list = ['video']

    # 1. Initialize the model.
    model_path = args.model_path
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name)
    model = model.to('cuda')
    conv_mode = 'llama_2'

    # 2. Visual preprocess (load & transform image or video).
    if modal_list[0] == 'video':
        tensor, video_timestamps = process_video(paths[0], processor, model.config.image_aspect_ratio, num_frames=64)
        tensor = tensor.to(dtype=torch.float16, device='cuda', non_blocking=True)
        default_mm_token = DEFAULT_MMODAL_TOKEN["VIDEO"]
        modal_token_index = MMODAL_TOKEN_INDEX["VIDEO"]
    else:
        tensor = process_image(paths[0], processor, model.config.image_aspect_ratio)[0].to(dtype=torch.float16, device='cuda', non_blocking=True)
        default_mm_token = DEFAULT_MMODAL_TOKEN["IMAGE"]
        modal_token_index = MMODAL_TOKEN_INDEX["IMAGE"]

    tensor = [tensor]
    video_timestamps = [video_timestamps]
    heads = [1]

    # 3. Text preprocess (tag process & generate prompt).
    question = default_mm_token + "\n" + questions[0]
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    prompt += '<sync>'
    print(prompt)
    input_ids = tokenizer_MMODAL_token_all(prompt, tokenizer, return_tensors='pt').unsqueeze(0).to('cuda')
    attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()
    stop_str = conv.sep if conv.sep_style in [SeparatorStyle.SINGLE] else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    do_sample = True

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_masks,
            images_or_videos=tensor,
            modal_list=modal_list,
            do_sample=do_sample,
            temperature=0.2 if do_sample else 0.0,
            max_new_tokens=1024,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            video_timestamps=video_timestamps,
            heads=heads
        )

    outputs = {
        'timestamps': [],
        'scores': [],
        'captions': [],
    }
    cur_timestamps = []
    cur_timestamp = []
    cur_scores = []
    cur_score = []
    cur_caption = []
    for idx in output_ids[0]:
        if idx <= 32000:
            if idx == 32000:
                new_caption = tokenizer.decode(cur_caption, skip_special_tokens=True)
                outputs['captions'].append(new_caption)
                cur_caption = []
            else:
                cur_caption.append(idx)
        elif idx <= 32013: # 32001 <sync>; 32002 <sep>
            if idx == 32001:
                if len(cur_timestamp) > 0:
                    cur_timestamps.append(float(''.join(cur_timestamp)))
                outputs['timestamps'].append(cur_timestamps)
                cur_timestamps = []
                cur_timestamp = []
            elif idx == 32002:
                if len(cur_timestamp) > 0:
                    cur_timestamps.append(float(''.join(cur_timestamp)))
                cur_timestamp = []
            else:
                cur_timestamp.append(model.get_model().time_tokenizer.decode(idx - 32001))
        else: # 32014 <sync>; 32015 <sep>
            if idx == 32014:
                if len(cur_score) > 0:
                    cur_scores.append(float(''.join(cur_score)))
                outputs['scores'].append(cur_scores)
                cur_scores = []
                cur_score = []
            elif idx == 32015:
                if len(cur_score) > 0:
                    cur_scores.append(float(''.join(cur_score)))
                cur_score = []
            else:
                cur_score.append(model.get_model().score_tokenizer.decode(idx - 32014))
    if len(cur_caption):
        outputs['captions'].append(tokenizer.decode(cur_caption, skip_special_tokens=True))

    try:
        results = []
        for i in range(len(outputs['timestamps'])):
            output = {
                'video': paths[0].split("/")[-1][:-4] + "_mask.mp4",
                'segment': f"{outputs['timestamps'][i][0]}_{outputs['timestamps'][i][1]}",
                'question': "",
                'answer': outputs['captions'][i],
            }
            results.append(output)

        with open(f'./results/{paths[0].split("/")[-1].split(".")[0]}_boundary.json', 'w') as f:
            json.dump(results, f)
    
    except Exception as e:
        print(e)
        print("Failed to save the output to a json file.")
        with open(f'./results/{paths[0].split("/")[-1].split(".")[0]}_boundary.json', 'w') as f:
            json.dump([{"video": paths[0].split("/")[-1], "segment": f"0.0_{video_timestamps[0][1]}", "question": "", "answer": ""}], f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for boundary detection.")
    parser.add_argument("--video_paths", nargs='+', required=True, help="Paths to the input video files.")
    parser.add_argument("--questions", nargs='+', required=True, help="Questions for video inference.")
    parser.add_argument("--model_path", required=True, help="Path to the pretrained model.")
    args = parser.parse_args()

    inference(args)
