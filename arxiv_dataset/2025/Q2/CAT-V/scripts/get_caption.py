import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys_paths = [
    os.path.abspath(os.path.join(script_dir, '../'))
]
print(sys_paths)
for sys_path in sys_paths:
    if sys_path not in sys.path:
        sys.path.append(sys_path)
import numpy as np
from tqdm import tqdm
import torch
import torchvision.transforms as T
import json
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
from eval_utils import get_answers_output_path, gen_QAs_dataloader, unpack_QAs, basic_parser

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = float(bound[0]), float(bound[1])
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, int(start * fps))
    end_idx = min(int(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list
if __name__ == "__main__":
    parser = basic_parser()
    args = parser.parse_args()
    answers_output_path = get_answers_output_path(args)
    dataloader = gen_QAs_dataloader(args)
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    generation_config = dict(max_new_tokens=1024, do_sample=False)
    answers = []
    event_list = []
    # duration = 0
    for idx, (QA_data, _) in enumerate(tqdm(dataloader)):
        final_question, video_path, short_answer, video, segment, question = unpack_QAs(QA_data)
        event = f'From {segment[0]} to {segment[1]}s, {short_answer}'
        event_list.append(event)
        # duration = max(duration, float(segment[1]))

    # print(event_list)
    pixel_values, num_patches_list = load_video(video_path, num_segments=args.max_frames_num, max_num=1)
    n_frames = len(num_patches_list)
    pixel_values = pixel_values.to(torch.bfloat16).cuda()
    video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
    video_prefix += ''.join([f'{event}\n' for event in event_list])
    question_tmp = video_prefix + final_question
    print(question_tmp)
    try:
        with torch.inference_mode():
            response = model.chat(tokenizer, pixel_values, question_tmp, generation_config)
        print(response)
        answers.append({
            "video": video,
            "segment": segment,
            "question": question,
            # "options": options, 
            # "task_class": task_class, 
            "short_answer": short_answer,
            "model_answer": response           
        })
    except Exception as e:
        print(f"Error encountered at idx {idx}: {e}")
        answers.append({
            "video": video,
            "segment": segment,
            "question": question,
            # "options": options, 
            # "task_class": task_class, 
            "short_answer": short_answer,
            "model_answer": '<error_processing>'           
        })
    # exit()



    # for idx, (QA_data, _) in enumerate(tqdm(dataloader)):
    #     # print(QA_data)
    #     # continue
    #     try:
    #         final_question, video_path, short_answer, \
    #             video, segment, question = unpack_QAs(QA_data)
    #         # print(video)
    #         # print(segment, question)
    #         print(f'From {segment[0]} to {segment[1]}s,', short_answer)
    #         print(final_question)
    #         if args.provide_boundaries:
    #             pixel_values, num_patches_list = load_video(video_path, bound=segment, num_segments=args.max_frames_num, max_num=1)
    #         else:
    #             pixel_values, num_patches_list = load_video(video_path, num_segments=args.max_frames_num, max_num=1)
    #         # exit()
    #         print(num_patches_list)
    #         exit()
    #         pixel_values = pixel_values.to(torch.bfloat16).cuda()
    #         video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
    #         question_tmp = video_prefix + final_question
    #         # Frame1: <image>\nFrame2: <image>\n...\nFrame8: <image>\n{question}
    #         with torch.inference_mode():
    #             response = model.chat(tokenizer, pixel_values, question_tmp, generation_config,
    #                             num_patches_list=num_patches_list)
    #         print(response)
    #         answers.append({
    #             "video": video,
    #             "segment": segment,
    #             "question": question,
    #             # "options": options, 
    #             # "task_class": task_class, 
    #             "short_answer": short_answer,
    #             "model_answer": response           
    #         })

    #     except Exception as e:
    #         print(f"Error encountered at idx {idx}: {e}")
    #         answers.append({
    #             "video": video,
    #             "segment": segment,
    #             "question": question,
    #             # "options": options, 
    #             # "task_class": task_class, 
    #             "short_answer": short_answer,
    #             "model_answer": '<error_processing>'           
    #         })
    # exit()
    with open(answers_output_path, 'w', encoding='utf-8') as f:
        json.dump(answers, f, indent=4, ensure_ascii=False)
    with open(args.final_json_path, 'w', encoding='utf-8') as f:
        json.dump(answers, f, indent=4, ensure_ascii=False)
