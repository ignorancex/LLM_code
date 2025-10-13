import dotenv

dotenv.load_dotenv(override=True)
import base64
import json
import os
import string
from io import BytesIO
from pathlib import Path
from pprint import pformat, pprint

import click
import datasets
import matplotlib.pyplot as plt
import pandas as pd
from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value
from datasets import Image as ImageData
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)


@click.command()
@click.option(
    "--local_dir", type=Path, default="data/processed/pmc_vqa_limit_tokens_2048"
)
@click.option("--data_source", type=str, default="med-vlrm/PMC-VQA-EasyR1")
@click.option("--max_image_size", type=int, default=1024)
@click.option("--max_token_length", type=int, default=2048)
@click.option("--num_proc", type=int, default=32)
def main(local_dir, data_source, max_image_size, max_token_length, num_proc):
    dataset = datasets.load_dataset(data_source)
    dataset = dataset["train"]

    # filter invalid image size
    raw_dataset_size = len(dataset)
    dataset = dataset.filter(has_valid_image_size, num_proc=num_proc)
    filtered_dataset_size = len(dataset)
    num_filtered = raw_dataset_size - filtered_dataset_size
    print(
        f"Filtered dataset size: {filtered_dataset_size} from {raw_dataset_size}, removed {num_filtered} examples with invalid image size."
    )

    # resize images up to max_image_size
    dataset = dataset.map(
        build_resize_sample_upto_fn(max_image_size),
        num_proc=num_proc,
        desc=f"Resizing images up to {max_image_size}x{max_image_size}",
    )

    # filter max token length
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    instruction_following = r"You will solve a problem/request. You should provide your thoughts within <think> </think> tags before providing the answer.\nWrite your final answer within <answer> </answer> tags. Here is the question:\n\n"

    dataset = dataset.filter(
        build_filter_token_example_fn(
            processor, instruction_following, max_token_length
        ),
        num_proc=num_proc,
        desc=f"Filtering examples with more than {max_token_length} tokens",
    )
    filtered_by_token_dataset_size = len(dataset)
    num_filtered_by_token = filtered_dataset_size - filtered_by_token_dataset_size
    print(
        f"Filtered dataset size: {filtered_by_token_dataset_size} from {filtered_dataset_size}, removed {num_filtered_by_token} examples with more than {max_token_length} tokens."
    )

    # save dataset to local directory
    dataset = DatasetDict({"train": dataset})
    dataset.save_to_disk(local_dir, num_proc=num_proc)
    print(f"Dataset saved to {local_dir}")

    # dataset.push_to_hub("med-vlrm/PMC-VQA-EasyR1-Filtered-Resized")
    # print(f"Dataset pushed to hub: med-vlrm/PMC-VQA-EasyR1-Filtered-Resized")


def build_filter_token_example_fn(processor, instruction_following, max_token_length):
    def tokenize_example(example, verbose=False):
        problem = example.pop("problem")
        prompt = problem + "\n\n" + instruction_following
        answer = example.pop("answer")
        # only a simple image
        image = example.pop("image")

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": pil_to_base64_data_uri(image),
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        if verbose:
            print(f"image shape: {image.size}")
            print(f"image inputs shape: {image_inputs[0].size}")
            print(f"inputs pixel shape: {inputs['pixel_values'].shape}")
        return inputs

    def filter_token_example(example):
        inputs = tokenize_example(example)
        input_ids = inputs["input_ids"]
        num_total_tokens = input_ids[0].numel()
        if num_total_tokens > max_token_length:
            return False
        return True

    return filter_token_example


def pil_to_base64_data_uri(img: Image.Image, format="JPEG"):
    buffer = BytesIO()
    img.save(buffer, format=format)
    img_bytes = buffer.getvalue()
    base64_str = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/base64,{base64_str}"


# resize the longest dim to 384
def resize_longest_dim(image, target_size=1024):
    width, height = image.size
    if width > height:
        new_width = target_size
        new_height = int(height * (target_size / width))
    else:
        new_height = target_size
        new_width = int(width * (target_size / height))
    return image.resize((new_width, new_height), Image.BILINEAR)


def resize_sample_upto(example, size=1024):
    example["image"] = resize_longest_dim(example["image"], size)
    return example


def build_resize_sample_upto_fn(size=1024):
    def resize_sample_upto_fn(example):
        image = example["image"]
        width, height = image.size
        if width > size or height > size:
            # Resize the image to the specified size while maintaining aspect ratio
            example["image"] = resize_longest_dim(image, size)
        return example

    return resize_sample_upto_fn


# https://github.com/Yuxiang-Lai117/Med-R1/blob/53e46ba24e04d7d7705db4750551786a93e96493/src/r1-v/local_scripts/prepare_hf_data.py#L144
def has_valid_image_size(example):
    # for Qwen2-VL-2B's processor requirement
    # Assuming the image is in a format that can be checked for dimensions
    # You might need to adjust this depending on how the image is stored in your dataset
    try:
        image = example["image"]  # or however your image is accessed
        if isinstance(image, dict) and "height" in image and "width" in image:
            return image["height"] >= 28 and image["width"] >= 28
        # If image is a PIL Image or similar
        return image.height >= 28 and image.width >= 28
    except Exception:
        return False


if __name__ == "__main__":
    main()
