from egorag.utils.mm_process import merge_videos

try:
    import base64
    import os
    from io import BytesIO
    from typing import Any, Dict, List

    import cv2
    import decord
    import numpy as np
    import requests
    from egorag.models.base import BaseQueryModel
    from PIL import Image
except:
    print(
        "Please install the required packages: pip install requests, pip install decord, pip install Pillow"
    )

import argparse
import math
import os
import re
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import transformers
from decord import VideoReader, cpu
from egorag.utils.util import time_to_frame_idx
from longva.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from longva.conversation import SeparatorStyle, conv_templates
from longva.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from longva.model.builder import load_pretrained_model
from PIL import Image
from transformers import AutoConfig


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def preprocess_qwen(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    max_len=2048,
    system_message: str = "You are a helpful assistant.",
) -> Dict:
    roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}

    im_start, im_end = tokenizer.additional_special_tokens_ids[:2]
    nl_tokens = tokenizer("\n").input_ids
    _system = tokenizer("system").input_ids + nl_tokens
    _user = tokenizer("user").input_ids + nl_tokens
    _assistant = tokenizer("assistant").input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []

    source = sources
    if roles[source[0]["from"]] != roles["human"]:
        source = source[1:]

    input_id, target = [], []
    system = (
        [im_start]
        + _system
        + tokenizer(system_message).input_ids
        + [im_end]
        + nl_tokens
    )
    input_id += system
    target += [im_start] + [IGNORE_INDEX] * (len(system) - 3) + [im_end] + nl_tokens
    assert len(input_id) == len(target)
    for j, sentence in enumerate(source):
        role = roles[sentence["from"]]
        if (
            has_image
            and sentence["value"] is not None
            and "<image>" in sentence["value"]
        ):
            num_image = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            texts = sentence["value"].split("<image>")
            _input_id = tokenizer(role).input_ids + nl_tokens
            for i, text in enumerate(texts):
                _input_id += tokenizer(text).input_ids
                if i < len(texts) - 1:
                    _input_id += [IMAGE_TOKEN_INDEX] + nl_tokens
            _input_id += [im_end] + nl_tokens
            assert sum([i == IMAGE_TOKEN_INDEX for i in _input_id]) == num_image
        else:
            if sentence["value"] is None:
                _input_id = tokenizer(role).input_ids + nl_tokens
            else:
                _input_id = (
                    tokenizer(role).input_ids
                    + nl_tokens
                    + tokenizer(sentence["value"]).input_ids
                    + [im_end]
                    + nl_tokens
                )
        input_id += _input_id
        if role == "<|im_start|>user":
            _target = (
                [im_start]
                + [IGNORE_INDEX] * (len(_input_id) - 3)
                + [im_end]
                + nl_tokens
            )
        elif role == "<|im_start|>assistant":
            _target = (
                [im_start]
                + [IGNORE_INDEX] * len(tokenizer(role).input_ids)
                + _input_id[len(tokenizer(role).input_ids) + 1 : -2]
                + [im_end]
                + nl_tokens
            )
        else:
            raise NotImplementedError
        target += _target

    input_ids.append(input_id)
    targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    return input_ids


class Longva(BaseQueryModel):
    def __init__(
        self,
        pretrained: str = "",
        conv_template="qwen_1_5",
        overwrite: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_path = pretrained
        self.model_name = get_model_name_from_path(self.model_path)
        self.conv_template = conv_template
        self.overwrite = overwrite
        (
            self.tokenizer,
            self.model,
            self.image_processor,
            self._max_length,
        ) = load_pretrained_model(pretrained, None, "llava_qwen", device_map="cuda:0")

        self._config = self.model.config
        self.model.to("cuda").eval()

    def process_video(
        self,
        video_path: str,
        video_start_time: int,
        start_time: int,
        end_time: int,
        fps=1,
        max_frames_num=32,
    ):
        # Initialize video reader
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=1)
        total_frame_num = len(vr)

        # Get the actual FPS of the video
        video_fps = vr.get_avg_fps()

        # Convert time to frame index based on the actual video FPS
        video_start_frame = int(time_to_frame_idx(video_start_time, video_fps))
        start_frame = int(time_to_frame_idx(start_time, video_fps))
        end_frame = int(time_to_frame_idx(end_time, video_fps))

        # Ensure the end time does not exceed the total frame number
        if end_frame - start_frame > total_frame_num:
            end_frame = total_frame_num + start_frame

        # Adjust start_frame and end_frame based on video start time
        start_frame -= video_start_frame
        end_frame -= video_start_frame
        start_frame = int(round(start_frame))
        end_frame = int(round(end_frame))

        # Sample frames based on the provided fps (e.g., 1 frame per second)
        frame_idx = [
            i
            for i in range(start_frame, end_frame)
            if (i - start_frame) % int(video_fps / fps) == 0
        ]
        uniform_sampled_frames = np.linspace(
            start_frame, end_frame - 1, max_frames_num, dtype=int
        )
        frame_idx = uniform_sampled_frames.tolist()

        # Get the video frames for the sampled indices
        video = vr.get_batch(frame_idx).asnumpy()

        # Return processed video and the corresponding frame indices
        return {
            "processed_video": video,
            "frame_idx": frame_idx,
            "start_time": start_time,
            "end_time": end_time,
        }

    def inference_video(
        self,
        video_path,
        video_start_time,
        start_time,
        end_time,
        human_query,
        system_message="",
    ):
        processed_data = self.process_video(
            video_path, video_start_time, start_time, end_time
        )

        image_arrays = processed_data["processed_video"]
        frame_idx = processed_data["frame_idx"]

        video = [Image.fromarray(frame) for frame in image_arrays]
        video = np.array(video)

        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<image>\n{system_message+human_query}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self.model.device)
        )

        video_tensor = self.image_processor.preprocess(video, return_tensors="pt")[
            "pixel_values"
        ].to(self.model.device, dtype=torch.float16)
        gen_kwargs = {
            "do_sample": True,
            "temperature": 0.5,
            "top_p": None,
            "num_beams": 1,
            "use_cache": True,
            "max_new_tokens": 1024,
        }
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids, images=[video_tensor], modalities=["video"], **gen_kwargs
            )
        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
            0
        ].strip()

        return outputs
