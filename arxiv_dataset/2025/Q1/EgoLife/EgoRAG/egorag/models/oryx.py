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
from llava.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_anyres_video_genli,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from PIL import Image
from transformers import AutoConfig

# sys.path.append("/mnt/lzy/llava-video")


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
    _assistant = tokenizer("assistant").input.ids + nl_tokens

    input_ids, targets = [], []

    source = sources
    if roles[source[0]["from"]] != roles["human"]:
        source = source[1:]

    input_id, target = [], []
    system = (
        [im_start]
        + _system
        + tokenizer(system_message).input.ids
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
            _input_id = tokenizer(role).input.ids + nl_tokens
            for i, text in enumerate(texts):
                _input_id += tokenizer(text).input.ids
                if i < len(texts) - 1:
                    _input_id += [IMAGE_TOKEN_INDEX] + nl_tokens
            _input_id += [im_end] + nl_tokens
            assert sum([i == IMAGE_TOKEN_INDEX for i in _input_id]) == num_image
        else:
            if sentence["value"] is None:
                _input_id = tokenizer(role).input.ids + nl_tokens
            else:
                _input_id = (
                    tokenizer(role).input.ids
                    + nl_tokens
                    + tokenizer(sentence["value"]).input.ids
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
                + [IGNORE_INDEX] * len(tokenizer(role).input.ids)
                + _input_id[len(tokenizer(role).input.ids) + 1 : -2]
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


class Oryx(BaseQueryModel):
    def __init__(
        self,
        pretrained: str = "/home/models/oryx-7b",
        conv_template="qwen_1_5",
        overwrite: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_path = pretrained
        self.model_name = get_model_name_from_path(self.model_path)
        self.conv_template = conv_template
        self.overwrite = overwrite
        if self.overwrite == True:
            overwrite_config = {}
            overwrite_config["mm_resampler_type"] = "dual_perceiver_region_avg"
            overwrite_config["patchify_video_feature"] = False
            overwrite_config["attn_implementation"] = (
                "sdpa" if torch.__version__ >= "2.1.2" else "eager"
            )

            cfg_pretrained = AutoConfig.from_pretrained(self.model_path)
        if "7b" in self.model_path:
            (
                self.tokenizer,
                self.model,
                self.image_processor,
                context_len,
            ) = load_pretrained_model(
                self.model_path,
                None,
                self.model_name,
                device_map="cuda:0",
                overwrite_config=overwrite_config,
            )
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
        vr = decord.VideoReader(video_path, ctx=decord.cpu(0), num_threads=1)
        total_frame_num = len(vr)

        video_fps = vr.get_avg_fps()

        video_start_frame = int(time_to_frame_idx(video_start_time, video_fps))
        start_frame = int(time_to_frame_idx(start_time, video_fps))
        end_frame = int(time_to_frame_idx(end_time, video_fps))

        if end_frame - start_frame > total_frame_num:
            end_frame = total_frame_num + start_frame

        start_frame -= video_start_frame
        end_frame -= video_start_frame
        start_frame = int(round(start_frame))
        end_frame = int(round(end_frame))

        frame_idx = [
            i
            for i in range(start_frame, end_frame)
            if (i - start_frame) % int(video_fps / fps) == 0
        ]
        uniform_sampled_frames = np.linspace(
            start_frame, end_frame - 1, max_frames_num, dtype=int
        )
        frame_idx = uniform_sampled_frames.tolist()

        video = vr.get_batch(frame_idx).asnumpy()

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
        system_message=None,
    ):
        processed_data = self.process_video(
            video_path, video_start_time, start_time, end_time
        )
        image_arrays = processed_data["processed_video"]
        frame_idx = processed_data["frame_idx"]

        video = [Image.fromarray(frame) for frame in image_arrays]

        question = system_message + human_query

        conv = conv_templates[self.conv_template].copy()

        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = preprocess_qwen(
            [{"from": "human", "value": question}, {"from": "gpt", "value": None}],
            self.tokenizer,
            has_image=True,
        ).cuda()

        video_processed = []
        for idx, frame in enumerate(video):
            self.image_processor.do_resize = False
            self.image_processor.do_center_crop = False
            frame = process_anyres_video_genli(frame, self.image_processor, None)

            video_processed.append(frame.unsqueeze(0))

        video_processed = torch.cat(video_processed, dim=0).bfloat16().cuda()
        video_processed = (video_processed, video_processed)

        video_data = (video_processed, (384, 384), "video")

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )

        temperature = 0.2
        top_p = 0.9
        num_beams = 1
        with torch.inference_mode():
            output_ids = self.model.generate(
                inputs=input_ids,
                images=video_data[0][0],
                images_highres=video_data[0][1],
                modalities=video_data[2],
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_new_tokens=128,
                use_cache=True,
            )

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]
        outputs = outputs.strip()
        return outputs
