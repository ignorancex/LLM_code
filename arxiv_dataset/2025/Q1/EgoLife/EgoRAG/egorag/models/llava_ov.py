import copy
import sys
import warnings
from typing import Any, Dict, List

import decord
import numpy as np
import requests
import torch
from decord import VideoReader, cpu
from egorag.models.base import BaseQueryModel
from egorag.utils.util import time_to_frame_idx
from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_anyres_image,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from PIL import Image
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer


class LLaVA_Onevision(BaseQueryModel):
    def __init__(self, pretrained):
        model_name = "llava_qwen"
        device = "cuda"
        device_map = "auto"
        llava_model_args = {
            "multimodal": True,
        }
        (
            self.tokenizer,
            self.model,
            self.image_processor,
            self.max_length,
        ) = load_pretrained_model(
            pretrained,
            None,
            model_name,
            device_map=device_map,
            attn_implementation="sdpa",
            **llava_model_args,
        )
        self.model.eval()

    def process_video(self, video_path, video_start_time, start_time, end_time, fps=1):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        video_fps = vr.get_avg_fps()

        video_start_frame = int(time_to_frame_idx(video_start_time, video_fps))
        start_frame = int(time_to_frame_idx(start_time, video_fps))
        end_frame = int(time_to_frame_idx(end_time, video_fps))

        if end_frame - start_frame > total_frame_num:
            end_frame = total_frame_num + start_frame

        start_frame -= video_start_frame
        end_frame -= video_start_frame
        start_frame = max(0, int(round(start_frame)))  # Ensure it is not less than 0
        end_frame = min(
            total_frame_num, int(round(end_frame))
        )  # Ensure it does not exceed total frames
        start_frame = int(round(start_frame))
        end_frame = int(round(end_frame))

        frame_idx = [
            i
            for i in range(start_frame, end_frame)
            if (i - start_frame) % int(video_fps / fps) == 0
        ]

        frames = vr.get_batch(frame_idx).asnumpy()
        return frames

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
        # video = process_images(processed_data, self.image_processor, self.model.config)
        image_tensors = []
        frames = (
            self.image_processor.preprocess(processed_data, return_tensors="pt")[
                "pixel_values"
            ]
            .half()
            .cuda()
        )
        image_tensors.append(frames)

        human_query = f"{DEFAULT_IMAGE_TOKEN}\n{human_query}"

        conv_template = (
            "qwen_1_5"  # Make sure you use correct chat template for different models
        )

        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], human_query)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .to(self.model.device)
        )
        image_sizes = [frame.size for frame in processed_data]

        cont = self.model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=4096,
            modalities=["video"],
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
        print(text_outputs[0])
        return text_outputs[0]
