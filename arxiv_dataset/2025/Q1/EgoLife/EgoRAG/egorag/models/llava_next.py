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


class LLaVA_NeXT(BaseQueryModel):
    def __init__(self, pretrained):
        model_name = get_model_name_from_path(pretrained)
        (
            self.tokenizer,
            self.model,
            self.image_processor,
            self.context_len,
        ) = load_pretrained_model(pretrained, None, model_name)

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
        start_frame = max(0, int(round(start_frame)))  # 确保不会小于0
        end_frame = min(total_frame_num, int(round(end_frame)))  # 确保不会超过总帧数
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
        video = self.process_video(video_path, video_start_time, start_time, end_time)
        video = (
            self.image_processor.preprocess(video, return_tensors="pt")["pixel_values"]
            .half()
            .cuda()
        )
        video = [video]

        human_query = DEFAULT_IMAGE_TOKEN + "\n" + "Please describe the video in detail"

        conv = conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0], human_query)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(
                prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
            )
            .unsqueeze(0)
            .cuda()
        )
        attention_masks = input_ids.ne(self.tokenizer.pad_token_id).long().cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )

        with torch.inference_mode():
            output_ids = self.model.generate(
                inputs=input_ids,
                images=video,
                attention_mask=attention_masks,
                modalities="video",
                do_sample=False,
                temperature=0.0,
                max_new_tokens=1024,
                top_p=0.1,
                num_beams=1,
                use_cache=True,
            )
        response = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
            0
        ].strip()
        print("start..." + response + "...end")
        return response
