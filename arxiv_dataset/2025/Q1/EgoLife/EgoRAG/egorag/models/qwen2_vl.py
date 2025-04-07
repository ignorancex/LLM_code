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
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import transformers
from egorag.utils.util import time_to_frame_idx
from egorag.utils.vision_process import *
from PIL import Image
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    Qwen2VLForConditionalGeneration,
)


class Qwen2_VL(BaseQueryModel):
    def __init__(
        self,
        pretrained: str = "Qwen/Qwen2-VL-7B-Instruct",
    ) -> None:
        super().__init__()
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            pretrained, torch_dtype="auto", device_map="cuda:0"
        )
        self.processor = AutoProcessor.from_pretrained(pretrained)

        self.model.to("cuda").eval()

    def process_video(
        self,
        video_path: str,
        video_start_time: int,
        start_time: int,
        end_time: int,
        fps=1,
        max_frames_num=10,
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
        print(start_frame, end_frame)
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
            video_path, video_start_time, start_time, end_time, fps=1, max_frames_num=30
        )
        image_arrays = processed_data["processed_video"]
        frame_idx = processed_data["frame_idx"]

        video = [Image.fromarray(frame) for frame in image_arrays]
        question = system_message + human_query
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video,
                        "max_pixels": 360 * 420,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]
        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return output_text[0]
