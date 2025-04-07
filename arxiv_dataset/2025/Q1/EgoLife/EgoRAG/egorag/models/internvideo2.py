from typing import Any, Dict, List

import decord
import numpy as np
import torch
from decord import VideoReader, cpu
from egorag.models.base import BaseQueryModel
from egorag.utils.util import time_to_frame_idx
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer

decord.bridge.set_bridge("torch")

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

transform = transforms.Compose(
    [
        transforms.Lambda(lambda x: x.float().div(255.0)),
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.Normalize(mean, std),
    ]
)


class InternVideo2(BaseQueryModel):
    def __init__(self, pretrained):
        self.model = AutoModel.from_pretrained(
            pretrained, torch_dtype=torch.bfloat16, trust_remote_code=True
        ).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained, trust_remote_code=True, use_fast=False
        )

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
        video_tensor = processed_data["processed_video"]
        frame_idx = processed_data["frame_idx"]
        video_tensor = video_tensor.to(self.model.device)
        chat_history = []

        if system_message is None:
            system_message = ""

        response, chat_history = self.model.chat(
            self.tokenizer,
            system_message,
            human_query,
            media_type="video",
            media_tensor=video_tensor,
            chat_history=chat_history,
            return_history=True,
            generation_config={"do_sample": False},
        )
        return response

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

        # In the internvideo2 model, the input is a tensor
        frames = vr.get_batch(frame_idx)
        frames = frames.permute(0, 3, 1, 2)
        frames = transform(frames)

        return {
            "processed_video": frames,
            "frame_idx": frame_idx,
            "start_time": start_time,
            "end_time": end_time,
        }
