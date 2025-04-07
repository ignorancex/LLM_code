import os
import shutil

import cv2
import decord
import numpy as np
import torch
from decord import VideoReader, cpu
from egorag.models.base import BaseQueryModel
from egorag.utils.util import time_to_frame_idx
from transformers import AutoModel, AutoTokenizer

torch.set_grad_enabled(False)


class IXC2_5(BaseQueryModel):
    def __init__(self, pretrained):
        torch.set_grad_enabled(False)
        self.model = (
            AutoModel.from_pretrained(
                pretrained, torch_dtype=torch.bfloat16, trust_remote_code=True
            )
            .cuda()
            .eval()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained, trust_remote_code=True
        )
        self.model.tokenizer = self.tokenizer

    def inference_video(
        self,
        video_path,
        video_start_time,
        start_time,
        end_time,
        human_query,
        system_message=None,
    ):
        torch.set_grad_enabled(False)
        image = [
            video_path,
        ]
        if system_message is None:
            system_message = ""
        query = system_message + human_query
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            response, _ = self.model.chat(
                self.tokenizer,
                query,
                image,
                do_sample=False,
                num_beams=3,
                use_meta=True,
            )
        print(response)
        return response
