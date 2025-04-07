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

from egorag.utils.util import time_to_frame_idx


class Azure(BaseQueryModel):
    def __init__(self, api_key, endpoint):
        # import pdb; pdb.set_trace()
        self.API_KEY = api_key
        self.ENDPOINT = endpoint

    def process_video(
        self,
        video_path: str,
        video_start_time: int,
        start_time: int,
        end_time: int,
        fps=1,
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
        start_frame = max(0, int(round(start_frame)))  # 确保不会小于0
        end_frame = min(total_frame_num, int(round(end_frame)))  # 确保不会超过总帧数
        start_frame = int(round(start_frame))
        end_frame = int(round(end_frame))

        # Sample frames based on the provided fps (e.g., 1 frame per second)
        frame_idx = [
            i
            for i in range(start_frame, end_frame)
            if (i - start_frame) % int(video_fps / fps) == 0
        ]

        # Get the video frames for the sampled indices
        video = vr.get_batch(frame_idx).asnumpy()
        # Resize video frames to 384x384
        resized_video = np.zeros((video.shape[0], 384, 384, 3), dtype=np.uint8)
        for i in range(video.shape[0]):
            resized_video[i] = cv2.resize(video[i], (384, 384))
        video = resized_video

        # Return processed video and the corresponding frame indices
        return {
            "processed_video": video,
            "frame_idx": frame_idx,
            "start_time": start_time,
            "end_time": end_time,
        }

    def encode_image_array(self, image_array):
        image = Image.fromarray(image_array)
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

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
        headers = {
            "Content-Type": "application/json",
            "api-key": self.API_KEY,
        }
        image_contents = []
        image_contents.append({"type": "text", "text": human_query})
        for image_array in image_arrays:
            encoded_image = self.encode_image_array(image_array)
            image_contents.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}",
                    },
                }
            )

        # Payload for the request
        if system_message is not None:
            message = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": image_contents},
            ]
        else:
            message = [
                {"role": "user", "content": image_contents},
            ]

        payload = {
            "messages": message,
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 800,
        }

        # Send request

        response = requests.post(self.ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code

        # Handle the response as needed (e.g., print or process)
        return response.json()["choices"][0]["message"]["content"]
