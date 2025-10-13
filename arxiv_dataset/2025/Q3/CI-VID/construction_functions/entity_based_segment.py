from data_engine.core.base import BaseFilter
from pydantic import Field
import os, av, re, random
from loguru import logger
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor

import copy
import re
import torch
import os
import random
from tqdm import tqdm
import json
from PIL import Image
import math, argparse
import io
import tempfile

class VideoMapperEntityBasedSegment(BaseFilter):
    """
    Entity Based Segment

    Purpose:
    Detect the most prominent and consistently appearing entity in a given video.

    Key functionalities:
    - Extract frames from the video
    - Stitch frames into a grid image
    - Use a large model to identify the main entity
    - Filter out irrelevant segments
    - Verify identity consistency (e.g., same person across clips)
    """

    use_class = True  # If the model is used in the processing, use_class = True, use the ray actor mode to avoid repeated loading of the model

    def __init__(
        self, model_name_or_path: str, middel_result_save_dir: str, min_pixels: int = 256*28*28, max_pixels: int = 1280*28*28, **kwargs):
        """
        Initialization method
        Args:
            model_name_or_path: Path to model weights
            middle_result_save_dir: Directory to store intermediate results
            min_pixels: Minimum input resolution
            max_pixels: Maximum input resolution
        """

        super().__init__(**kwargs)
        self.model = None
        self.model_name_or_path = model_name_or_path
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.middel_result_save_dir = middel_result_save_dir

    def save_image_temp(self, img: Image.Image, dir_path: str):
        """Temporarily store a PIL image as a PNG file and return the file path"""
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        with tempfile.NamedTemporaryFile(delete=False, suffix='.png', dir=dir_path) as temp_file:
            temp_path = temp_file.name
            img.save(temp_path)  
            return temp_path

    def get_merged_img_auto(self, frames, batch_size=None ):
        """
        Stitch a batch of frames into a grid image

        Automatically determines grid size based on number of input frames and resizes if necessary
        """

        l = len(frames)

        images = [Image.open(frame).convert('RGB') for frame in frames]

        img_width, img_height = images[0].size

        if batch_size==None:
            batch_size = l

        merged_images = []
        for i in range(0, len(images), batch_size):
            # # Retrieve current batch of images
            batch_images = images[i:i + batch_size]

            # Calculate grid layout: number of rows and columns
            img_grid_w = max(math.ceil(math.sqrt(len(batch_images))), 1)
            img_grid_h = math.ceil(len(batch_images) / img_grid_w)

            if img_width > img_height:
                img_grid_w, img_grid_h = img_grid_h, img_grid_w

            grid_image = Image.new('RGB', (img_grid_w * img_width, img_grid_h * img_height))

            for index, image in enumerate(batch_images):
                x = (index % img_grid_w) * img_width
                y = (index // img_grid_w) * img_height
                grid_image.paste(image, (x, y))

            target_height = img_grid_h * img_height
            target_width = img_grid_w * img_width

            if target_height > 1200:
                scale_factor = 1200 / target_height
                target_width = int(target_width * scale_factor)
                target_height = int(target_height * scale_factor)

            grid_image = grid_image.resize((target_width, target_height), Image.Resampling.LANCZOS)

            merged_images.append(grid_image)
        return merged_images[0]



    def extract_frames_uniformly(self, video_path, num_frames=3):
        """
        Uniformly sample `num_frames` frames from the video

        Args:
            video_path (str): Path to the video file
            num_frames (int): Number of frames to sample

        Returns:
            frames (List[PIL.Image.Image]), clip_idx (int)
        """

        clip_idx = int(video_path.split("Scene-")[1].split(".")[0])

        container = av.open(video_path)
        stream = container.streams.video[0]

        duration_seconds = float(stream.duration * stream.time_base)
        avg_fps = float(stream.average_rate) if stream.average_rate else 0
        estimated_frames = int(duration_seconds * avg_fps)

        if estimated_frames <= 0:
            print(f"Warning: cannot estimate frames for {video_path}, skipping.")
            return []

        # Take the frame index of the middle position of each segment
        split_size = estimated_frames / (num_frames + 1)
        indices = [int(split_size * (i + 1)) for i in range(num_frames)]

        frames = []
        for i, frame in enumerate(container.decode(stream)):
            if i in indices:
                img = frame.to_image()
                frames.append(img)
            if i > max(indices):
                break

        return (frames, clip_idx)


    def save_vertical_concat(self, imgs_list, sample_save_dir, pad=0):
        """
        Vertically stitch each group of images with `pad` pixels of spacing between them, and save as an image.

        Args:
            imgs_list: List of [List[PIL.Image], clip_idx] 
            save_dir: Directory to save output
            pad: Padding between images (in pixels)

        Returns:
            frame_paths: List[str] - Paths of the saved images
        """
        save_dir = sample_save_dir
        os.makedirs(save_dir, exist_ok=True)
        frame_paths = []

        for imgs, clip_idx in imgs_list:
            widths, heights = zip(*(img.size for img in imgs))
            max_width = max(widths)
            total_height = sum(heights) + pad * (len(imgs) - 1)

            new_img = Image.new('RGB', (max_width, total_height), color=(255, 255, 255))
            y_offset = 0
            for img in imgs:
                new_img.paste(img, (0, y_offset))
                y_offset += img.size[1] + pad

            save_path = os.path.join(save_dir, f'video_frames_{clip_idx:05d}.jpg')
            new_img.save(save_path)
            frame_paths.append(save_path)

        return frame_paths


    def save_frames(self, frames, save_dir, prefix="frame"):
        """
        Save an existing list of PIL.Image objects to the specified directory.

        Args:
            frames (List[Image.Image]): List of images to be saved
            save_dir (str): Directory to save the images
            prefix (str): Filename prefix (default: "frame")

        Returns:
            List[str]: List of saved file paths
        """
        os.makedirs(save_dir, exist_ok=True)
        saved_paths = []

        for i, img in enumerate(frames):
            save_path = os.path.join(save_dir, f"{prefix}_{i:05d}.jpg")
            img.save(save_path)
            saved_paths.append(save_path)

        return saved_paths
    
    def get_merged_img_pad(self, frames, img_grid_h=1, padding=10, padding_color=(255, 255, 255)):
        print('frames,', frames)
        merged_rows = []
        for frame_list in frames:
            # Load all images in this row
            images = [Image.open(p).convert('RGB') for p in frame_list]

            img_width, img_height = images[0].size
            batch_images = images
            img_grid_h = img_grid_h
            img_grid_w = math.ceil(len(batch_images) / img_grid_h)

            grid_image = Image.new(
                'RGB',
                (img_grid_w * img_width + (img_grid_w - 1) * padding,
                img_grid_h * img_height + (img_grid_h - 1) * padding),
                padding_color
            )

            for index, image in enumerate(batch_images):
                x = (index % img_grid_w) * (img_width + padding)
                y = (index // img_grid_w) * (img_height + padding)
                grid_image.paste(image, (x, y))

            # Restricted size
            target_height = grid_image.height
            target_width = grid_image.width

            if target_height > 8000:
                scale_factor = 8000 / target_height
                target_width = int(target_width * scale_factor)
                target_height = int(target_height * scale_factor)

            if target_width > 60000:
                scale_factor = 60000 / target_width
                target_width = int(target_width * scale_factor)
                target_height = int(target_height * scale_factor)

            grid_image = grid_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
            merged_rows.append(grid_image)

        # Concatenate all rows
        total_height = sum(img.height for img in merged_rows) + padding * (len(merged_rows)-1)
        max_width = max(img.width for img in merged_rows)
        final_image = Image.new('RGB', (max_width, total_height), padding_color)

        current_y = 0
        for img in merged_rows:
            final_image.paste(img, (0, current_y))
            current_y += img.height + padding

        return final_image
        

    def cal_qwen(self, messages, model, processor ):
        # Preparation for inference
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
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]


    def cal_qwen_entity(self, img, l, model, processor ):

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img,
                    },
                    {"type": "text", "text": f"""
                                There {l} pictures. Each column is a picture. 
                                Can you identify the most common entities (objects/people/goals) among these images? 
                                Note:
                                1) Only return the most common entity, such as one person or one object.
                                2) The entity must be the same one.
                                3) The one must be the main entity, not the background or edge entity.
                                4) The entity must appear in more than 60% of the images. Return 'none' if there are none."
                                5) Return the entity name directly, with its characteristics. 
                                6）The same person is also an entity, return person‘s characteristics(hair, dress), don't guess person‘s name.
                                """},
                ],
            }
        ]
        return self.cal_qwen(messages, model, processor )


    def cal_qwen_ifexist_entity(self, img_path, object, model, processor ):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img_path,
                    },
                    {"type": "text", "text": f"""
                                Is there "{object}" in the picture?
                                """},
                ],
            }
        ]
        return self.cal_qwen(messages, model, processor )

    def cal_qwen_ifone(self, img_path, object, model, processor ):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img_path,
                    },
                    {"type": "text", "text": f"""
                                Is there more than one "{object}" in the picture?
                                """},
                ],
            }
        ]

        return self.cal_qwen(messages, model, processor )

    def cal_qwen_ispeople(self, object, model, processor ):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"""
                                Is "{object}" refers to people?
                                """},
                ],
            }
        ]

        return self.cal_qwen(messages, model, processor )



    def cal_qwen_same_person(self, img_path, object, l, model, processor ):
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img_path,
                    },
                    {"type": "text", "text": f"""
                                There are {l} pictures here. Are the "{object}" in each picture the same person?
                                """},
                ],
            }
        ]

        return self.cal_qwen(messages, model, processor )

    def find_subsequences_with_conditions(self, lst, max_gap=3, min_length=2):
        """
        Find subsequences in a list that satisfy the following conditions:
        - Consecutive elements differ by no more than `max_gap`
        - Each subsequence is at least `min_length` in length

        Returns:
            List of valid subsequences
        """

        subsequences = []  # A list of substrings that match the conditions

        if len(lst) == 0:
            return []
        
        current_subseq = [lst[0]]  # The initial substring contains the first element

        for i in range(1, len(lst)):
            # Determine whether the difference between the current number and the previous number is within the maximum allowed interval
            current_value = int(lst[i][0].split('/')[-1].split('_')[0])
            prev_value = int(lst[i - 1][0].split('/')[-1].split('_')[0])
            # Determine whether the difference between the current number and the previous number is within the maximum interval
            if current_value - prev_value <= max_gap:
                # If the length of the current substring is less than 10, add the current element                
                if len(current_subseq) < 10:
                    current_subseq.append(lst[i])  # If the conditions are met, add the current substring
                else:
                    # If the current substring is full, add it to the result list and start a new substring
                    subsequences.append(current_subseq)
                    current_subseq = [lst[i]]  # 重新开始新的子串
            else:
                # If the length of the current substring is greater than the minimum length, it is added to the result
                if len(current_subseq) >= min_length:
                    subsequences.append(current_subseq)
                current_subseq = [lst[i]]  # 重新开始新的子串

        # Check the current substring one last time to make sure it is added to the result
        if len(current_subseq) >= min_length:
            subsequences.append(current_subseq)

        return subsequences


    def create_replacer(self, replacements):
        def replacer(match):
            return replacements.pop(0)

        return replacer

    def get_model(self):
        if not self.model:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                                                                    self.model_name_or_path,
                                                                    torch_dtype=torch.float16
                                                                ).to("cuda").eval()
            processor = AutoProcessor.from_pretrained(self.model_name_or_path, min_pixels=self.min_pixels, max_pixels=self.max_pixels)
            self.model = model, processor
        return 

    def process(self, data: dict):
        """
        Main processing flow for a single video segment sequence:

        Steps:
        1. Extract frames
        2. Stitch into grid
        3. Use model to recognize entities
        4. Filter based on identity consistency
        Returns:
            Updated data or None
        """

        self.get_model()

        model, processor = self.model

        video_path = data['path']
        video_path_md5_hash = data['video_path_md5_hash']
        clip_seq = data['clip_seq']
        seq_idx = data["seq_idx"] 

        sample_save_dir = os.path.join(self.middel_result_save_dir, video_path_md5_hash, f'{seq_idx:05d}')

        frame_seqs = [self.extract_frames_uniformly(video_path) for video_path in clip_seq]
        merged_frame_paths = self.save_vertical_concat(frame_seqs, sample_save_dir) 

        frame2video_dict = {}
        for merged_frame_path, video_path in zip(merged_frame_paths, clip_seq):
            frame2video_dict[merged_frame_path] = video_path

        split_frame_paths = [self.save_frames(frame_seq, os.path.join(sample_save_dir, f'video_{clip_idx:05d}'), 'split') for frame_seq, clip_idx  in frame_seqs]
        middle_frame_paths = [self.save_frames([frame_seq[1]], os.path.join(sample_save_dir, f'video_{clip_idx:05d}'), 'middle') for frame_seq, clip_idx  in frame_seqs]

        responses = []

        sample_merged_middle = self.get_merged_img_pad(middle_frame_paths)
        sample_merged_middle.save(os.path.join(sample_save_dir, 'sample_merged_middle.jpg'))

        sample_merged_split = self.get_merged_img_pad(split_frame_paths)
        sample_merged_split.save(os.path.join(sample_save_dir, 'sample_merged_split.jpg'))

        # Get the main entity
        response_search_entity = self.cal_qwen_entity(sample_merged_middle, len(middle_frame_paths), model, processor)

        responses.append(f'response_search_entity:  {response_search_entity}')
        if 'none' in response_search_entity or 'None' in response_search_entity:
            return None

        entity = response_search_entity.split('is', 1)[-1]
        entity_ispeople = False

        # Check if the entity refers to a person
        response_ispeople = self.cal_qwen_ispeople(entity, model, processor )
        responses.append(f'response_ispeople:  {response_ispeople}')
        if  'Yes' in response_ispeople:
            entity_ispeople = True

        good_clips = []
        for i, frame3_path in enumerate(merged_frame_paths):
            # Processing a single slice
            frame3_split_path = split_frame_paths[i]
            
            per_frame_paths = frame3_split_path

            # Check if there is an entity
            ifexist_entiy = False
            # Three frames for processing slices
            for frame_id, per_frame_path in enumerate(per_frame_paths): # frame_id : 0, 1, 2
                response_ifexist_entiy = self.cal_qwen_ifexist_entity(per_frame_path, entity, model, processor)
                responses.append(f'response_ifexist_entiy:  {response_ifexist_entiy}')
                if 'Yes' in response_ifexist_entiy:
                    ifexist_entiy = True
                    good_idx = frame_id
                    good_img = per_frame_path
                    break

            if ifexist_entiy:
                good_clips.append((frame3_path, good_img))

        # If it is a person, check whether it is a person and filter out pictures with multiple people
        if entity_ispeople:
            response_ifones = []
            for good_clip in good_clips:
                response_ifone = self.cal_qwen_ifone(good_clip[1] , entity, model, processor)
                responses.append(f'response_ifone:  {str(response_ifone)}')
                response_ifones.append(response_ifone)

            temp_good_clips = []
            for response_ifone, good_clip in zip(response_ifones, good_clips):
                if 'Yes' in response_ifone:
                    continue
                else:
                    temp_good_clips.append(good_clip)
            good_clips = temp_good_clips
            

        if len(good_clips) < 2:
            return None
        
        if entity_ispeople:
            # Determine if it is the same person
            merged_img = self.get_merged_img_auto([x[1] for x in good_clips])
            response_same_person = self.cal_qwen_same_person(merged_img, entity, len(good_clips), model, processor)
            responses.append(('response_same_person:  ', response_same_person))
            if not ('Yes' in response_same_person):
                return None

        good_clips = [frame2video_dict[x[0]] for x in good_clips]
        data["good_clips"] = good_clips

        good_clip_idx = [int(os.path.basename(c).split('-')[-1].split('.')[0]) for c in good_clips]

        data["good_clips"] = good_clips
        data["entity"] = entity
        data["good_clip_idx"] = good_clip_idx


        return data

