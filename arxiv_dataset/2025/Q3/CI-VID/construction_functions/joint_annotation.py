
from data_engine.core.base import BaseFilter
import json
import time, torch
from PIL import Image
import os
import openai
import re, math

class VideoMapperClipSeqJointAnnotation(BaseFilter):
    """
    Joint description generation based on video segments  
    Purpose: Describe continuity and changes between two consecutive clips, including:  
    - Content continuity and variation  
    - Environmental continuity and variation  
    - Changes in camera angle and movement  
    The description is generated using a large model.
    """

    use_class = True  # If the model is used in the processing, use_class = True, use the ray actor mode to avoid repeated loading of the model
    def __init__(
        self, model_name_or_path: str, middel_result_save_dir: str, min_pixels: int = 256*28*28, max_pixels: int = 1280*28*28, **kwargs):
        """
        Initialization method  
        Args:  
            model_name_or_path: Path to model weights  
            middle_result_save_dir: Directory for saving intermediate results  
            min_pixels: Minimum pixel count for model input  
            max_pixels: Maximum pixel count for model input
        """

        super().__init__(**kwargs)
        self.model = None
        self.model_name_or_path = model_name_or_path
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.middel_result_save_dir = middel_result_save_dir

    def cal_response_diff(self, response_singles, diff_paths):
        """
        Generate comparative descriptions for clip pairs  
        Args:
            response_singles (List[str]): Descriptions of individual clips  
            diff_paths (List[str]): Paths to stitched comparison images of clip pairs  
        Returns:
            response_texts (List[str]): Model-generated comparative descriptions
        """

        response_texts = []
        for i, diff_path in enumerate(diff_paths[:]):
            content = []
            h_response_single = response_singles[i]
            t_response_single = response_singles[i+1]

            prompt_single = {"type": "text",
                        "text": f"""    
                                Here are two consecutive video clips along with their descriptions:
                                # the first clip description: {h_response_single}
                                # the second clip description: {t_response_single}
                                The second clip continues the theme of the first clip. Based on the provided frames, please describe the continuation and differences between the two clips in terms of:
                                1) the continuation part and change in video content (characters, objects,key, actions, the unfolding plot, and visual details, about 300 words)       
                                2) the continuation part and change in background (the environmental factors, lighting, space, surrounding elements that provide context to the scene, about 60 words）
                                3) the change in camera angle (the perspective from which the camera captures the main object or scene, about 30 words)
                                4) the change in camera movement (panning, tilting, and tracking, as well as changes in zoom (in or out) and focus, about 30 words）                           
                                Do not analyze, subjective interpretations, aesthetic rhetoric, such as "context", "atmosphere", "suggest", "drama", etc. focus solely on objective descriptions.
                                DO not including any reasoning description like "suggest", "indicate", "probably because", "appears to be".
                                **Directly return in the json format like this:
                                {{"continuation_in_video_content": "...", "change_in_video_content": "...", "continuation_in_video_background": "...", "change_in_video_background": "...", "change_in_camera_angle": "...", "change_in_camera_movement": "...",   }}.  
                                #[clip frames of the first clip (above row) and the second clip (bottom row)]:
                        """
                        }

            content.append(prompt_single)
            content = self.add_img([diff_path], content)
            
            messages = [
                    {
                        "role": "user",
                        "content": content
                    }
                ]

            retries = 0
            max_retries = 10
            retry_delay = 10
    
            while retries < max_retries:
                try:
                    response_text = self.cal_vlm(messages)
                    response_texts.append(response_text)
                    break  
                except Exception as e:  
                    print(f"Failed to get response. retry {retries + 1}/{max_retries} times，error: {e}")
                    retries += 1
                    if retries >= max_retries:
                        raise Exception("Exceeded maximum retry attempts. Failed to get response.")
                    time.sleep(retry_delay)  

            if retries >= max_retries:
                raise Exception("Exceeded maximum retry attempts. Failed to get response.")
                
        return response_texts

    def add_img(self, imgs, content, text = ''):

        for i, img in enumerate(imgs):
            content.append({
                            "type": "image",
                            "image": img,
                        })
        return content


    def get_merged_img_pad(self, frames, img_grid_h=1, padding=10, padding_color=(255, 255, 255)):
        """
        Stitch a pair of clips into a comparison image  
        Args:
            frames: 2D list, e.g., [[clip1_frame1, ...], [clip2_frame1, ...]]  
        Returns:
            PIL.Image: Stitched comparison image
        """

        merged_rows = []
        for frame_list in frames:
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

    def cal_vlm(self, messages, max_retries=5, sleep_time=2):
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=3000
        )

        # Handle different API versions
        if isinstance(response, str):
            response = json.loads(response)
            response_txt = response['choices'][0]['message']['content']
        else:
            response_txt = response.choices[0].message.content

        return response_txt

    def get_durations(self, video_paths):
        durations = []
        for path in video_paths:
            try:
                container = av.open(path)
                stream = container.streams.video[0]
                duration_sec = float(stream.duration * stream.time_base)
            except Exception as e:
                print(f"Error reading {path}: {e}")
                duration_sec = 0.0
            durations.append(duration_sec)
        return durations

    def split_string_by_list(self, input_str, split_list):
        pattern = '|'.join(map(re.escape, split_list))
        return re.split(pattern, input_str)
        
    def response2json(self, response_text, keys):
        extract_json = self.split_string_by_list(response_text, [f"\"{key}\":" for key in keys])[1:]
        extract_json = [x.strip(" '\"\\\n{},") for x in extract_json]

        response_json = {key: extract_json[i] for i,key in enumerate(keys)}
        response_json = json.dumps(response_json)
        response_text = json.loads(response_json)
        return response_json


    def process(self, data: dict):
        """
        Main processing logic:
        1. Retrieve clip information and individual descriptions  
        2. Stitch side-by-side comparison image for consecutive clips  
        3. Call the model to generate descriptions of differences between clips  
        4. Return the updated data
        """

        openai.api_key = "" 
        openai.api_base = ""

        video_path_md5_hash = data['video_path_md5_hash']
        seq_idx = data["seq_idx"] 
        good_clip_idx = data["good_clip_idx"] 
        response_singles = data["response_singles"] 

        sample_save_dir = os.path.join(self.middel_result_save_dir, video_path_md5_hash, f'{seq_idx:05d}')
        diff_paths = []
        frames = [[os.path.join(sample_save_dir, f'video_{clip_idx:05d}', f'split_{i:05d}.jpg') for i in range(3)] for clip_idx in good_clip_idx]
        for i in range(len(frames) - 1):
            pre_frame_list = frames[i]
            tail_frame_list = frames[i + 1]
            merged_frame_pair = self.get_merged_img_pad([pre_frame_list, tail_frame_list])
            diff_path = os.path.join(sample_save_dir, f'video_{good_clip_idx[i]:05d}', 'merged_frame_pair.jpg')
            merged_frame_pair.save(diff_path)
            diff_paths.append(diff_path)

        captions = self.cal_response_diff(response_singles, diff_paths)

        data["response_joint"] = captions
        return data