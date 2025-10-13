
from data_engine.core.base import BaseFilter
import json
import time, torch
from PIL import Image
import os
import re
import openai

class VideoMapperClipSeqSingleAnnotation(BaseFilter):
    """
    Single-clip description generation based on video segments  
    Purpose: Generate detailed descriptions for selected clip sequences to help visually impaired users understand video segments.

    Main procedure:
    1. Extract frames from each clip and stitch them into a single image
    2. Use a large vision-language model to generate descriptions based on the stitched image  
    (Stitched input supports mainstream VLMs such as Qwen, GPT, etc. For multi-frame input, please customize message composition)
    3. Structure the description into JSON format, including four parts:  
    video_content, camera_angle, camera_movement, and video_background
    """

    use_class = True  # If the model is used in the processing, use_class = True, use the ray actor mode to avoid repeated loading of the model

    def __init__(
        self, model_name_or_path: str, middel_result_save_dir: str, min_pixels: int = 256*28*28, max_pixels: int = 1280*28*28, **kwargs):
        """
        Initialization method  
        Args:  
            model_name_or_path: Path to the model weights  
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

    def cal_response_single(self, sin_paths):
        """
        Generate descriptions for a batch of clip sequences  
        Args:
            sin_paths (List[Tuple[frame_list, num_seconds]]): List of frame lists and their durations  
        Returns:
            List[str]: Model-generated descriptions
        """

        response_texts = []
        for i, sin_path in enumerate(sin_paths[:]):
            frames, num_seconds = sin_path
            content = []
            prompt_single = {"type": "text",
                        "text": f"""
                                You are the most powerful video understanding model which is responsible for generation video description to help the blind people to understand the video. Since they cannot see, you should describe the video as detailed as possible.
                                You will see some consecutive frames extracted evenly from the video. The total number of frames in the video is 3,  and the total duration of the video is {num_seconds:.2f} seconds.
                                **Description Hints**:
                                - If the video is focused on a specific subject, please provide detailed descriptions of the subject's textures, attributes, locations, presence, status, characteristics, countings, etc. If there are multiple subjects, please accurately describe their relationships with each other.
                                - Summarize the possible types of current video. You can refer to: landscape videos, aerial videos, action videos, documentaries, educational videos, product promotional videos/advertisements, slow-motion videos, time-lapse videos, music videos/MVs, interview videos, animations, movie clips, How-To videos, and so on.
                                - If there are commen sense or world knowledge, for example, species, celebrities, scenic spots and historical sites, you must state them explicitly instead of using phrases like "a person", "a place", etc.
                                - If there is any textual information in the video, describe it in its original language without translating it.
                                - If there are any camera movements, please describe them in detail. You may refer to professional photography terms like "Pan" "Tilt" "follow focus" "multiple angles", but remember only state them when you're absolutely sure. DO NOT make up anything you don't know.
                                - Include temporal information in the description of the video.
                                - Scene transitions: For example, transitioning from indoors to outdoors, or from urban to rural areas. This can be indicated by specifying specific time points or using transition sentences.
                                - Progression of events: Use time-order words such as "first," "then," "next," "finally" to construct the logical sequence of events and the flow of time.
                                - Use verbs and adverbs to describe the speed, intensity, etc., of actions, such as "walking slowly," "suddenly jumping."
                                - Facial expressions and emotional changes: Capture facial expressions of characters, such as "frowning," "smiling."
                                - Whether the video is in slow motion or fast motion: Determine and indicate whether the video is in slow motion or accelerated.
                                - Any other temporal information you can think of.
                                **Restriction Policies**:
                                - The description should be purely factual, with no subjective speculation.
                                - DO NOT add any unnecessary speculation about the things that are not part of the video such as "it is inspiring to viewers" or "seeing this makes you feel joy".
                                - DO NOT add the evidence or thought chain. If there are some statement are inferred, just state the conclusion.
                                - DO NOT add things such as "creates a unique and entertaining visual" "creating a warm atmosphere" as these descriptions are interpretations and not a part of the video itself.
                                - DO NOT analyze the text content in the video, and only tell the content themselves.
                                - DO NOT include words like "image" "frame" "sequence" "video" "visuals" "content" in your response.  Describe only and directly content and events.
                                - Do NOT use words like 'series of shots', 'sequence', 'scene', 'video', 'frame', 'image', 'visuals', 'content' as the subject of description; directly describe the content of the video.
                                - DO NOT describe frame by frame, or use "first frame" "second frame". Describe the video as a whole directly.
                                - DO NOT analyze, subjective interpretations, aesthetic rhetoric, such as context, atmosphere, suggest, drama, etc. focus solely on objective descriptions.
                                - DO NOT including any reasoning description like "probably because" or "appears to be".
                                **Description Requirment**:
                                Please describe the video by:
                                1) the video content: comprehensive description of the video content, encompassing key actions, the unfolding plot, characters, objects, and visual details. This includes describing the movement and behavior of the characters, the progression of the narrative, and how objects or settings are used within the story. Additionally, highlight any relevant visual or thematic elements that contribute to the overall tone and message of the video. This analysis should consider how all these components work together to enhance the viewer's understanding and experience of the scene.（about 300 words）                            
                                2) the camera angle: the perspective from which the camera captures the main object or scene. （about 30 words）
                                3) the camera movement: panning, tilting, and tracking, as well as changes in zoom (in or out) and focus.（about 30 words）                           
                                4) the background: the environmental factors, lighting, space, surrounding elements that provide context to the scene, and the setting in which the action unfolds.（about 60 words）
                                Do not analyze, subjective interpretations, aesthetic rhetoric, such as "context", "atmosphere", "suggest", "drama", etc. focus solely on objective descriptions.
                                DO not including any reasoning description like "suggest", "indicate", "probably because", "appears to be".
                                Note that the consecutive frames reflect the passage of time, so the video can be described along the timeline.
                                Directly return in the json format like this:
                                {{"video_content": "...", "camera_angle": "...", "camera_movement": "...",  "video_background": "...", }}.  
                                #[Video Frames]: 
                                """
                    }

            content.append(prompt_single)
            content = self.add_img(frames, content, )
            
            messages = [
                    {
                        "role": "user",
                        "content": content
                    }
                ]

            retries = 0
            max_retries =5
            retry_delay=10

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
        """Add image paths as image-type messages in the input"""

        for i, img in enumerate(imgs):
            content.append({
                            "type": "image",
                            "image": img,
                        })
        return content

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
        """
        Get video durations  
        Args:
            video_paths: List of video file paths  
        Returns:
            durations: Duration (in seconds) of each corresponding video (float)
        """

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
        1. Load the model  
        2. Extract and stitch all frames from each clip into an image  
        3. Call the large model to generate descriptions from the stitched image  
        4. Update the description into the data
        """

        openai.api_key = ""
        openai.api_base = ""

        video_path_md5_hash = data['video_path_md5_hash']
        good_clips = data["good_clips"] 
        seq_idx = data["seq_idx"] 
        good_clip_idx = data["good_clip_idx"] 

        sample_save_dir = os.path.join(self.middel_result_save_dir, video_path_md5_hash, f'{seq_idx:05d}')

        duration_list = self.get_durations(good_clips)
        frame_list = [[os.path.join(sample_save_dir, f'video_frames_{clip_idx:05d}.jpg')] for clip_idx in good_clip_idx]

        sin_paths = [(frame, duration) for frame, duration in zip(frame_list, duration_list)]
        captions = self.cal_response_single(sin_paths)

        data["response_singles"] = captions
        return data