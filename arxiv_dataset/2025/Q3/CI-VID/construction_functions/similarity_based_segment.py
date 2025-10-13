import os
from data_engine.core.base import BaseFilter
import torch
import numpy as np
from loguru import logger
from tqdm import tqdm
import av
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import save_images
from scenedetect.video_splitter import split_video_ffmpeg
import imagebind
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

from pathlib import Path
import os
import hashlib
import copy

class VideoMapperSimilarityBasedSegment(BaseFilter):
    """
    Similarity-based video segmentation
    By calculating the similarity of frames, a sequence of video slices with sufficient visual distinction but consistent scenes is constructed.
    """
    use_class = True # If the model is used in the processing, use_class = True, use the ray actor mode to avoid repeated loading of the model


    def __init__(
        self, model_name_or_path, middel_result_save_dir, threshold_upper, threshold_lower, **kwargs
    ):
        """
        Initialization method
        model_name_or_path: Model weight path #Model weights or model identifier path (e.g., CLIP model name/path)
        middel_result_save_dir: Intermediate result storage directory #Intermediate result storage directory
        threshold_upper: Similarity upper threshold #Similarity upper threshold, segments above this threshold will be skipped
        threshold_lower: Similarity lower threshold #Similarity lower threshold, segments below this threshold indicate the start of a new sequence
        """
        super().__init__(**kwargs)
        self.model = None  # Models and preprocessors will be lazy loaded when first used
        self.model_name_or_path = model_name_or_path
        self.threshold_upper = threshold_upper
        self.threshold_lower = threshold_lower
        self.middel_result_save_dir = middel_result_save_dir

    def cosine_similarity_between_embeddings(self, emb1, emb2):
        """
        Calculate the cosine similarity between two feature vectors 
        """
        emb1 = emb1.cpu().numpy().reshape(1, -1)
        emb2 = emb2.cpu().numpy().reshape(1, -1)
        return cosine_similarity(emb1, emb2)[0][0]

    def calculate_similarity(self, embeddings):
        """
        Calculate the similarity sequence between adjacent feature vectors   
        """
        similarities = []
        # Traverse adjacent eigenvector pairs
        for i in range(len(embeddings) - 1):
            similarity = self.cosine_similarity_between_embeddings(
                embeddings[i], embeddings[i + 1]
            )
            similarities.append(float(similarity))
        return similarities

    def extract_frames_uniformly(self, clip_path, num_frames=3):
        """
        Divide the video into num_frames segments, extract a frame from the middle of each segment, and return a list of PIL.Image objects.
        Does not save the frames to disk; only the image objects are returned.
        """
        container = av.open(clip_path)
        stream = container.streams.video[0]
        total_frames = stream.frames

        if total_frames <= 0:
            print(f"Warning: Cannot get frame count for {clip_path}, skipping.")
            return []

        # The frame index of the middle position of each segment
        split_size = total_frames / (num_frames + 1)
        indices = [int(split_size * (i + 1)) for i in range(num_frames)]

        frames = []
        for i, frame in enumerate(container.decode(stream)):
            if i in indices:
                img = frame.to_image()
                frames.append(img)
            if i > max(indices):
                break

        return frames

    def save_horizontal_concat(self, frames, frame_dir_base, video_path, video_path_md5_hash, pad=10):
        """
        Stitch each set of images horizontally, padding them with pixels, and save them as an image.

        Args:
        frames: List[List[PIL.Image]]
        frame_dir_base: Save directory
        pad: Padding between each image, in pixels

        Returns:
        frame_paths: List[str] List of saved image paths
            
        """

        video_path = Path(video_path)
        frame_dir_base = Path(frame_dir_base)


        video_name = video_path_md5_hash  # File name without suffix
        save_dir = frame_dir_base / video_name 


        os.makedirs(save_dir, exist_ok=True)
        frame_paths = []

        num_frames = len(frames)
        digits = max(4, len(str(num_frames)))  # At least 4 digits, if not enough, expand

        for i, imgs in enumerate(frames):
            widths, heights = zip(*(img.size for img in imgs))
            total_width = sum(widths) + pad * (len(imgs) - 1)
            max_height = max(heights)

            new_img = Image.new('RGB', (total_width, max_height), color=(255, 255, 255))
            x_offset = 0
            for img in imgs:
                new_img.paste(img, (x_offset, 0))
                x_offset += img.size[0] + pad

            filename = f"video_frames_{i+1:0{digits}d}.jpg"  # Fill the calculated digits with zeros
            save_path = os.path.join(save_dir, filename)
            new_img.save(save_path)
            frame_paths.append(save_path)

        return frame_paths
        
    def find_subsequences_with_conditions(self, lst, max_gap=5, min_length=2):
        """
        Split the sequence by the interval between adjacent file numbers and return all subsequences that meet the criteria.
        Parameters:
        lst: A list of file paths (e.g., in the format '.../123_xxx'). The subsequences should be split based on the numerical continuity of the file names after sorting.
        max_gap: The maximum allowed interval between adjacent file numbers (default 5). Any gap exceeding this value is considered a segment.
        min_length: The minimum length of a subsequence (default 2). Subsequences shorter than this length will be discarded.
        Returns:
        subsequences: A list of subsequences filtered by the criteria.
        """
        subsequences = []  # Used to store subsequences that meet the conditions

        if len(lst) == 0:
            return []  # An empty list returns an empty result directly

        # Initialize the current subsequence, starting with the first element
        current_subseq = [lst[0]]

        # Traverse the list starting from the second element
        for i in range(1, len(lst)):
            # Extract the file number part from the path. Assuming the file number in the file path is in the file name, press split('_')[0] to get
            
            current_value = int(lst[i].split('-')[-1].replace('.mp4', ''))
            prev_value = int(lst[i - 1].split('-')[-1].replace('.mp4', ''))

            # Determine whether the difference between the current number and the previous number is within the max_gap range
            if current_value - prev_value <= max_gap:
                # If the difference is less than or equal to max_gap, continue to add to the current subsequence
                if len(current_subseq) < 10:
                    current_subseq.append(lst[i])
                else:
                    # The current subsequence length has reached the upper limit (10), first add it to the result, and then restart the new sequence from the current element
                    subsequences.append(current_subseq)
                    current_subseq = [lst[i]]
            else:
                # The difference between the current number and the previous number is greater than max_gap, ending the current subsequence.
                # Only when the subsequence length is greater than or equal to min_length, it is added to the result
                if len(current_subseq) >= min_length:
                    subsequences.append(current_subseq)
                # Start a new subsequence
                current_subseq = [lst[i]]

        # After the traversal is completed, the last subsequence needs to be checked
        if len(current_subseq) >= min_length:
            subsequences.append(current_subseq)

        return subsequences

    def split_video_by_content(self, video_path, clip_dir_base, video_path_md5_hash, threshold=3.0):
        """
        Use PySceneDetect to segment the video content and save it as multiple clip files.

        Parameters:
        video_path: Original video path
        clip_dir: Directory where the clips are saved
        threshold: Content change detection threshold (default 30.0)

        Returns:
        clip_path_list: List of paths to all segmented video clips
        """

        video_path = Path(video_path)
        clip_dir_base = Path(clip_dir_base)


        video_name = video_path_md5_hash     # File name without suffix
        clip_dir = clip_dir_base / video_name 


        clip_dir.mkdir(parents=True, exist_ok=True)

        video_manager = VideoManager([str(video_path)])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold))

        video_manager.set_downscale_factor()
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)

        scene_list = scene_manager.get_scene_list()
        print(f"[INFO] {len(scene_list)} scenes detected.")

        # save clip
        split_video_ffmpeg([str(video_path)], scene_list, output_dir=str(clip_dir), show_progress=True)

        # Get the clip path of all outputs
        clip_path_list = sorted(str(p) for p in clip_dir.glob("*.mp4"))

        return clip_path_list

    def process(self, data: dict) -> list:
        """
        Input: Dictionary of video paths
        Return: List of dictionaries, each containing a set of clip_seq

        clip_seq is a sequence of clip paths clustered by similarity.
        """

        if not self.model:
            self.model = imagebind_model.imagebind_huge(pretrained=True).eval().to("cuda")

        # Extract relevant path information from data
        video_path = data['path']
        clip_dir_base = self.middel_result_save_dir
        frame_dir_base = self.middel_result_save_dir

        # Generate a unique hash using the video path for naming
        video_path_md5_hash = hashlib.md5(video_path.encode('utf-8')).hexdigest()

        # Divide the video into clip paths according to the video content
        clip_paths = self.split_video_by_content(
            video_path, clip_dir_base, video_path_md5_hash
        )

        # Create a frame storage directory
        os.makedirs(frame_dir_base, exist_ok=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Extract several frames from each clip, stitch them together into a single image, and save it.
        frames = [
            self.extract_frames_uniformly(clip_path, num_frames=3)
            for clip_path in clip_paths
        ]
        frame_paths = self.save_horizontal_concat(
            frames, frame_dir_base, video_path, video_path_md5_hash
        )

        inputs = {
            ModalityType.VISION: imagebind.data.load_and_transform_vision_data(frame_paths, device),
        }
        with torch.no_grad():
            embeddings = self.model(inputs)[ModalityType.VISION]
            
        similarities = self.calculate_similarity(embeddings)

        # Add similarity 0 before the first clip to facilitate alignment
        similarities.insert(0, 0)
        print(similarities)

        temp = []
        samples = []
        # Traverse the similarity and corresponding clip
        for similarity, clip_path in zip(similarities, clip_paths):
            if similarity > self.threshold_upper:
                # The similarity is too high, skip the clip directly
                continue
            elif similarity > self.threshold_lower:
                # Medium similarity, added to the current temp sequence
                temp.append(clip_path)
                # If temp length reaches 10, truncate into a group
                if len(temp) >= 10:
                    samples.append(temp)
                    temp = []
            else:
                # If the similarity is low, end the current sequence and start a new one
                if temp:
                    samples.append(temp)
                temp = [clip_path]

        # At the end of the traversal, if there is still clip in temp, add it as well
        if temp:
            samples.append(temp)

        # Add video_path_md5_hash to data
        data['video_path_md5_hash'] = video_path_md5_hash

        # Clips in the same sequence cannot be too far apart. The cutting results are based on the distance.
        clip_seqs = []
        for sample in samples:
            sample = self.find_subsequences_with_conditions(sample)
            clip_seqs.extend(sample)

        # Final result: construct a new dictionary for each clip sequence
        result = []
        for seq_idx, clip_seq in enumerate(clip_seqs):
            new_data = copy.deepcopy(data)
            new_data["clip_seq"] = clip_seq
            new_data["seq_idx"] = seq_idx
            result.append(new_data)

        return result


