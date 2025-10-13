# utils.py
import random
import os
import json
import argparse
from torch.utils.data import Dataset, DataLoader
import numpy as np
from decord import VideoReader, cpu
from datetime import datetime
import traceback
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

def gen_frames_numpy(video_path, max_frames_num, start_sec=None, end_sec=None):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    # print(f"Total frames in video: {total_frame_num}")
    if start_sec is not None and end_sec is not None:
        # str to int, since start_sec and end_sec are str in seconds, like '0.0', '10.2'
        start_frame = int(float(start_sec) * vr.get_avg_fps())
        end_frame = int(float(end_sec) * vr.get_avg_fps())
        # uniformly sample frames from start_frame to end_frame
        uniform_sampled_frames = np.linspace(start_frame, end_frame, max_frames_num, dtype=int)
        frames = vr.get_batch(frame_idx).asnumpy()
    else:
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frames = vr.get_batch(frame_idx).asnumpy()
    return frames

class QADataset(Dataset):
    def __init__(self, args):
        QA_file_path = args.QA_file_path
        video_folder = args.video_folder
        self.frames_from = args.frames_from
        self.provide_boundaries = args.provide_boundaries
        self.model_name = os.path.basename(args.model_path)
        self.extract_frames_method = args.extract_frames_method
        if args.extract_frames_method == 'fps':
            self.fps = args.fps
            self.max_frames_num = args.max_frames_num
        elif args.extract_frames_method == 'max_frames_num': 
            self.max_frames_num = args.max_frames_num       
        self._conditional_imports(self.model_name)
        with open(QA_file_path, 'r') as file:
            data = json.load(file)
        
        # choices = "A"
        message = [
            "Above are the event captions given by the user, whose timestamps are very accurate but the subjects of the sentences are not necessarily what we want to highlight.",
            "Please pay attention to the object highlighted (HO) by colored bounding box and blue mask in the video frames, and generate accurate object-centric caption for the HO.",
            "Please make sure in object-centric paragraph caption, the sentences should be detailed and specific, and the subjects of all sentences **MUST be \"HO\"**.",
            "Please follow the format:\n",
            "HO: ...\n",
            "HO's itself attributes: ...\n",
            "All actions done by HO: ...\n",
            "All statuses of HO: ...\n",
            "All other objects interacted with HO: ...\n",
            "All environments/backgrounds of HO: ...\n",
            "All events related to HO: ...\n",
            "Final object-centric paragraph caption: The HO is [attributes], [environment]. From ... to ...s, the HO [status], [any action], [any status/attribute/environment changes]. From ... to ...s, the HO [status], [any action], [any status/attribute/environment changes]. From ... to ...s, the HO [status], [any action], [any status/attribute/environment changes]. The OH's [final status].",
        ]
        message = ' '.join(message)

        self.QAs = []
        for item in data:
            video = item['video']
            segment = item['segment'].split('_')
            # task_class = item['class']
            question = item['question']
            # options = '\n'.join([f"({key}){value}" for key, value in item["options"].items()])
            correct_answer = item['answer']
            # if not self.provide_boundaries:
                # final_question = message + f'Please describe the {question} in the video from {segment[0]} to {segment[1]}.\n'
            # else:
            final_question = message + question + '\n'
            video_id = video[:-4]
            suffix_groups = ['.avi', '.mp4', '.mkv', '.webm']
            video_path = None
            for vid_folder in [video_folder]:
                for suffix in suffix_groups:
                    video = video_id + suffix
                    if os.path.isfile(os.path.join(vid_folder, video)):
                        video_path = os.path.join(vid_folder, video)
                        break
            if video_path is None:
                print(f"Video {video_id} not found in video folders.")
                continue

            self.QAs.append((final_question, video_path, correct_answer, video, segment, question))
    
    def __len__(self):
        return len(self.QAs)
    
    def __getitem__(self, idx):
        if self.frames_from == 'video':
            QAs, frames = self._frames_from_video(idx)
            return QAs, frames
        elif self.frames_from == 'frames':
            if 'LongVA-7B-DPO' in self.model_name:
                frames = self._load_frames_from_png(self.QAs[idx][1])
                return self.QAs[idx], frames
            elif "llava-onevision" in self.model_name:
                frames = self._load_frames_from_png(self.QAs[idx][1])
                return self.QAs[idx], frames
            elif 'MiniCPM-V-2_6' in self.model_name:
                frames = self._encode_video_minicmp(self.QAs[idx][1])
                return self.QAs[idx], frames
            else:
                return self.QAs[idx]
            
    def _frames_from_video(self, idx):
        if self.extract_frames_method == 'max_frames_num':
            if 'LongVA-7B-DPO' in self.model_name:
                frames = self._gen_frames_numpy(self.QAs[idx][1], max_frames_num=self.max_frames_num)
                return self.QAs[idx], frames
            elif "llava-onevision" in self.model_name:
                frames = self._gen_frames_numpy(self.QAs[idx][1], max_frames_num=self.max_frames_num, start_sec=self.QAs[idx][4][0], end_sec=self.QAs[idx][4][1])
                return self.QAs[idx], frames
            elif "long-llava" in self.model_name:
                frames = self._gen_frames_numpy(self.QAs[idx][1], max_frames_num=self.max_frames_num)
                return self.QAs[idx], frames 
            elif "Video-LLaVA" in self.model_name:
                frames = self._gen_frames_numpy(self.QAs[idx][1], max_frames_num=self.max_frames_num)
                return self.QAs[idx], frames            
            elif 'MiniCPM-V-2_6' in self.model_name:
                frames = self._encode_video_minicmp(self.QAs[idx][1], max_frames_num=self.max_frames_num)
                return self.QAs[idx], frames
            elif 'Oryx' in self.model_name:
                frames = self._gen_frames_numpy(self.QAs[idx][1], max_frames_num=self.max_frames_num)
                return self.QAs[idx], frames
            else:
                empty_frames = np.empty((0,))
                return self.QAs[idx], empty_frames
        elif self.extract_frames_method == 'fps':
            if 'LongVA-7B-DPO' in self.model_name:
                frames = self._gen_frames_numpy_fps(self.QAs[idx][1], fps=self.fps, max_frames_num=self.max_frames_num)
                return self.QAs[idx], frames
            elif "llava-onevision" in self.model_name:
                frames = self._gen_frames_numpy_fps(self.QAs[idx][1], fps=self.fps, max_frames_num=self.max_frames_num, start_sec=self.QAs[idx][4][0], end_sec=self.QAs[idx][4][1])
                return self.QAs[idx], frames
            elif "long-llava" in self.model_name:
                frames = self._gen_frames_numpy_fps(self.QAs[idx][1], fps=self.fps, max_frames_num=self.max_frames_num)
                return self.QAs[idx], frames
            elif "Video-LLaVA" in self.model_name:
                frames = self._gen_frames_numpy_fps(self.QAs[idx][1], fps=self.fps, max_frames_num=self.max_frames_num)
                return self.QAs[idx], frames
            elif 'MiniCPM-V-2_6' in self.model_name:
                frames = self._encode_video_minicmp_fps(self.QAs[idx][1], fps=self.fps, max_frames_num=self.max_frames_num)
                return self.QAs[idx], frames
            elif "Oryx" in self.model_name:
                frames = self._gen_frames_numpy_fps(self.QAs[idx][1], fps=self.fps, max_frames_num=self.max_frames_num)
                return self.QAs[idx], frames
            else:
                empty_frames = np.empty((0,))
                return self.QAs[idx], empty_frames 
               
    def _gen_frames_numpy(self, video_path, max_frames_num=32, start_sec=None, end_sec=None):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        # print(f"Total frames in video: {total_frame_num}")
        if self.provide_boundaries:
            # str to int, since start_sec and end_sec are str in seconds, like '0.0', '10.2'
            try:
                start_frame = max(int(float(start_sec) * vr.get_avg_fps()), 0)
                end_frame = min(int(float(end_sec) * vr.get_avg_fps()), total_frame_num - 1)
                if end_frame < start_frame:
                    start_frame, end_frame = 0, total_frame_num - 1
                # uniformly sample frames from start_frame to end_frame
                max_frames_num = max(min(max_frames_num, end_frame - start_frame + 1), 1)
                uniform_sampled_frames = np.linspace(start_frame, end_frame, max_frames_num, dtype=int)
                frame_idx = uniform_sampled_frames.tolist()
                frames = vr.get_batch(frame_idx).asnumpy()
            except ValueError:
                uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
                frame_idx = uniform_sampled_frames.tolist()
                frames = vr.get_batch(frame_idx).asnumpy()
        else:
            # max_frames_num = min(max_frames_num, end_frame - start_frame + 1)
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            frames = vr.get_batch(frame_idx).asnumpy()
        return frames
    
    def _gen_frames_numpy_fps(self, video_path, fps=1, max_frames_num=32):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        original_fps = vr.get_avg_fps()
        total_frame_num = len(vr)
        
        frame_step = max(int(original_fps / fps), 1)
        frame_idx = list(range(0, total_frame_num, frame_step))
        
        if len(frame_idx) > max_frames_num:
            frame_idx = list(np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int))
        
        frames = vr.get_batch(frame_idx).asnumpy()
        return frames

    
    def _encode_video_minicmp(self, video_path, max_frames_num=32):
        frames = self._gen_frames_numpy(video_path, max_frames_num)
        frames = [Image.fromarray(v.astype('uint8')) for v in frames]
        return frames
    
    def _encode_video_minicmp_fps(self, video_path, fps=1, max_frames_num=32):
        frames = self._gen_frames_numpy_fps(video_path, fps, max_frames_num)
        frames = [Image.fromarray(v.astype('uint8')) for v in frames]
        return frames
    
    def _load_frames_from_png(self, png_folder, num_threads=4):
        #return frames as numpy
        png_files = sorted([f for f in os.listdir(png_folder) if f.endswith('.png')])
        def load_image(png_file):
            return np.array(Image.open(os.path.join(png_folder, png_file)))
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            frames = list(executor.map(load_image, png_files))
        frames = np.stack(frames)
        return frames
    
    def _conditional_imports(self, model_name):
        if model_name == 'LongVA-7B-DPO':
            try:
                global VideoReader, cpu  
                from decord import VideoReader, cpu
            except ImportError:
                raise ImportError("The library 'decord' is required for video processing but is not installed.")
        
        elif model_name == 'MiniCPM-V-2_6':  
            try: 
                global cv2, Image  
                import cv2
                from PIL import Image
            except ImportError:
                raise ImportError("The libraries 'cv2' and 'PIL' are required for image processing but are not installed.")
    
def gen_QAs_dataloader(args, batch_size=1, shuffle=False, num_workers=2, prefetch_factor=2, collate_fn=None):
    dataset = QADataset(args)
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        prefetch_factor=prefetch_factor, 
        collate_fn=collate_fn  
    )
    
    return dataloader


def unpack_QAs(QAs):
    final_question, video_path, correct_answer, video, segment, question = QAs
    # print(segment, final_question)
    final_question, video_path, correct_answer, video, segment, question = final_question[0], video_path[0], correct_answer[0], video[0], [segment[0][0], segment[1][0]], question[0]
    return final_question, video_path, correct_answer, video, segment, question


def gen_QAs(QA_file_path, video_folder):
    QAs = []  
    choices = "A"
    message = [f'Please watch the video and list all events related to the object X that is highlighted by the colored bounding box.',
            f'Each event is a descriptive and detailed sentences, e.g., attributes of X, X\'s environment, other objects that interact with X).',
            # f'The subject of the sentences MUST be the X.',
            # f'There are timestamps like \"1.5s\" display in every frame.',
            # f'Your answer should be a list of events with timestamps.',
            # f'Please provide your answer in the following format:\n',
            # f'\"Start time 1: End time 1: Event 1 description\"\n\"Start time 2: End time 2: Event 2 description\"\n',\
            # f'Please make sure merge the events that are closely related to each other. Do not provide the same event multiple times.'
        ]
    message = ' '.join(message)
    with open(QA_file_path, 'r') as file:
        data = json.load(file)
    for item in data:
        video = item['video']
        segment = item['segment']
        # task_class = item['class']
        question = item['question']
        # options = '\n'.join([f"({key}){value}" for key, value in item["options"].items()])
        correct_answer = item['answer']
        final_question = message + question
        video_path = os.path.join(video_folder, video)

        QAs.append((final_question, video_path, correct_answer, video, segment, question))
    return QAs


import os
from datetime import datetime

def get_answers_output_path(args):
    QA_file_name = os.path.splitext(os.path.basename(args.QA_file_path))[0]
    current_time = datetime.now().strftime('%Y%m%d_%H%M')
    print(current_time)
    model_name = os.path.basename(args.model_path)
    provided_boundaries = 'wb' if args.provide_boundaries else 'nb'
    if args.extract_frames_method == 'fps':
        frame_count = str(args.fps)
    elif args.extract_frames_method == 'max_frames_num':
        frame_count = str(args.max_frames_num)
    answers_output_name = model_name + '_' + QA_file_name + '_' + args.extract_frames_method + '_' + frame_count + '_' + provided_boundaries + '.json'
    answers_output_path = os.path.join(args.answers_output_folder, answers_output_name)
    return answers_output_path


def basic_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--QA_file_path', type=str)
    parser.add_argument('--video_folder', type=str)
    parser.add_argument('--provide_boundaries', action='store_true')
    parser.add_argument('--answers_output_folder', type=str)
    parser.add_argument('--extract_frames_method', type=str, choices=['fps', 'max_frames_num'])
    parser.add_argument('--frames_from', type=str, choices=['video', 'images'])
    # Conditional argument groups for different extract_frames_method options
    fps_group = parser.add_argument_group('fps_method')
    fps_group.add_argument('--fps', type=int, help='Frames per second for extraction')

    parser.add_argument('--max_frames_num', type=int, help='Maximum number of frames to extract')
    parser.add_argument('--final_json_path', type=str, help='Path to save the final json file')
    
    return parser

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

def log_error_to_file(exc):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    error_log_path = os.path.join(script_dir, 'error_log.txt')

    with open(error_log_path, 'a', encoding='utf-8') as f:
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=f)

def generate_question_tip():
    question_tip = [
        "Watch the video and choose the best answer to the multiple-choice question that follows. Respond only with the letter (A, B, C, or D) corresponding to your choice.",
        "Please view the video and select the most appropriate answer to the following multiple-choice question. Reply solely with the letter (A, B, C, or D) of your chosen option.",
        "After watching the video, pick the best answer to the multiple-choice question below. Provide just the letter (A, B, C, or D) that matches your selection.",
        "Kindly watch the video and choose the correct option in the upcoming multiple-choice question. Answer only with the letter (A, B, C, or D) corresponding to your choice.",
        "Please watch the video and select the most suitable answer to the subsequent multiple-choice question. Respond by giving only the letter (A, B, C, or D) of your selected option."
    ]
    chosen_tip = random.choice(question_tip)
    return chosen_tip
