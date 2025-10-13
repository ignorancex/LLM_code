import os

import numpy as np
import pandas as pd
import torch

import decord


class TorchDatasetWithDecord(torch.utils.data.Dataset):
    def __init__(self,
                 csv_path,
                 max_samples=None,
                 sampling_rate=16_000):
        self.sampling_rate = sampling_rate

        # metadata is a list of file paths that each line includes "{id}.mp4, 'test label'" in the first two columns.
        self.basedir = os.path.dirname(csv_path)
        self.metadata = pd.read_csv(csv_path, header=None)

        if max_samples is not None and len(self.metadata) > max_samples:
            # ok to fix a seed because this is for the debugging purpose.
            self.metadata = self.metadata.sample(n=max_samples, random_state=83)

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        mp4_relpath = self.metadata.iloc[idx][0]
       
        if not mp4_relpath.endswith(".mp4"):
            mp4_relpath += ".mp4"

        full_path = os.path.join(self.basedir, mp4_relpath)

        reader = decord.AVReader(full_path,
                                 ctx = decord.cpu(0),
                                 sample_rate = self.sampling_rate)

        # load text description
        text = self.metadata.iloc[idx][1]

        return {"av_reader": reader, "text": text}



def _get_collate_fn(duration_per_sample=6,
                    sampling_rate=16_000,
                    tokenizer=None,
                    n_segments_per_video=1,
                    min_interval_per_segment=0,
                    sample_random_segments=False,
                    need_segments_dims=False):
    n_segments_per_video = max(n_segments_per_video, 1)
    
    def collate_fn(examples):
        audio_list = []
        video_list = []
        caption_list = []
        for x in examples:
            reader = x["av_reader"]
            n_frames_per_vid = len(reader)
            fps = int(reader._AVReader__video_reader.get_avg_fps())

            # sampling segments
            n_frames_per_seg = fps * duration_per_sample
            if n_segments_per_video == 1 and not sample_random_segments:
                segments = [0]
            else:
                if sample_random_segments:
                    segments = []
                    for seg_id in range(n_segments_per_video):
                        cur_start_v = 0 if len(segments) == 0 else segments[-1] + fps * min_interval_per_segment
                        cur_end_v = max(n_frames_per_vid, (n_segments_per_video - 1) * n_frames_per_seg) \
                                    - n_frames_per_seg - (n_segments_per_video - seg_id - 1) * min_interval_per_segment * fps
                        
                        cur_end_v = max(cur_end_v, 0)
                        try:
                            start_v = np.random.randint(low=cur_start_v, high=cur_end_v + 1)
                        except Exception as e:
                            print(f"n_frames_per_vid: {n_frames_per_vid}\n",
                                  f"n_frames_per_seg: {n_frames_per_seg}\n",
                                  f"n_segments_per_video: {n_segments_per_video}\n",
                                  f"min_interval_per_segment: {min_interval_per_segment}\n",
                                  f"segments: {segments}")
                            raise ValueError(e)
                        
                        segments.append(start_v)
                else:
                    n_frames_interval = fps * min_interval_per_segment
                    segments = list(range(0, 
                                          n_segments_per_video * n_frames_interval,
                                          n_frames_interval))

            cur_video_list = []
            cur_audio_list = []
            # return frames in random order
            seg_gen = np.random.permutation \
                if sample_random_segments else range
            for seg_id in seg_gen(n_segments_per_video):
                # sample audio and video
                start_v = segments[seg_id]
                end_v = start_v + n_frames_per_seg
                
                n_samples = duration_per_sample * sampling_rate
                subclip_a, subclip_v = reader[start_v:end_v]
                subclip_v = subclip_v.asnumpy().transpose((0, 3, 1, 2))
                try:
                    subclip_a = np.concatenate([x.asnumpy() for x in subclip_a], axis=1).reshape(-1)
                except Exception as e:
                    print(f"start_v: {start_v}\n",
                            f"end_v: {end_v}\n",
                            f"n_frames_per_vid: {n_frames_per_vid}\n",
                            f"n_frames_per_seg: {n_frames_per_seg}\n",
                            )
                    raise ValueError(e)
                
                if len(subclip_v) < n_frames_per_seg:
                    # Note: not sure it's ok to pad with 0
                    subclip_v = np.pad(subclip_v, 
                                       ((0, n_frames_per_seg - len(subclip_v)), (0, 0), (0, 0), (0, 0)), 
                                       mode="constant",
                                       constant_values=0)
                cur_video_list.append(subclip_v)

                if len(subclip_a) < n_samples:
                    # Note: not sure it's ok to pad with 0
                    subclip_a = np.pad(subclip_a, 
                                       (0, n_samples - len(subclip_a)), 
                                       mode="constant", 
                                       constant_values=0)
                cur_audio_list.append(subclip_a)

            if len(cur_video_list) == 1 and len(cur_audio_list) == 1:
                cur_audio_list = cur_audio_list[0]
                cur_video_list = cur_video_list[0]
                
                if need_segments_dims:
                    cur_audio_list = cur_audio_list[None]
                    cur_video_list = cur_video_list[None]
            else:
                cur_audio_list = np.asarray(cur_audio_list)
                cur_video_list = np.asarray(cur_video_list)

            audio_list.append(cur_audio_list)
            video_list.append(cur_video_list)
            caption_list.append(x["text"])
        
        ret =  {
            "audio": torch.from_numpy(np.asarray(audio_list)).float(),
            "video": torch.from_numpy(np.asarray(video_list)).to(memory_format=torch.contiguous_format).float()
        }

        if tokenizer is not None:
            ret["text"] = tokenizer(caption_list)

        return ret

    return collate_fn


def get_dataset_and_collate_fn_from_csv(csv_path,
                                        max_samples=None,
                                        duration_per_sample=6,
                                        sampling_rate=16_000,
                                        tokenizer=None,
                                        n_segments_per_video=1,
                                        min_interval_per_segment=0,
                                        sample_random_segments=False,
                                        need_segments_dims=False):
    # instantiate dataset class
    dataset = TorchDatasetWithDecord(csv_path, 
                                        max_samples, 
                                        sampling_rate=sampling_rate)
        
    print(f"The total number of data is {len(dataset):,}.")
    
    # create collate function
    collate_fn = _get_collate_fn(duration_per_sample=duration_per_sample,
                                 sampling_rate=sampling_rate,
                                 tokenizer=tokenizer,
                                 n_segments_per_video=n_segments_per_video,
                                 sample_random_segments=sample_random_segments,
                                 min_interval_per_segment=min_interval_per_segment,
                                 need_segments_dims=need_segments_dims)
    
    return dataset, collate_fn
