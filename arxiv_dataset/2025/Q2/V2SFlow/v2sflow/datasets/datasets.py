import os
import numpy as np
import torch
import pandas as pd
from v2sflow.registry import DATASETS
from v2sflow.datasets.utils import cut_or_pad, pad
from third_party.fairseq.data.fairseq_dataset import FairseqDataset

def read_file(input_path):
    if input_path.endswith(".csv"):
        return pd.read_csv(input_path)
    elif input_path.endswith(".tsv"):
        return pd.read_csv(input_path, sep="\t")
    elif input_path.endswith(".parquet"):
        return pd.read_parquet(input_path)
    else:
        raise NotImplementedError(f"Unsupported file format: {input_path}")

@DATASETS.register_module()
class VariableFeatureDataset(FairseqDataset):
    def __init__(self,
        data_path,
        root_path = "",
        max_sample_size = None,
        min_sample_size = None,
        video_feat_name = None,
        video_fps = 25,
        audio_feat_name = None,
        audio_fps = 100,
        audio_max = None,
        audio_min = None,
        audio_stack = 1,
        content_name = None,
        content_fps = 50,
        pitch_name = None,
        pitch_fps = 25,
        speaker_name = None,
        shuffle = False,
    ):
        self.data_path = data_path
        self.data = read_file(data_path)

        if max_sample_size is not None:
            print(f"# of data samples: {len(self.data)}")
            self.data = self.data[self.data['n_video'] <= max_sample_size].reset_index(drop=True)
            print(f"# of data samples after filtering with max_sample_size {max_sample_size}: {len(self.data)}")
        if min_sample_size is not None:
            print(f"# of data samples: {len(self.data)}")
            self.data = self.data[self.data['n_video'] >= min_sample_size].reset_index(drop=True)
            print(f"# of data samples after filtering with min_sample_size {min_sample_size}: {len(self.data)}")
        # self.data = self.data.iloc[:100].reset_index(drop=True) # for debug

        self.max_sample_size = max_sample_size

        self.sizes = self.data['n_video']

        self.root_path = root_path

        self.video_feat_name = video_feat_name
        self.video_fps = video_fps

        self.audio_feat_name = audio_feat_name
        self.audio_fps = audio_fps
        self.audio_max = audio_max
        self.audio_min = audio_min
        self.audio_stack = audio_stack

        self.content_name = content_name
        self.content_fps = content_fps

        self.pitch_name = pitch_name
        self.pitch_fps = pitch_fps

        self.speaker_name = speaker_name

        self.shuffle = shuffle

    def __len__(self):
        return len(self.sizes)

    def __getitem__(self, index):
        try:
            return self.getitem(index)
        except:
            print("Error: data load failed")
            return {"path": None}

    def getitem(self, index):
        sample = self.data.iloc[index]
        path = os.path.join(self.root_path, sample["video_path"])

        ret = {
            "path": path,
            "num_frames": self.size(index),
        }

        if self.video_feat_name is not None:
            video_path = os.path.splitext(path.replace("/video/", f"/{self.video_feat_name}/"))[0] + ".pth"
            video = torch.load(video_path)

            video_req_len = self.size(index)
            diff = video_req_len - len(video)
            if abs(diff/len(video)) > 0.1 and abs(diff) > 2: # 0.1 sec
                print(f"Warning: wrong video length {len(video)} {video_req_len}")

            video = cut_or_pad(video, video_req_len, mode="replicate")
            ret["video"] = video

        if self.audio_feat_name is not None:
            audio_path = os.path.splitext(path.replace("/video/", f"/{self.audio_feat_name}/"))[0] + ".npy"
            audio = torch.from_numpy(np.load(audio_path))
            if self.audio_max is not None and self.audio_min is not None:
                audio = (audio - self.audio_min) / (self.audio_max - self.audio_min) * 2 - 1

            audio_req_len = round(self.size(index) * (self.audio_fps / self.video_fps))
            diff = audio_req_len - len(audio)
            if abs(diff/len(audio)) > 0.1 and abs(diff) > 10: # 0.1 sec
                print(f"warning: {audio_path}: audio video size mismatch")

            audio = cut_or_pad(audio, audio_req_len, mode="replicate")
            if self.audio_stack > 1:
                audio = self.stacker(audio, self.audio_stack)
            ret["audio"] = audio

        if self.content_name is not None:
            content_path = os.path.splitext(path.replace("/video/", f"/{self.content_name}/"))[0] + ".unit"
            try:
                content = torch.LongTensor(torch.load(content_path))
            except:
                content = torch.LongTensor(list(map(int, open(content_path[:-5]+'.unit').readline().strip().split())))

            content_req_len = round(self.size(index) * (self.content_fps / self.video_fps))
            diff = content_req_len - len(content)
            if abs(diff/len(content)) > 0.1 and abs(diff) > 5: # 0.1 sec
                print(f"warning: {content_path}: content video size mismatch")
            content = cut_or_pad(content,content_req_len, mode="constant", value=content[-1])
            ret["content"] = content

        if self.pitch_name is not None:
            pitch_path = os.path.splitext(path.replace("/video/", f"/{self.pitch_name}/"))[0] + ".txt"
            pitch = torch.LongTensor(list(map(int, open(pitch_path).readline().strip().split())))
            pitch = pitch.repeat_interleave(2) ## 12.5 -> 25

            pitch_req_len = round(self.size(index) * (self.pitch_fps / self.video_fps))
            diff = pitch_req_len - len(pitch)
            if abs(diff/len(pitch)) > 0.1 and abs(diff) > 5: # 0.1 sec
                print(f"warning: {pitch_path}: pitch video size mismatch")
            pitch = cut_or_pad(pitch, pitch_req_len, mode="constant", value=pitch[-1])
            ret["pitch"] = pitch

        if self.speaker_name is not None:
            speaker_path = os.path.splitext(path.replace("/video/", f"/{self.speaker_name}/"))[0] + ".npy"
            speaker = torch.from_numpy(np.load(speaker_path))
            ret["speaker"] = speaker

        return ret

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        return self.sizes[index]

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        order.append(self.sizes)
        return np.lexsort(order)[::-1]

    def collater(self, samples):
        samples = [s for s in samples if s["path"] is not None]
        if len(samples) == 0:
            return {}
        batch = {}
        for data_type in samples[0].keys():
            if data_type in ["path"]:
                batch[data_type] = [s[data_type] for s in samples]
            elif data_type in ["speaker"]:
                batch[data_type] = torch.stack([s[data_type] for s in samples])
            elif data_type in ["video", "audio", "content", "pitch"]:
                pad_val = -1 if data_type == "content" or data_type == "pitch" else 0.0
                c_batch, sample_lengths = pad([s[data_type] for s in samples], pad_val=pad_val)
                batch[data_type + "s"] = c_batch
                batch[data_type + "_lengths"] = torch.tensor(sample_lengths)
        return batch

    def stacker(self, feats, stack_order):
        """
        Concatenating consecutive audio frames
        Args:
        feats - numpy.ndarray of shape [T, F]
        stack_order - int (number of neighboring frames to concatenate
        Returns:
        feats - numpy.ndarray of shape [T', F']
        """
        feat_dim = feats.shape[1]
        if len(feats) % stack_order != 0:
            res = stack_order - len(feats) % stack_order
            res = np.zeros([res, feat_dim]).astype(feats.dtype)
            feats = np.concatenate([feats, res], axis=0)
        feats = feats.reshape((-1, stack_order, feat_dim)).reshape(-1, stack_order*feat_dim)
        return feats
