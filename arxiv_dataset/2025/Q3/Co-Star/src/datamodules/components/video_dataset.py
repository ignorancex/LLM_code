# import torch.utils.data as data
# import numpy as np
# from PIL import Image
# import os
# import glob
# import torch

# class VideoDataset(data.Dataset):
#     def __init__(
#         self,
#         list_file,
#         num_segments=16,
#         new_length=1,
#         transform=None,
#         random_shift=True,
#         test_mode=False,
#         index_bias=1,
#         data_folder=None,
#     ):
#         self.list_file = list_file
#         self.num_segments = num_segments
#         self.new_length = new_length
#         self.transform = transform
#         self.random_shift = random_shift
#         self.test_mode = test_mode
#         self.index_bias = index_bias
#         self.data_folder = data_folder
#         self._parse_list()

#     def _parse_list(self):
#         """Parse the file list and load video paths."""
#         self.video_list = []
#         with open(self.list_file, 'r') as f:
#             for line in f.readlines():
#                 label, video_path = line.strip().split('\t')[1:3]
#                 self.video_list.append(VideoRecord(video_path, int(label), self.data_folder))

#     def _sample_indices(self, record):
#         """Sample indices for frames from each segment."""
#         if record.num_frames <= self.num_segments:
#             indices = np.arange(record.num_frames)
#             indices = np.pad(indices, (0, self.num_segments - record.num_frames), 'wrap')
#             return np.sort(indices)
#         ticks = [i * record.num_frames // self.num_segments for i in range(self.num_segments + 1)]
#         offsets = [tick + np.random.randint(0, max(1, ticks[i + 1] - tick)) for i, tick in enumerate(ticks[:-1])]
#         offsets = np.clip(offsets, 0, record.num_frames - 1)  # Ensure indices are within bounds
#         return np.array(offsets) + self.index_bias

#     def _sample_indices_alt(self, record):
#         """Sample alternative indices for frames from each segment."""
#         indices = self._sample_indices(record)
#         alt_indices = []
#         for i in range(self.num_segments):
#             start = i * record.num_frames // self.num_segments
#             end = (i + 1) * record.num_frames // self.num_segments
#             segment_indices = list(range(start, end))
#             if indices[i] in segment_indices:
#                 segment_indices.remove(indices[i])
#             if segment_indices:
#                 alt_indices.append(np.random.choice(segment_indices))
#             else:
#                 alt_indices.append(indices[i])  # If no alternative, use the same index
#         return np.array(alt_indices) + self.index_bias

#     def _get_val_indices(self, record):
#         """Get indices for validation."""
#         if self.num_segments == 1:
#             return np.array([record.num_frames // 2], dtype=np.int32) + self.index_bias
#         ticks = [i * record.num_frames // self.num_segments for i in range(self.num_segments + 1)]
#         offsets = [tick + (ticks[i + 1] - tick) // 2 for i, tick in enumerate(ticks[:-1])]
#         return np.array(offsets) + self.index_bias

#     def __getitem__(self, index):
#         record = self.video_list[index]
#         if self.random_shift:  # Training mode
#             segment_indices = self._sample_indices(record)
#             alt_segment_indices = self._sample_indices_alt(record)
#             global_images, label = self.get(record, segment_indices)
#             alt_global_images, _ = self.get(record, alt_segment_indices, use_alt_transform=True)
#             return global_images, alt_global_images, label
#         else:  # Validation mode
#             segment_indices = self._get_val_indices(record)
#             images, label = self.get(record, segment_indices)
#             return images, label

#     def get(self, record, indices, use_alt_transform=False):
#         images = []
#         frames = sorted(glob.glob(os.path.join(record.path, '*.jpg')))
        
#         for seg_ind in indices:
#             p = int(seg_ind) % len(frames)
#             seg_img = Image.open(frames[p]).convert("RGB")
#             if self.transform:
#                 if use_alt_transform:
#                     # Apply a different random augmentation
#                     seg_img = self.transform(seg_img)
#                 else:
#                     seg_img = self.transform(seg_img)
#             images.append(seg_img)
#         images = torch.stack(images, dim=0)
#         return images, record.label

#     def __len__(self):
#         return len(self.video_list)

# class VideoRecord:
#     def __init__(self, path, label, data_folder=None):
#         self.path = os.path.join(data_folder, path) if data_folder else path
#         self.label = label
#         self.num_frames = len(glob.glob(os.path.join(self.path, '*.jpg')))

import torch.utils.data as data
import numpy as np
from PIL import Image
import os
import glob
import torch

class VideoDataset(data.Dataset):
    def __init__(
        self,
        list_file,
        num_segments=16,
        new_length=1,
        transform=None,
        random_shift=True,
        test_mode=False,
        index_bias=1,
        data_folder=None,
    ):
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.index_bias = index_bias
        self.data_folder = data_folder
        self._parse_list()

    def _parse_list(self):
        """Parse the file list and load video paths."""
        self.video_list = []
        with open(self.list_file, 'r') as f:
            for idx, line in enumerate(f.readlines()):
                label, video_path = line.strip().split('\t')[1:3]
                self.video_list.append(VideoRecord(idx, video_path, int(label), self.data_folder))

    def _sample_indices(self, record):
        """Sample indices for frames from each segment."""
        if record.num_frames <= self.num_segments:
            indices = np.arange(record.num_frames)
            indices = np.pad(indices, (0, self.num_segments - record.num_frames), 'wrap')
            return np.sort(indices)
        ticks = [i * record.num_frames // self.num_segments for i in range(self.num_segments + 1)]
        offsets = [tick + np.random.randint(0, max(1, ticks[i + 1] - tick)) for i, tick in enumerate(ticks[:-1])]
        offsets = np.clip(offsets, 0, record.num_frames - 1)  # Ensure indices are within bounds
        return np.array(offsets) + self.index_bias

    def _sample_indices_alt(self, record):
        """Sample alternative indices for frames from each segment."""
        indices = self._sample_indices(record)
        alt_indices = []
        for i in range(self.num_segments):
            start = i * record.num_frames // self.num_segments
            end = (i + 1) * record.num_frames // self.num_segments
            segment_indices = list(range(start, end))
            if indices[i] in segment_indices:
                segment_indices.remove(indices[i])
            if segment_indices:
                alt_indices.append(np.random.choice(segment_indices))
            else:
                alt_indices.append(indices[i])  # If no alternative, use the same index
        return np.array(alt_indices) + self.index_bias

    def _get_val_indices(self, record):
        """Get indices for validation."""
        if self.num_segments == 1:
            return np.array([record.num_frames // 2], dtype=np.int32) + self.index_bias
        ticks = [i * record.num_frames // self.num_segments for i in range(self.num_segments + 1)]
        offsets = [tick + (ticks[i + 1] - tick) // 2 for i, tick in enumerate(ticks[:-1])]
        return np.array(offsets) + self.index_bias

    def __getitem__(self, index):
        record = self.video_list[index]
        if self.random_shift:  # Training mode
            segment_indices = self._sample_indices(record)
            alt_segment_indices = self._sample_indices_alt(record)
            global_images, label = self.get(record, segment_indices)
            alt_global_images, _ = self.get(record, alt_segment_indices, use_alt_transform=True)
            return global_images, alt_global_images, label, record.idx
        else:  # Validation mode
            segment_indices = self._get_val_indices(record)
            images, label = self.get(record, segment_indices)
            return images, label, record.idx

    def get(self, record, indices, use_alt_transform=False):
        images = []
        frames = sorted(glob.glob(os.path.join(record.path, '*.jpg')))
        
        for seg_ind in indices:
            p = int(seg_ind) % len(frames)
            seg_img = Image.open(frames[p]).convert("RGB")
            if self.transform:
                if use_alt_transform:
                    # Apply a different random augmentation
                    seg_img = self.transform(seg_img)
                else:
                    seg_img = self.transform(seg_img)
            images.append(seg_img)
        images = torch.stack(images, dim=0)
        return images, record.label

    def __len__(self):
        return len(self.video_list)

class VideoRecord:
    def __init__(self, idx, path, label, data_folder=None):
        self.idx = idx
        self.path = os.path.join(data_folder, path) if data_folder else path
        self.label = label
        self.num_frames = len(glob.glob(os.path.join(self.path, '*.jpg')))