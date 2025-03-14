# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import random
import torch
import torch.utils.data
from fvcore.common.file_io import PathManager
from torchvision import transforms
import numpy as np
import time

import timesformer.utils.logging as logging

from . import multi_decoder as decoder
from . import utils as utils
from . import video_container as container
from .build import DATASET_REGISTRY
logger = logging.get_logger(__name__)


@DATASET_REGISTRY.register()
class Multisample(torch.utils.data.Dataset):
    """
    Finegym video loader. Construct the Finegym video loader, then sample
    clips from the videos. For training and validation, a single clip is
    randomly sampled from every video with random cropping, scaling, and
    flipping. For testing, multiple clips are uniformaly sampled from every
    video with uniform cropping. For uniform cropping, we take the left, center,
    and right crop if the width is larger than height, or take top, center, and
    bottom crop if the height is larger than the width.
    """

    def __init__(self, cfg, mode, num_retries=10):
        """
        Construct the Finegym video loader with a given csv file. The format of
        the csv file is:
        ```
        path_to_video_1 label_1
        path_to_video_2 label_2
        ...
        path_to_video_N label_N
        ```
        Args:
            cfg (CfgNode): configs.
            mode (string): Options includes `train`, `val`, or `test` mode.
                For the train and val mode, the data loader will take data
                from the train or val set, and sample one clip per video.
                For the test mode, the data loader will take data from test set,
                and sample multiple clips per video.
            num_retries (int): number of retries.
        """
        # Only support train, val, and test mode.
        assert mode in [
            "train",
            "val",
            "unlabel",
            "test",
        ], "Split '{}' not supported for Finegym".format(mode)
        self.mode = mode
        self.cfg = cfg

        self._video_meta = {}
        self._num_retries = num_retries
        # For training or validation mode, one single clip is sampled from every
        # video. For testing, NUM_ENSEMBLE_VIEWS clips are sampled from every
        # video. For every clip, NUM_SPATIAL_CROPS is cropped spatially from
        # the frames.
        self._num_clips = 1

        logger.info("Constructing Finegym {}...".format(mode))
        self._construct_loader()

    def _construct_loader(self):
        """
        Construct the video loader.
        """
        path_to_file = os.path.join(
            self.cfg.DATA.PATH_TO_DATA_DIR, "{}.csv".format(self.mode)
        )
        assert PathManager.exists(path_to_file), "{} dir not found".format(
            path_to_file
        )

        self._path_to_videos = []
        self._labels = []
        self._spatial_temporal_idx = []
        with PathManager.open(path_to_file, "r") as f:
            for clip_idx, path_label in enumerate(f.read().splitlines()):
                assert (
                    len(path_label.split(self.cfg.DATA.PATH_LABEL_SEPARATOR))
                    == 2
                )
                path, label = path_label.split(
                    self.cfg.DATA.PATH_LABEL_SEPARATOR
                )
                
                for idx in range(self._num_clips):
                    self._path_to_videos.append(
                        os.path.join(self.cfg.DATA.PATH_PREFIX, path)
                    )
                    self._labels.append(int(label))
                    self._spatial_temporal_idx.append(idx)
                    self._video_meta[clip_idx * self._num_clips + idx] = {}
        assert (
            len(self._path_to_videos) > 0
        ), "Failed to load Finegym split {} from {}".format(
            self._split_idx, path_to_file
        )
        logger.info(
            "Constructing kinetics dataloader (size: {}) from {}".format(
                len(self._path_to_videos), path_to_file
            )
        )

    def __getitem__(self, index):
        """
        Given the video index, return the list of frames, label, and video
        index if the video can be fetched and decoded successfully, otherwise
        repeatly find a random video that can be decoded as a replacement.
        Args:
            index (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (int): the label of the current video.
            index (int): if the video provided by pytorch sampler can be
                decoded, then return the index of the video. If not, return the
                index of the video replacement that can be decoded.
        """
        short_cycle_idx = None
        # When short cycle is used, input index is a tupple.
        if isinstance(index, tuple):
            index, short_cycle_idx = index

        if self.mode in ["train", "val", "unlabel"]:
            # -1 indicates random sampling.
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[0]
            max_scale = self.cfg.DATA.TRAIN_JITTER_SCALES[1]
            crop_size = self.cfg.DATA.TRAIN_CROP_SIZE
            if short_cycle_idx in [0, 1]:
                crop_size = int(
                    round(
                        self.cfg.MULTIGRID.SHORT_CYCLE_FACTORS[short_cycle_idx]
                        * self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
            if self.cfg.MULTIGRID.DEFAULT_S > 0:
                # Decreasing the scale is equivalent to using a larger "span"
                # in a sampling grid.
                min_scale = int(
                    round(
                        float(min_scale)
                        * crop_size
                        / self.cfg.MULTIGRID.DEFAULT_S
                    )
                )
        elif self.mode in ["test"]:
            temporal_sample_index = -1
            spatial_sample_index = -1
            min_scale, max_scale, crop_size = (
                [self.cfg.DATA.TEST_CROP_SIZE] * 3
            )
            # The testing is deterministic and no jitter should be performed.
            # min_scale, max_scale, and crop_size are expect to be the same.
            assert len({min_scale, max_scale}) == 1
        else:
            raise NotImplementedError(
                "Does not support {} mode".format(self.mode)
            )
        sampling_rate = utils.get_random_sampling_rate(
            self.cfg.MULTIGRID.LONG_CYCLE_SAMPLING_RATE,
            self.cfg.DATA.SAMPLING_RATE,
        )
        num_frames_list = self.cfg.DATA.SAMPLE_FRAMES
        # Try to decode and sample a clip from a video. If the video can not be
        # decoded, repeatly find a random video replacement that can be decoded.
        for i_try in range(self._num_retries):
            # Decode video in each granularity. Meta info is used to perform selective decoding.
            # frames_list = []
            # for nframe in num_frames_list:
            #     frames = decoder.decode(
            #         self._path_to_videos[index],
            #         sampling_rate,
            #         nframe,
            #         temporal_sample_index,
            #         self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
            #         whole=True,
            #     )
            #     # If decoding failed, break
            #     if frames is None:
            #         break
            #     frames_list.append(frames)

            # frames_aug_list = []
            # for nframe in num_frames_list:
            #     frames_aug = decoder.decode(
            #         self._path_to_videos[index],
            #         sampling_rate,
            #         nframe,
            #         temporal_sample_index,
            #         self.cfg.TEST.NUM_ENSEMBLE_VIEWS,
            #         whole=True,
            #     )
            #     if frames_aug is None:
            #         break
            #     frames_aug_list.append(frames_aug)
            frames_list = decoder.decode(self._path_to_videos[index], num_frames_list)
            frames_aug_list = decoder.decode(self._path_to_videos[index], num_frames_list)

            # If decoding failed (wrong format, video is too short, and etc),
            # select another video.
            if frames_list is None or frames_aug_list is None or (len(frames_list) != len(num_frames_list)) or (len(frames_aug_list) != len(num_frames_list)):
                logger.warning(
                    "Failed to decode video idx {} from {}; trial {}".format(
                        index, self._path_to_videos[index], i_try
                    )
                )
                # if self.mode not in ["test"] and i_try > self._num_retries // 2:
                if i_try > self._num_retries // 2:
                    # let's try another one
                    index = random.randint(0, len(self._path_to_videos) - 1)
                continue
            
            label = self._labels[index]
            frames_strong_list = []
            for frames_aug in frames_aug_list:
                frames_strong = frames_aug
                if frames_strong is None:
                    from . import video_transform
                    from PIL import Image

                    aug_transform = video_transform.create_random_augment(
                        input_size=(crop_size, crop_size),
                        auto_augment='rand-m9-n5-mstd0.5-inc1',
                        interpolation='bicubic',
                    )
                    frames_strong = frames_aug.permute(0, 3, 1, 2)

                    frames_strong = [
                        transforms.ToPILImage()(frame) for frame in frames_strong
                    ]

                    frames_strong = aug_transform(frames_strong)
                    frames_strong = [transforms.ToTensor()(img) for img in frames_strong]
                    frames_strong = torch.stack(frames_strong)  # T C H W

                    frames_strong = frames_strong.permute(0, 2, 3, 1)  # T H W C
                    frames_strong = frames_strong * 255.0

                    frames_strong = utils.tensor_normalize(
                        frames_strong, self.cfg.DATA.MEAN, self.cfg.DATA.STD
                    )
                    frames_strong = frames_strong.permute(3, 0, 1, 2)

                frames_strong = utils.tensor_normalize(
                    frames_strong, self.cfg.DATA.MEAN, self.cfg.DATA.STD
                )
                frames_strong = frames_strong.permute(3, 0, 1, 2)
                frames_strong = utils.spatial_sampling(
                    frames_strong,
                    spatial_idx=spatial_sample_index,
                    min_scale=min_scale,
                    max_scale=max_scale,
                    crop_size=crop_size,
                    random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                    inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
                )
                frames_strong_list.append(frames_strong)

            frames_weak_list = []
            for frames in frames_list:
                frames = utils.tensor_normalize(
                    frames, self.cfg.DATA.MEAN, self.cfg.DATA.STD
                )
                # T H W C -> C T H W.
                frames = frames.permute(3, 0, 1, 2)

                if self.mode == 'unlabel':
                    # Perform data augmentation.
                    frames_weak = utils.spatial_sampling(
                        frames,
                        spatial_idx=4,
                        min_scale=min_scale,
                        max_scale=max_scale,
                        crop_size=crop_size,
                        random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                        inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
                    )
                else:
                    frames_weak = utils.spatial_sampling(
                        frames,
                        spatial_idx=-1,
                        min_scale=min_scale,
                        max_scale=max_scale,
                        crop_size=crop_size,
                        random_horizontal_flip=self.cfg.DATA.RANDOM_FLIP,
                        inverse_uniform_sampling=self.cfg.DATA.INV_UNIFORM_SAMPLE,
                    )
                frames_weak_list.append(frames_weak)

            if not self.cfg.MODEL.ARCH in ['vit']:
                if self.mode != "test":
                    for i in range(len(frames_weak_list)):
                        frames_weak_list[i] = utils.pack_pathway_output(self.cfg, frames_weak_list[i])
                    for i in range(len(frames_strong_list)):
                        frames_strong_list[i] = utils.pack_pathway_output(self.cfg, frames_strong_list[i])
                else:
                    for i in range(len(frames_weak_list)):
                        frames_weak_list[i] = torch.index_select(
                            frames_weak_list[i],
                            1,
                            torch.linspace(
                                0, frames_weak_list[i].shape[1] - 1, num_frames_list[i]

                            ).long(),
                        )
                    for i in range(len(frames_strong_list)):
                        frames_strong_list[i] = torch.index_select(
                            frames_strong_list[i],
                            1,
                            torch.linspace(
                                0, frames_strong_list[i].shape[1] - 1, num_frames_list[i]

                            ).long(),
                        )
            else:
                # Perform temporal sampling from the fast pathway.
                for i in range(len(frames_weak_list)):
                    frames_weak_list[i] = torch.index_select(
                        frames_weak_list[i],
                        1,
                        torch.linspace(
                            0, frames_weak_list[i].shape[1] - 1, num_frames_list[i]

                        ).long(),
                    )
                for i in range(len(frames_strong_list)):
                    frames_strong_list[i] = torch.index_select(
                        frames_strong_list[i],
                        1,
                        torch.linspace(
                            0, frames_strong_list[i].shape[1] - 1, num_frames_list[i]

                        ).long(),
                    )
            return frames_weak_list, frames_strong_list, label, index,  {}
        else:
            raise RuntimeError(
                "Failed to fetch video after {} retries.".format(
                    self._num_retries
                )
            )

    def __len__(self):
        """
        Returns:
            (int): the number of videos in the dataset.
        """
        return len(self._path_to_videos)
