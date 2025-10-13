import os
import glob
from pathlib import Path
import decord
from .dataset import VideoDataset
from ..utils import generate_coordinates
from ..configs import RunConfig, StrainerConfig, STRAINERNetConfig
from fractions import Fraction


def cyclic_even_spacing(start: int, end: int, num_elements: int):
    step = (end - start + 1) / num_elements
    result: list[int] = []
    for i in range(num_elements):
        value = round(start + i * step) % (end + 1)
        result.append(value)
    if len(result) != num_elements:
        raise RuntimeError("cyclic_even_spacing is broken")
    return result


class DataProcess:
    """
    Data processing pipeline for Video/Image data.
    """

    def __init__(self, cfg: RunConfig, data_path: Path, training: bool = False):
        self.data_path = data_path
        self.cfg = cfg.encoder_cfg
        self.run_cfg = cfg
        self.training = training
        if training and not isinstance(cfg.trainer_cfg, StrainerConfig):
            raise ValueError("Strainer required for training dataset")
        self.build()

    def setup_coords(self):
        self.data_shape = self.data_set.data_shape
        self.input_data_shape = self.data_set.input_data_shape
        normalize_range = self.cfg.normalize_range

        if isinstance(self.cfg.net, STRAINERNetConfig):
            correct_pos_enc_cfg = self.cfg.net.mlp_cfg.pos_encode_cfg
        else:
            correct_pos_enc_cfg = self.cfg.net.pos_encode_cfg

        self.coordinates = generate_coordinates(
            input_shape=self.data_shape,  # type: ignore
            patch_size=self.cfg.patch_size,
            normalize_range=normalize_range,
            positional_encoding=correct_pos_enc_cfg,
            patch_scale=Fraction(1, 1) if self.cfg.patch_scales is not None else None,
        )

    def build(self):
        if self.data_path.is_dir() and all(
            map(lambda x: x.is_dir(), self.data_path.iterdir())
        ):
            # If we want to treat multiple videos as one video
            self.data_set = self.load_multiple_data_dir()
        elif os.path.isfile(self.data_path) and self.data_path.suffix in (
            (".mp4", ".avi", ".mov", ".mkv")
        ):
            self.data_set = self.load_video()
        elif os.path.isfile(self.data_path) and self.data_path.suffix == ".yuv":
            self.data_set = self.load_data_yuv()
        elif os.path.isdir(self.data_path):
            self.data_set = self.load_data_dir()
        else:
            raise ValueError(f"Invalid data path: {self.data_path}")

        self.setup_coords()
        if self.training and isinstance(self.run_cfg.trainer_cfg, StrainerConfig):
            meta_frames = self.run_cfg.trainer_cfg.meta_frames
            self.frame_idx = cyclic_even_spacing(0, self.num_frames - 1, meta_frames)

    def load_data_yuv(self) -> VideoDataset:
        data_set = VideoDataset(self.data_path, self.cfg)
        assert self.cfg.yuv_size is not None
        H, W = self.cfg.yuv_size
        frame_size = H * W * 3 // 2
        self.num_frames = self.data_path.stat().st_size // frame_size
        return data_set

    def load_data_dir(self) -> VideoDataset:
        files = glob.glob(str(self.data_path / "*"))
        files = sorted(files)
        inference = False
        for file in files:
            if "dataset_state.json" in file:
                inference = True

        if len(files) == 0:
            raise ValueError(f"Empty directory: {self.data_path}")
        elif not files[0].endswith((".png", ".jpg", ".jpeg")) and not inference:
            raise ValueError(f"Unsupported file format: {files[0]}")

        data_set = VideoDataset(files, self.cfg, not inference)
        self.num_frames = len(files)
        if inference:
            data_set.load_state(self.data_path)
            self.num_frames = len(data_set)
        return data_set

    def load_multiple_data_dir(self) -> VideoDataset:
        files = sorted(map(str, self.data_path.rglob("*/*")))
        inference = False
        for file in files:
            if "dataset_state.json" in file:
                inference = True

        if len(files) == 0:
            raise ValueError(f"Empty directory: {self.data_path}")
        else:
            if not files[0].endswith((".png", ".jpg", ".jpeg")):
                raise ValueError(f"Unsupported file format: {files[0]}")

        data_set = VideoDataset(files, self.cfg, not inference)
        if inference:
            data_set.load_state(self.data_path)
            self.num_frames = len(data_set)
        self.num_frames = len(files)
        return data_set

    def load_video(self) -> VideoDataset:
        self.video_reader = decord.VideoReader(str(self.data_path))
        decord.bridge.set_bridge("torch")

        self.patch_size = self.cfg.patch_size
        data_set = VideoDataset(self.video_reader, self.cfg)
        self.num_frames = len(self.video_reader)

        return data_set
