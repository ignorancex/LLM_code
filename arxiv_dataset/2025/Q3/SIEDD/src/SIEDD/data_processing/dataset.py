import torch
import torchvision
from pathlib import Path
from decord import VideoReader
import cv2
import numpy as np
from scipy.ndimage import zoom
from torchvision.transforms.v2.functional import to_dtype
from ..utils import patchify_images, get_padded_patch_size
from ..configs import EncoderConfig
from .transform import Transform


def yv12_to_rgb(yv12: np.ndarray, width, height):
    Y = yv12[:height]
    V = yv12[height : height + height // 4].reshape((height // 2, width // 2))
    U = yv12[height + height // 4 :].reshape((height // 2, width // 2))
    V = zoom(V, (2, 2), order=0)
    U = zoom(U, (2, 2), order=0)
    return torch.from_numpy(
        cv2.cvtColor(np.stack((Y, U, V), axis=-1), cv2.COLOR_YUV2BGR)
    ).permute(2, 0, 1)


class VideoDataset:
    def __init__(
        self,
        data_list: Path | VideoReader | list[str],
        cfg: EncoderConfig,
        setup: bool = True,
    ):
        self.data_list = data_list
        self.patch_size = cfg.patch_size
        self.resize = cfg.resize
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cached_feats = None
        if isinstance(self.data_list, Path):
            self.yuv_file = self.data_list.open("rb")

        if setup:
            self.setup_data_shape()
        self.transform = Transform(self.cfg.image_transform)

    def setup_data_shape(self) -> None:
        sample = self.load_sample(0)
        self.input_data_shape: list[int] = list(sample.shape)
        if self.resize:
            self.input_data_shape = [3] + list(self.resize)
        self.data_shape = (
            get_padded_patch_size(self.input_data_shape, self.patch_size)
            if self.patch_size
            else self.input_data_shape
        )

    def load_sample(self, idx: int) -> torch.Tensor:
        if isinstance(self.data_list, list):
            return torchvision.io.decode_image(self.data_list[idx])
        elif isinstance(self.data_list, VideoReader):
            return self.data_list[idx].permute(2, 0, 1)
        elif self.cfg.yuv_size is not None:
            H, W = self.cfg.yuv_size
            frame_size = H * W * 3 // 2
            self.yuv_file.seek(idx * frame_size)
            img_bytes = self.yuv_file.read(frame_size)
            arr = np.frombuffer(img_bytes, dtype=np.uint8).reshape(H * 3 // 2, W)
            return yv12_to_rgb(arr, W, H)
        else:
            raise ValueError()

    def preprocess_image(self, image: torch.Tensor, frame: int):
        image = to_dtype(image, torch.float32, True)  # CHW in [0, 1] range
        if self.resize:
            image = torch.nn.functional.interpolate(
                image.unsqueeze(0),
                size=self.resize,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        original_image = image
        image = self.transform.transform(image, frame)
        if self.patch_size:
            image = patchify_images(
                image.unsqueeze(0), self.patch_size, self.patch_size
            ).squeeze(0)
            original_image = patchify_images(
                original_image.unsqueeze(0), self.patch_size, self.patch_size
            ).squeeze(0)
        else:
            image = image.view(image.shape[0], -1).t()
            original_image = original_image.view(original_image.shape[0], -1).t()
        return image, original_image

    def cache_frames(self, frame_idx: list[int]):
        original_images = []
        cached_feats = []
        for i in frame_idx:
            frame = self.load_sample(i)
            processed_frame, original_image = self.preprocess_image(frame, i)
            cached_feats.append(processed_frame.to(self.device))
            original_images.append(original_image.to(self.device))

        self.current_cache_idx = frame_idx
        self.cached_feats = torch.stack(cached_feats).to(self.device)
        self.original_images = torch.stack(original_images).to(self.device)

    def uncache_frames(self):
        del self.cached_feats
        self.cached_feats = None
        torch.cuda.empty_cache()

    def __len__(self):
        if isinstance(self.data_list, Path):
            if self.cfg.yuv_size is None:
                raise ValueError()
            H, W = self.cfg.yuv_size
            frame_size = H * W * 3 // 2
            frames = self.data_list.stat().st_size // frame_size
            return frames

        return len(self.data_list)

    def __getitem__(self, idx) -> torch.Tensor:
        if self.cached_feats is not None:
            image = self.cached_feats[idx]
        else:
            image, _ = self.preprocess_image(self.load_sample(idx), idx)
        return image

    def state(self):
        return {
            "input_data_shape": self.input_data_shape,
            "data_shape": self.data_shape,
            "num_frames": len(self),
            "transform_state": self.transform.state(),
        }

    def save_state(self, pth: Path):
        state = self.state()
        (pth / "dataset_state.json").write_text(str(state))

    def load_state(self, pth: Path):
        from ast import literal_eval

        state_str = (pth / "dataset_state.json").read_text()
        state = literal_eval(state_str)
        self.input_data_shape: list[int] = state["input_data_shape"]
        self.data_shape: list[int] = state["data_shape"]
        self.transform.inverse_map = state["transform_state"]
        num_frames = state["num_frames"]
        assert isinstance(num_frames, int)
        self.data_list = [""] * num_frames
