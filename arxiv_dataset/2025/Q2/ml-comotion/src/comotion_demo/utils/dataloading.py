# Copyright (C) 2025 Apple Inc. All Rights Reserved.
import logging
from pathlib import Path
from typing import Generator, Tuple

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torchvision import transforms

IMG_MEAN = torch.tensor([0.4850, 0.4560, 0.4060]).view(-1, 1, 1)
IMG_STD = torch.tensor([0.2290, 0.2240, 0.2250]).view(-1, 1, 1)
INTERNAL_RESOLUTION = (512, 512)
VIDEO_EXTENSIONS = {".mp4"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def normalize_image(image: torch.Tensor) -> torch.Tensor:
    """Apply ImageNet normalization to an image tensor."""
    if not isinstance(image, torch.Tensor):
        raise ValueError("Expect input to be a torch.Tensor")
    return (image - IMG_MEAN) / IMG_STD


def unnormalize_image(image: torch.Tensor) -> torch.Tensor:
    """Undo ImageNet normalization to an image tensor."""
    if not isinstance(image, torch.Tensor):
        raise ValueError("Expect input to be a torch.Tensor")
    return image * IMG_STD + IMG_MEAN


def convert_image_to_tensor(image: NDArray[np.uint8]) -> torch.Tensor:
    """Convert an uint8 numpy array of shape HWC to a float tensor of shape CHW."""
    if not isinstance(image, np.ndarray):
        raise ValueError("Expect input to be a numpy array.")
    if image.dtype != np.uint8:
        raise ValueError("Expect input to be np.uint8 typed.")
    return torch.from_numpy(image).permute(2, 0, 1)


def convert_tensor_to_image(tensor: torch.Tensor) -> NDArray[np.uint8]:
    """Convert a float tensor of shape CHW to an uint8 numpy array of shape HWC."""
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Expect input to be a torch Tensor")
    return tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)


def crop_image_and_update_K(
    image: torch.Tensor,
    K: torch.Tensor,
    target_resolution: Tuple[int, int] = INTERNAL_RESOLUTION,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad and resize image to target resolution and update intrinsics."""
    target_height, target_width = target_resolution
    target_hw_ratio = target_height / target_width
    source_height, source_width = torch.tensor(image.shape[-2:])
    source_hw_ratio = source_height / source_width

    if source_hw_ratio >= target_hw_ratio:
        # pad needed along the width axis
        crop_height = source_height
        crop_width = int(source_height / target_hw_ratio)
    else:
        # pad needed along the height axis
        crop_height = int(source_width * target_hw_ratio)
        crop_width = source_width

    img_center_x = source_width / 2
    img_center_y = source_height / 2

    offset_x = int(img_center_x - crop_width / 2)
    offset_y = int(img_center_y - crop_height / 2)

    crop_args = [
        offset_y,
        offset_x,
        crop_height,
        crop_width,
        target_resolution,
        transforms.InterpolationMode.BILINEAR,
    ]
    # Pad, crop, and resize image
    cropped_image = transforms.functional.resized_crop(
        image, *crop_args, antialias=True
    )

    scale_y = target_height / crop_height
    scale_x = target_width / crop_width

    cropped_K = K.clone()
    cropped_K[..., 0, 0] *= scale_x
    cropped_K[..., 1, 1] *= scale_y
    cropped_K[..., 0, 2] = (cropped_K[..., 0, 2] - offset_x) * scale_x
    cropped_K[..., 1, 2] = (cropped_K[..., 1, 2] - offset_y) * scale_y

    return cropped_image, cropped_K


def yield_image_from_directory(
    directory: Path,
    start_frame: int,
    num_frames: int,
    frameskip: int = 1,
) -> Generator[NDArray[np.uint8], None, None]:
    """Generate the next frame from a directory."""
    if not directory.is_dir():
        raise ValueError(f"Path is not a directory: {directory}")
    image_files = sorted(
        [
            file
            for file in directory.glob("*")
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )
    if not image_files:
        raise ValueError(f"No images found in directory: {directory}")

    image_files = image_files[start_frame::frameskip]

    if len(image_files) > num_frames:
        image_files = image_files[:num_frames]

    for image_file in image_files:
        image = np.array(Image.open(image_file).convert("RGB"))
        yield image


def yield_image_from_video(
    filepath: Path,
    start_frame: int,
    num_frames: int,
    frameskip: int = 1,
) -> Generator[NDArray[np.uint8], None, None]:
    """Generate the next frame from a video."""
    if filepath.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError(f"Input file is not a video: {filepath}")

    video = cv2.VideoCapture(filepath.as_posix())

    if not video.isOpened():
        raise ValueError(f"Could not open video file: {filepath}")

    max_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    logging.info(
        f"Yielding {num_frames} frames from {filepath} ({max_frames} in total)."
    )

    if max_frames <= start_frame:
        logging.warning(f"Cannot start on frame {start_frame}.")
        video.release()
        return

    success = video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    if success:
        logging.info(f"Starting from frame {start_frame}")

    frame_limit = min(num_frames, max_frames - start_frame)
    frame_count = 0
    while frame_count < frame_limit:
        success, frame = video.read()
        if not success:
            break
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        yield image
        frame_count += 1

        # Skip intermediate frames if frameskip > 1
        for _ in range(1, frameskip):
            success, frame = video.read()
        if not success:
            break

    video.release()


def is_a_video(input_path: Path):
    return input_path.suffix in VIDEO_EXTENSIONS


def get_input_video_fps(input_path: Path) -> float:
    cap = cv2.VideoCapture(input_path.as_posix())
    if not cap.isOpened():
        raise RuntimeError(f"Failed to load video at {input_path}.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        raise RuntimeError("Failed to retrieve FPS.")

    return fps


def yield_image(
    input_path: Path,
    start_frame: int,
    num_frames: int,
    frameskip: int = 1,
) -> Generator[NDArray[np.uint8], None, None]:
    """Generate the next frame from either a video or a directory."""
    if is_a_video(input_path):
        yield from yield_image_from_video(
            input_path, start_frame, num_frames, frameskip
        )
    elif input_path.is_dir():
        yield from yield_image_from_directory(
            input_path, start_frame, num_frames, frameskip
        )
    else:
        raise ValueError("Input path must point to a video file or a directory")


def get_default_K(image: torch.Tensor) -> torch.Tensor:
    """Get a default approximate intrinsic matrix."""
    res = image.shape[-2:]
    max_res = max(res)
    K = torch.tensor([[2 * max_res, 0, 0.5 * res[1]], [0, 2 * max_res, 0.5 * res[0]]])
    return K


def yield_image_and_K(
    input_path: Path,
    start_frame: int,
    num_frames: int,
    frameskip: int = 1,
) -> Generator[Tuple[torch.Tensor, torch.Tensor], None, None]:
    """Generate image and the intrinsic matrix."""
    for image in yield_image(input_path, start_frame, num_frames, frameskip):
        image = convert_image_to_tensor(image)
        K = get_default_K(image)
        yield (image, K)


def prepare_network_inputs(
    image: torch.Tensor,
    K: torch.Tensor,
    device: torch.device | str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Image and intrinsics prep before inference.

    We crop and pad the input to a 512x512 image and update the intrinsics
    accordingly. The input is also expected to be ImageNet normalized.

    This demo code only supports processing individual samples. Some operations
    assume the existence of a batch dimension,  so we add a singleton batch
    dimension here. Other operations such as NMS and track management do not
    correctly handle batched inputs.

    Args:
    ----
        image: Input image (C x H x W) float (0-1) or uint8 (0-255) tensor
        K: Intrinsics matrix (2 x 3) float tensor
        device: Target device for inference

    """
    if image.dtype == torch.uint8:
        # Cast to float and normalize to 0-1
        image = image.float() / 255
    cropped_image, cropped_K = crop_image_and_update_K(image, K)
    cropped_image = normalize_image(cropped_image)

    # Add "batch" dimension and cast to target device
    cropped_image = cropped_image[None].to(device)
    cropped_K = cropped_K[None].to(device)

    return cropped_image, cropped_K
