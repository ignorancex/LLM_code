# *************************************************************************
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0 
#
# This file has been modified by ByteDance Ltd. and/or its affiliates.
#
# Original file was released under Aniportrait, with the full license text
# available at https://github.com/Zejun-Yang/AniPortrait/blob/main/LICENSE.
#
# This modified file is released under the same license.
# *************************************************************************

import argparse
import os
from datetime import datetime
import mediapipe as mp
import numpy as np
import cv2
import torch
from skimage import img_as_ubyte
from skimage.transform import resize
from scripts.vid2pose import extract_bbox_mp
from diffusers import DDIMScheduler, AutoencoderKLTemporalDecoder

from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms
from transformers import CLIPVisionModelWithProjection
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_3d import UNet3DConditionModel
from src.pipelines.pipeline_pose2vid_motenc_long import Pose2VideoPipeline
from src.utils.util import save_videos_grid
from decord import VideoReader
from scipy.ndimage import gaussian_filter1d

from src.models.motion_encoder.encoder import MotEncoder_withExtra as MotEncoder


def check_oob_new(bbox, frame_shape):
    left, top, right, bot = bbox
    left_oob = -left
    right_oob = right - frame_shape[1]
    top_oob = -top
    bot_oob = bot - frame_shape[0]

    return (
        left_oob,
        right_oob,
        top_oob,
        bot_oob,
        max(left_oob, right_oob, top_oob, bot_oob) > 0,
    )


def get_bbox_param(bbox, ref_bbox):
    left, top, right, bot = bbox
    center = np.array([(bot + top) * 0.5, (left + right) * 0.5])
    length = max(right - left, bot - top)

    ref_left, ref_top, ref_right, ref_bot = ref_bbox
    ref_center = np.array([(ref_bot + ref_top) * 0.5, (ref_left + ref_right) * 0.5])
    ref_length = max(ref_right - ref_left, ref_bot - ref_top)  # 最好保证bbox是正方形

    return np.asarray(
        ((center - ref_center) / ref_length).tolist() + [length / ref_length]
    )


def check_oob(bbox, size):
    left, top, right, bot = bbox
    return left < 0 or top < 0 or right > size[1] - 1 or bot > size[0] - 1


def get_bbox_from_center(center, length, size):
    center_X, center_Y = center
    w, h = length
    left, top, right, bot = [
        center_X - w / 2,
        center_Y - h / 2,
        center_X + w / 2,
        center_Y + h / 2,
    ]
    if check_oob((left, top, right, bot), size):
        x_offset = max(-left, 0) + min((size[1] - 1 - right), 0)
        y_offset = max(-top, 0) + min((size[0] - 1 - bot), 0)

        return np.array([left, top, right, bot]) + np.array(
            [x_offset, y_offset, x_offset, y_offset]
        )
    else:
        return np.array([left, top, right, bot])


def scale_bb(bbox, scale, size):
    left, top, right, bot = bbox[:4].tolist()
    width = right - left
    height = bot - top
    length = max(width, height) * scale
    center_X = (left + right) * 0.5
    center_Y = (top + bot) * 0.5
    left, top, right, bot = [
        center_X - length / 2,
        center_Y - length / 2,
        center_X + length / 2,
        center_Y + length / 2,
    ]
    if check_oob((left, top, right, bot), size):
        return bbox
    else:
        return np.array([left, top, right, bot])

def compute_aspect_preserved_bbox(bbox, increase_area):
    left, top, right, bot = bbox
    width = right - left
    height = bot - top
 
    width_increase = max(increase_area, ((1 + 2 * increase_area) * height - width) / (2 * width))
    height_increase = max(increase_area, ((1 + 2 * increase_area) * width - height) / (2 * height))

    left = int(left - width_increase * width)
    top = int(top - height_increase * height)
    right = int(right + width_increase * width)
    bot = int(bot + height_increase * height)

    return (left, top, right, bot)

def compute_increased_bbox(bbox, increase_area):
    left, top, right, bot = bbox
    width = right - left
    height = bot - top
 
    left = int(left - increase_area * width)
    top = int(top - increase_area * height)
    right = int(right + increase_area * width)
    bot = int(bot + increase_area * height)

    return (left, top, right, bot)

def get_bbox(tube_bbox, frame_shape, aspect_preserving, increase_area):
    if aspect_preserving:
        left, top, right, bot = compute_aspect_preserved_bbox(tube_bbox, increase_area)
    else:
        left, top, right, bot = compute_increased_bbox(tube_bbox, increase_area)

    left_oob, right_oob, top_oob, bot_oob, oob_flag = check_oob_new(
        [left, top, right, bot], frame_shape
    )
    # print(left_oob, right_oob, top_oob, bot_oob)
    if oob_flag:
        if left_oob > 0 and right_oob + left_oob <= 0:
            oob_flag = check_oob_new([left + left_oob, top, right + left_oob, bot], frame_shape)[-1]
            if not oob_flag:
                left, top, right, bot = left + left_oob, top, right + left_oob, bot
        elif right_oob > 0 and right_oob + left_oob <= 0:
            oob_flag = check_oob_new([left - right_oob, top, right - right_oob, bot], frame_shape)[-1]
            if not oob_flag:
                left, top, right, bot = left - right_oob, top, right - right_oob, bot
        if top_oob > 0 and top_oob + bot_oob <= 0:
            oob_flag = check_oob_new(
                [left, top + top_oob, right, bot + top_oob], frame_shape
            )[-1]
            if not oob_flag:
                left, top, right, bot = left, top + top_oob, right, bot + top_oob
        elif bot_oob > 0 and top_oob + bot_oob <= 0:
            oob_flag = check_oob_new(
                [left, top - bot_oob, right, bot - bot_oob], frame_shape
            )[-1]
            if not oob_flag:
                left, top, right, bot = left, top - bot_oob, right, bot - bot_oob

    return oob_flag, left, top, right, bot

def crop_bbox_from_frames(frame_list, tube_bbox, min_frames=16, image_shape=(512, 512), min_size=200,
                          increase_area=0.1, aspect_preserving=True, enable_oob=False, oob_padding='constant'):
    frame_shape = frame_list[0].shape
    # Filter short sequences
    if len(frame_list) < min_frames:
        print('short seq', len(frame_list))
        return [False, None, None]
    left, top, right, bot = tube_bbox
    width = right - left
    height = bot - top 
    # Filter if it is too small
    if max(width, height) < min_size:
        print('small seq', '%d<%d'%(max(width, height), min_size))
        return [False, None, None]

    oob_flag, left, top, right, bot = get_bbox(tube_bbox, frame_shape, aspect_preserving, increase_area)

    if oob_flag:
        if not enable_oob:
            return [oob_flag, None, None]
        else:
            if oob_padding == 'none':
                img_size = min(frame_shape[0], frame_shape[1])
                center_y = min(max(img_size / 2, (top + bot) / 2), frame_shape[0] - img_size / 2)
                center_x = min(max(img_size / 2, (left + right) / 2), frame_shape[1] - img_size / 2)
                top, bot = int(center_y - img_size / 2), int(center_y + img_size / 2)
                left, right = int(center_x - img_size / 2), int(center_x + img_size / 2)
            else:
                # padding
                left_oob, right_oob, top_oob, bot_oob, oob_flag = check_oob_new(
                    [left, top, right, bot], frame_shape
                )
                padding = max(left_oob, right_oob, top_oob, bot_oob)
                if oob_padding == 'constant':
                    frame_list = [np.pad(frame, ((padding, padding), (padding, padding), (0, 0)), 'constant', constant_values=0) for frame in frame_list]
                    left, top, right, bot = left + padding, top + padding, right + padding, bot + padding

                else:
                    raise NotImplementedError
            # top, left, bot, right = 0, 0, frame_shape[0], frame_shape[1]
    selected = [frame[top:bot, left:right] for frame in frame_list]   # tmp zxc

    if image_shape is not None:
        out = [img_as_ubyte(resize(frame, image_shape, anti_aliasing=True)) for frame in selected]
    else:
        out = selected

    return oob_flag, out, [left, top, right, bot]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='./configs/prompts/animation.yaml')
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("-L", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=3.5)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--fps", type=int)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    return args


def main(args):
    device = args.device
    print('device', device)
    config = OmegaConf.load(args.config)

    if config.weight_dtype == "fp16":
        weight_dtype = torch.float16
    else:
        weight_dtype = torch.float32

    vae = AutoencoderKLTemporalDecoder.from_pretrained(
        config.vae_path,
    ).to(device, dtype=weight_dtype)

    infer_config = OmegaConf.load(config.inference_config)
    reference_unet = UNet2DConditionModel.from_pretrained(
        config.pretrained_base_model_path,
        subfolder="unet",
    ).to(device=device, dtype=weight_dtype)
    denoising_unet = UNet3DConditionModel.from_pretrained_2d(
        config.pretrained_base_model_path,
        "",
        subfolder="unet",
        unet_additional_kwargs=infer_config.unet_additional_kwargs,
    ).to(dtype=weight_dtype, device=device)

    motion_encoder = MotEncoder().to(dtype=weight_dtype, device=device)
    motion_encoder.eval()

    image_enc = CLIPVisionModelWithProjection.from_pretrained(
        config.image_encoder_path
    ).to(dtype=weight_dtype, device=device)

    sched_kwargs = OmegaConf.to_container(
        OmegaConf.load(config.inference_config).noise_scheduler_kwargs
    )

    scheduler = DDIMScheduler(**sched_kwargs)

    generator = torch.manual_seed(args.seed)

    width, height = args.W, args.H

    # load pretrained weights
    denoising_unet.load_state_dict(
        torch.load(config.denoising_unet_path, map_location="cpu"), strict=False
    )
    reference_unet.load_state_dict(
        torch.load(
            config.denoising_unet_path.replace('denoising_unet', 'reference_unet'),
            map_location="cpu",
        ),
        strict=True,
    )
    motion_encoder.load_state_dict(
        torch.load(
            config.denoising_unet_path.replace('denoising_unet', 'motion_encoder'),
            map_location="cpu",
        ),
        strict=True,
    )
    denoising_unet.load_state_dict(
        torch.load(
            config.temporal_module_path,
            map_location="cpu",
        ),
        strict=False,
    )

    pipe = Pose2VideoPipeline(
        vae=vae,
        image_encoder=image_enc,
        reference_unet=reference_unet,
        denoising_unet=denoising_unet,
        motion_encoder=motion_encoder,
        scheduler=scheduler,
    )
    pipe = pipe.to(device, dtype=weight_dtype)

    date_str = datetime.now().strftime("%Y%m%d")
    if args.name is None:
        time_str = datetime.now().strftime("%H%M")
        save_dir_name = f"{date_str}--{time_str}"
    else:
        save_dir_name = f"{date_str}--{args.name}"
    save_vid_dir = os.path.join('output', save_dir_name, 'concat_vid')
    os.makedirs(save_vid_dir, exist_ok=True)
    save_split_vid_dir = os.path.join('output', save_dir_name, 'split_vid')
    os.makedirs(save_split_vid_dir, exist_ok=True)

    pose_transform = transforms.Compose(
        [transforms.Resize((height, width)), transforms.ToTensor()]
    )
    options = mp.tasks.vision.FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path='blaze_face_short_range.tflite'
        ),
    )
    img_detector = mp.tasks.vision.FaceDetector.create_from_options(options)
    clip_length = 500
    args.test_cases = OmegaConf.load(args.config)["test_cases"]
    for ref_image_path in list(args.test_cases.keys()):
        # Each ref_image may correspond to multiple actions
        for pose_video_path in args.test_cases[ref_image_path]:
            video_name = os.path.basename(pose_video_path).split(".")[0]
            source_name = os.path.basename(ref_image_path).split(".")[0]

            vid_name = f"{source_name}_drivenby_{video_name}.mp4"
            save_vid_path = os.path.join(save_vid_dir, vid_name)
            print(save_vid_path)
            if os.path.exists(save_vid_path):
                continue

            if ref_image_path.endswith('.mp4'):
                src_vid = VideoReader(ref_image_path)
                ref_img = src_vid[0].asnumpy()
            else:
                ref_img = cv2.cvtColor(cv2.imread(ref_image_path), cv2.COLOR_BGR2RGB)

            control = VideoReader(pose_video_path)
            fps = control.get_avg_fps()
            sel_idx = range(len(control))[:clip_length]
            control = control.get_batch([sel_idx]).asnumpy()

            ref_pose_image = ref_img.copy()
            ref_bbox = extract_bbox_mp(ref_img, None, img_detector)
            left, top, right, bot = scale_bb(ref_bbox, scale=1.1, size=ref_img.shape[:2])
            ref_pose_image = ref_pose_image[int(top) : int(bot), int(left) : int(right)]
            ref_pose_pil = Image.fromarray(ref_pose_image).convert("RGB")

            ref_img = crop_bbox_from_frames(
                [ref_img],
                ref_bbox,
                min_frames=0,
                min_size=0,
                increase_area=0.5,
                enable_oob=True,
            )[1][0]
            ref_image_pil = Image.fromarray(ref_img).convert("RGB")

            size = args.H
            generator = torch.Generator(device=device)
            generator.manual_seed(torch.initial_seed())

            init_latents = (
                torch.randn(
                    (1, 4, 1, height // 8, width // 8),
                    generator=generator,
                    device=pipe._execution_device,
                    dtype=image_enc.dtype,
                )
                .to(device)
                .repeat(1, 1, control.shape[0], 1, 1)
            )
            init_latents = None

            options = mp.tasks.vision.FaceDetectorOptions(
                base_options=mp.tasks.BaseOptions(
                    model_asset_path='blaze_face_short_range.tflite'
                ),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
            )
            detector = mp.tasks.vision.FaceDetector.create_from_options(options)
            timestamp_ms = 0
            ani_bbox_center = []
            fix_length = None
            bbox_params = []
            for idx_control in range(control.shape[0]):
                pose_image = control[idx_control][:, : control[0].shape[0]]

                timestamp_ms += int(1000 / fps)
                bbox = extract_bbox_mp(pose_image, None, detector, timestamp_ms)

                bbox_param = torch.from_numpy(get_bbox_param(bbox, ref_bbox))
                bbox_param = torch.ones_like(bbox_param).to(dtype=bbox_param.dtype)  #########
                bbox_param[:2] *= 0  #########
                bbox_params.append(bbox_param)

                if idx_control == 0:
                    bbox = scale_bb(bbox, scale=1.1, size=pose_image.shape[:2])
                    fix_length = (
                        round(bbox[2] - bbox[0]) // 2 * 2,
                        round(bbox[3] - bbox[1]) // 2 * 2,
                    )
                left, top, right, bot = bbox
                center_X = (left + right) * 0.5
                center_Y = (top + bot) * 0.5
                ani_bbox_center.append([center_X, center_Y])

            ani_bbox_center = np.asarray(ani_bbox_center, dtype=np.float32)
            ani_bbox_center = gaussian_filter1d(
                ani_bbox_center, sigma=5.0, axis=0
            ).tolist()
            bbox_params = torch.stack(bbox_params, dim=0)

            pose_images = []
            ori_pose_images = []
            for idx_control in range(control.shape[0]):
                # for idx_control in [0, 164]:
                # if not idx_control % 3 == 0: continue
                pose_image = control[idx_control][
                    :, : control[0].shape[0]
                ]
                pose_image_pil = Image.fromarray(pose_image).convert("RGB")
                ori_pose_image_pil = None
                ori_pose_image_pil = Image.fromarray(pose_image.copy()).convert(
                    "RGB"
                )
                bbox = get_bbox_from_center(
                    ani_bbox_center[idx_control], fix_length, pose_image.shape[:2]
                )

                left, top, right, bot = bbox
                pose_image = pose_image[int(top) : int(bot), int(left) : int(right)]
                pose_image_pil = Image.fromarray(pose_image).convert("RGB")

                pose_image_pil.resize((size, size))
                pose_images.append(pose_image_pil)
                ori_pose_images.append(ori_pose_image_pil)

            pose_tensor_list = []
            ori_pose_tensor_list = []
            ref_tensor_list = []

            for idx, pose_image_pil in enumerate(pose_images):
                pose_tensor_list.append(pose_transform(pose_image_pil))
                ori_pose_tensor_list.append(pose_transform(ori_pose_images[idx]))
                ref_tensor_list.append(pose_transform(ref_image_pil))

            ref_tensor = torch.stack(ref_tensor_list, dim=0)  # (f, c, h, w)
            ref_tensor = ref_tensor.transpose(0, 1).unsqueeze(0)  # (c, f, h, w)

            pose_tensor = torch.stack(pose_tensor_list, dim=0)  # (f, c, h, w)
            pose_tensor = pose_tensor.transpose(0, 1).unsqueeze(0)

            ori_pose_tensor = torch.stack(ori_pose_tensor_list, dim=0)  # (f, c, h, w)
            ori_pose_tensor = ori_pose_tensor.transpose(0, 1).unsqueeze(0)

            gen_video = pipe(
                ref_image_pil,
                pose_images,
                ref_pose_pil,
                width,
                height,
                len(pose_images),
                num_inference_steps=35,
                guidance_scale=2.5,
                generator=generator,
                init_latents=init_latents,
                mot_bbox_param=bbox_params,
                context_frames=24,
                context_overlap=4,
            ).videos

            # Concat it with pose tensor
            video = torch.cat([ref_tensor, pose_tensor, ori_pose_tensor, gen_video], dim=0)

            save_videos_grid(
                video,
                save_vid_path,
                n_rows=4,
                fps=25,
            )

            if True:
                save_vid_path = save_vid_path.replace(save_vid_dir, save_split_vid_dir)
                save_videos_grid(gen_video, save_vid_path, n_rows=1, fps=25, crf=18)

                save_videos_grid(
                    ori_pose_tensor,
                    save_vid_path.replace('.mp4', '_driving.mp4'),
                    n_rows=1,
                    fps=25,
                    crf=18,
                )

                refimage = (
                    ref_tensor[0, :, 0].permute(1, 2, 0).cpu().numpy()
                )  # (3, h, w)
                refimage = Image.fromarray((refimage * 255).astype(np.uint8))
                refimage.save(save_vid_path.replace('.mp4', '_src.png'))


if __name__ == "__main__":
    args = parse_args()
    main(args)
