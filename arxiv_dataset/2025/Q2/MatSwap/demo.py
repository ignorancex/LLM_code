import argparse
from pathlib import Path
from PIL import Image

import cv2
import torch
import numpy as np
import torchvision.transforms as TF
import torchvision.transforms.functional as tf
from torchvision.utils import save_image

from pipeline import MatSwapPipeline


@torch.no_grad()
def main(args):
    # --------------------- Naming ---------------------
    outdir = Path('outputs')
    outdir.mkdir(exist_ok=True)
    fname = outdir / f'{args.image.stem}_{args.material.stem}.png'

    # ------------- Device and generator ---------------
    cuda_avail = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_avail else "cpu")
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    # -------------------- Pipeline --------------------
    pipe = MatSwapPipeline.from_pretrained('ilopes/matswap')
    pipe = pipe.to(device)

    # -------------------- Load data --------------------
    open_rgb_linear = lambda p: tf.to_tensor(Image.open(p).convert('RGB'))
    open_grayscale = lambda p: tf.rgb_to_grayscale(open_rgb_linear(p))
    open_bool = lambda p: (open_grayscale(p)>.5).bool()

    target = open_rgb_linear(args.image)
    mask = open_bool(args.mask)
    source = open_rgb_linear(args.material)

    # ----------------- Source control ------------------
    if args.scale:
        scale = 2**args.scale
        size = source.size(-1)
        crop = size//scale
        source = tf.center_crop(source, crop)

    if args.hue:
        hue_shift = args.hue
        frame = (source.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        frame_hsv[:, :, 0] = (frame_hsv[:, :, 0].astype(np.float32) + hue_shift) % 180
        frame_rgb = cv2.cvtColor(frame_hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        source = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0

    # -------------------- Resizing ----------------------
    original_size = target.shape[-2:]
    old_height, old_width = original_size
    aspect_ratio = old_height / old_width
    new_height = max(512, int(512 * aspect_ratio))
    new_width = max(512, int(512 / aspect_ratio))
    input_size = (new_height // 8) * 8, (new_width // 8) * 8
    downsample_bilinear = TF.Resize(
        size=input_size,
        interpolation=TF.InterpolationMode.BILINEAR,
        antialias=True
    )
    downsample_nearest = TF.Resize(
        size=input_size,
        interpolation=TF.InterpolationMode.NEAREST,
        antialias=None
    )

    target = downsample_bilinear(target)
    mask = downsample_nearest(mask)

    # --------------------- Load AOVs -----------------
    aov_dir = args.image.parent.parent / 'aov'
    aov_dir.mkdir(exist_ok=True)
    normal_path = aov_dir / f'{args.image.stem}_normal.png'
    irradiance_path = aov_dir / f'{args.image.stem}_irradiance.png'

    if not normal_path.exists() or not irradiance_path.exists():
        import aovs
        normals, irradiance = aovs.predict(target)
        irradiance = irradiance.mean(0,True)
        save_image(normals, normal_path)
        save_image(irradiance, irradiance_path)
    else:
        normals = open_rgb_linear(normal_path)
        irradiance = open_grayscale(irradiance_path)

    # -------------------- Perform inference ----------------------
    out = pipe(
        target_normals=normals[None]*2-1,
        target_irradiance=irradiance[None],
        target_image=target[None],
        target_mask=mask[None],
        source_image=source[None],
        guidance_scale=args.guidance_scale,
        output_type='pil',
        generator=generator,
        show_progress_bar=True,
        device=device,
    )[0]

    out.save(fname)


if "__main__" == __name__:
    parser = argparse.ArgumentParser(
        description="Demo inference code for MatSwap"
    )
    parser.add_argument(
        "--image", type=Path, required=True, help="Path to the target image in which the material will be inserted"
    )
    parser.add_argument(
        "--mask", type=Path, required=True, help="Path to the target mask image (binary mask) in which the material will be inserted"
    )
    parser.add_argument(
        "--material", type=Path, required=True, help="Path to the material to be inserted in the target image in (texture image)"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="random seed for reproducibility (default: 0)"
    )
    parser.add_argument(
        "--scale", type=int, help="the amount s by which to scale up the material appearance by 2^s (default: 0, so 2^0 = 1, no scaling)"
    )
    parser.add_argument(
        "--hue", type=int, help="material hue shift between 0 (no change) and 180 (opposite hue)"
    )
    parser.add_argument(
        "--guidance_scale", type=int, default=3.0, help="Classifier-free guidance scale (default: 3.0)"
    )

    args = parser.parse_args()
    main(args)