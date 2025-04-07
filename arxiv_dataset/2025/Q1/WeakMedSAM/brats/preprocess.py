import SimpleITK as sitk
from PIL import Image
import os
import argparse
import multiprocessing as mp
from tqdm import tqdm
import numpy as np


def preprocess(args) -> None:
    input_path, output_path = args
    os.makedirs(output_path)
    sample_name = input_path.split(os.sep)[-1]
    image = sitk.GetArrayFromImage(
        sitk.ReadImage(os.path.join(input_path, f"{sample_name}_flair.nii"))
    ).astype(np.float32)
    image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
    label = sitk.GetArrayFromImage(
        sitk.ReadImage(os.path.join(input_path, f"{sample_name}_seg.nii"))
    ).astype(np.uint8)
    label[label != 0] = 255
    for i, (img, lab) in enumerate(zip(image, label)):
        Image.fromarray(img, mode="L").save(
            os.path.join(output_path, f"img-{str(i).zfill(3)}.jpg"), quality=90
        )
        Image.fromarray(lab, mode="L").save(
            os.path.join(output_path, f"seg-{str(i).zfill(3)}.png")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocessing")
    parser.add_argument("--input-path", type=str, help="input path", required=True)
    parser.add_argument("--output-path", type=str, help="output path", required=True)
    parser.add_argument("--workers", type=int, help="preprocess workers", default=4)
    args = parser.parse_args()

    assert not os.path.exists(args.output_path), "output path exists."
    os.makedirs(args.output_path)

    input_paths = [
        os.path.join(args.input_path, tumor_type, sample_name)
        for tumor_type in ["HGG", "LGG"]
        for sample_name in os.listdir(os.path.join(args.input_path, tumor_type))
    ]

    output_paths = [
        os.path.join(args.output_path, sample_name)
        for tumor_type in ["HGG", "LGG"]
        for sample_name in os.listdir(os.path.join(args.input_path, tumor_type))
    ]

    with mp.Pool(args.workers) as pool:
        list(
            tqdm(
                pool.imap(preprocess, zip(input_paths, output_paths)),
                total=len(input_paths),
                ncols=100,
            )
        )
