import dotenv

dotenv.load_dotenv(override=True)

from types import SimpleNamespace

import click
import datasets
from PIL import Image


# https://github.com/Yuxiang-Lai117/Med-R1/blob/53e46ba24e04d7d7705db4750551786a93e96493/src/r1-v/local_scripts/prepare_hf_data.py#L144
def resize_image_to_qwen2_5_vl(example):
    # for Qwen2-VL-2B's processor requirement
    # Assuming the image is in a format that can be checked for dimensions
    # You might need to adjust this depending on how the image is stored in your dataset
    images = example["images"]
    for i in range(len(images)):
        image = images[i]
        if image.height < 28 or image.width < 28:
            image = resize_shortest_dim(image, target_size=28)

        elif image.height > 1024 or image.width > 1024:
            image = resize_longest_dim(image, target_size=1024)
        images[i] = image

    example["images"] = images
    return example


def resize_shortest_dim(image, target_size=28):
    """Resize the shortest dimension of the image to the specified size."""
    width, height = image.size
    if width < height:
        new_width = target_size
        new_height = int(height * (target_size / width))
    else:
        new_height = target_size
        new_width = int(width * (target_size / height))
    return image.resize((new_width, new_height), Image.BILINEAR)


# resize the longest dim to 384
def resize_longest_dim(image, target_size=1024):
    width, height = image.size
    if width > height:
        new_width = target_size
        new_height = int(height * (target_size / width))
    else:
        new_height = target_size
        new_width = int(width * (target_size / height))
    return image.resize((new_width, new_height), Image.BILINEAR)


@click.command()
@click.option("--dataset_path", type=str, default="med-vlrm/med-vlm-eval")
@click.option(
    "--dataset_subset", type=str, default=None, help="Subset of the dataset to use."
)
@click.option(
    "--dataset_split", type=str, default="test", help="Split of the dataset to use."
)
@click.option(
    "--hf_hub_repo", type=str, default="med-vlrm/med-vlm-eval-qwen2_5_vl_size"
)
def main(**kwargs):
    args = SimpleNamespace(**kwargs)
    print(f"Arguments: {args}")

    dataset_path = args.dataset_path
    dataset_subset = args.dataset_subset
    dataset_split = args.dataset_split
    hf_hub_repo = args.hf_hub_repo

    dataset = datasets.load_dataset(
        dataset_path,
        subset=dataset_subset,
        split=dataset_split,
    )

    # Resize images
    dataset = dataset.map(
        resize_image_to_qwen2_5_vl,
        num_proc=16,
        desc="Resizing images to Qwen2-VL-2B requirements",
        keep_in_memory=True,
    )
    dataset = datasets.DatasetDict(
        {
            dataset_split: dataset,
        }
    )

    # Push to Hugging Face Hub
    dataset.push_to_hub(hf_hub_repo)
    print(f"Dataset pushed to {hf_hub_repo} with {len(dataset)} items.")


if __name__ == "__main__":
    main()
