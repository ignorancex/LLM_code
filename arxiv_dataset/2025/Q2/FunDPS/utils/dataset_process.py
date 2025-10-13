import argparse
import json
import os

import numpy as np
from datasets import Array3D, Dataset, Features, Value

from dataset_prop import STATS, SUPPORTED_DATASETS, get_dataset_info


def data_generator(dataset_name: str, training: bool):
    """Generator function to yield samples one at a time."""
    stats = STATS[dataset_name]
    means = stats["mean"]
    stds = stats["std"]

    # Get dataset info
    info = get_dataset_info(dataset_name, training)
    loader = info["loader"]
    path_pattern = info["path_pattern"]
    file_range = info["range"]

    sample_idx = 0  # Global counter for all samples


    # Process each file
    for _i, j in enumerate(file_range):
        file_path = path_pattern.format(j)

        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue

        # Load raw data
        try:
            a, u = loader(file_path)
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            continue

        # Convert to float64 if needed
        if _i == 0 and (a.dtype != np.float64 or u.dtype != np.float64):
            print(f"Warning: Converting dtypes ({a.dtype}, {u.dtype}) to float64")
            a = a.astype(np.float64)
            u = u.astype(np.float64)

        # Normalization
        a_normalized = (a - means[0]) * (0.5 / stds[0])
        u_normalized = (u - means[1]) * (0.5 / stds[1])

        # Stack channels for all samples at once
        combined = np.stack((a_normalized, u_normalized), axis=1)
        assert combined.shape[-3:] == (2, 128, 128), f"Unexpected shape: {combined.shape}"

        # Process each sample in the file
        batch_size = a.shape[0]
        for i in range(batch_size):
            # Yield one sample at a time
            yield {"id": sample_idx, "data": combined[i]}
            sample_idx += 1

        # Clean up memory
        del a, u, a_normalized, u_normalized


def process_dataset(dataset_name: str, training=True) -> Dataset:
    """Process dataset and save in HuggingFace datasets format."""

    # Create output directory
    output_dir = f"data/{dataset_name if training else dataset_name + '_test'}_hf"
    os.makedirs(output_dir, exist_ok=True)

    # Define features
    features = Features(
        {
            "id": Value("int32"),
            "data": Array3D(shape=(2, 128, 128), dtype="float64"),
        }
    )

    # Create dataset using generator
    dataset = Dataset.from_generator(generator=lambda: data_generator(dataset_name, training), features=features)
    dataset.save_to_disk(output_dir)

    # Save metadata
    num_samples = len(dataset)
    metadata = {
        "name": dataset_name,
        "stats": STATS[dataset_name],
        "shape": (2, 128, 128),
        "num_samples": num_samples,
        "__version__": STATS["__version__"],
    }

    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved {num_samples} samples to {output_dir}")
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and store PDE datasets in HuggingFace format.")
    parser.add_argument("dataset", type=str, help=f"Name of the dataset ({', '.join(SUPPORTED_DATASETS)}) or 'all' to process all datasets")
    parser.add_argument("--test", "-t", action="store_true", help="Process the test dataset")
    args = parser.parse_args()

    if args.dataset.lower() == "all":
        for dataset in SUPPORTED_DATASETS:
            print(f"\nProcessing {dataset} dataset...")
            process_dataset(dataset, not args.test)
            print(f"Finished processing {dataset} dataset.")
    else:
        if args.dataset not in SUPPORTED_DATASETS:
            raise ValueError(f"Dataset {args.dataset} not supported. Supported datasets are: {', '.join(SUPPORTED_DATASETS)}")
        process_dataset(args.dataset, not args.test)

    print("Finished processing all files.")
