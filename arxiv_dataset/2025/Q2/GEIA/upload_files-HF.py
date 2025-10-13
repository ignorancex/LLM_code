import argparse
import os

from huggingface_hub import HfApi, login


def upload_output_data(base_path, repo_id, token):
    # Initialize HF API with the token
    api = HfApi()
    login(token=token, write_permission=True)

    # Create the dataset repository if it doesn't exist
    try:
        api.create_repo(
            repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True
        )
    except Exception as e:
        print(f"Repository initialization warning (can be ignored if repo exists): {e}")

    # Upload the entire folder at once
    try:
        # Get the basename of the path (the last folder name)
        folder_name = os.path.basename(base_path)
        api.upload_folder(
            folder_path=base_path,
            path_in_repo=folder_name,  # Use the folder name as the path in repo
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Uploading log folder GPT2",
        )
        print(f"Successfully uploaded folder {base_path}.")
    except Exception as e:
        print(f"Error uploading folder: {e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload tracking center data to Hugging Face dataset repository."
    )

    parser.add_argument(
        "--base_path",
        type=str,
        required=True,
        help="Path to the outputs_3d directory containing the tracking data.",
    )

    parser.add_argument(
        "--username",
        default="antragoudaras",
        type=str,
        help="Your Hugging Face username.",
    )

    parser.add_argument(
        "--repo_name",
        default="geia-output",
        type=str,
        help="Name of your dataset repository.",
    )

    parser.add_argument(
        "--token",
        type=str,
        required=True,
        help="Hugging Face API token",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    repo_id = f"{args.username}/{args.repo_name}"
    upload_output_data(base_path=args.base_path, repo_id=repo_id, token=args.token)

# example ussage:
# python src/filming/upload_tracking_centers_to_hf.py --base_path [tracking_centers] --username antragoudaras --repo_name GEIA_output --token $HF_TOKEN