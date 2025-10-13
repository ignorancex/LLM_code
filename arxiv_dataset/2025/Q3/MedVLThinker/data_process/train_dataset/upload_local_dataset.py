import dotenv

dotenv.load_dotenv()

import click
import datasets


@click.command()
@click.option(
    "--local_data_dir",
    "-d",
    type=str,
    required=True,
    help="Path to the local dataset directory.",
)
@click.option(
    "--hf_repo",
    "-n",
    type=str,
    required=True,
    help="Name of the Hugging Face repository to upload the dataset to.",
)
def main(local_data_dir: str, hf_repo: str):
    """
    Upload a local dataset to a Hugging Face repository.

    Args:
        local_data_dir (str): Path to the local dataset directory.
        hf_repo (str): Name of the Hugging Face repository to upload the dataset to.
    """
    # Load the dataset from the local directory
    dataset = datasets.load_from_disk(local_data_dir)

    # Push the dataset to the specified Hugging Face repository
    dataset.push_to_hub(hf_repo)


if __name__ == "__main__":
    main()
