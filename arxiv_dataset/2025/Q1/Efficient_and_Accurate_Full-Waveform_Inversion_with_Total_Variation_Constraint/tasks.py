import subprocess
import tarfile
from pathlib import Path
import requests

from invoke import task
from tqdm import tqdm

project_root_path = Path(__file__).parent.resolve()


@task
def download_salt_and_overthrust_models(c):
    # ref: https://wiki.seg.org/wiki/SEG/EAGE_Salt_and_Overthrust_Models
    url = "https://s3.amazonaws.com/open.source.geoscience/open_data/seg_eage_models_cd/salt_and_overthrust_models.tar.gz"
    tar_path = project_root_path.joinpath("datasets/salt_and_overthrust_models.tar.gz")
    save_dir_path = project_root_path.joinpath("datasets")

    # download data
    print(f"Downloading...")
    with requests.get(url, stream=True) as response:
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(tar_path, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    # unzip
    print(f"Extracting...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(path=save_dir_path)

    print(f"Completed")

@task
def format(c):
    format_commands = [
        f"isort ./src ./lib",
        f"black ./src ./lib"
    ]

    for cmd in format_commands:
        subprocess.run(cmd, shell=True, cwd=project_root_path)
