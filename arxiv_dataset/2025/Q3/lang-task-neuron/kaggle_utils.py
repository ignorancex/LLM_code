from kaggle.api.kaggle_api_extended import KaggleApi
import os
import json
import torch
from tqdm import tqdm
from typing import List, Optional, Union
import pandas as pd
from sklearn.metrics import average_precision_score
import shutil
import subprocess
from pathlib import Path
import requests


kaggle_api_token = {
    "username": "", #  kaggle api
    "key": ""
}

os.environ['KAGGLE_USERNAME'] = kaggle_api_token['username']
os.environ['KAGGLE_KEY'] = kaggle_api_token['key']

# Initialize Kaggle API
api = KaggleApi()
api.authenticate()

def download_from_kaggle(
        dataset_name:str,
        file_name:str,
        download_path: str = None

        ):
    """
    dataset_name: is username/dataset_name like 'inayarahmanisa/testssssss'
    """
    download_path = download_path if download_path else "data/"
    # api.dataset_download_file(dataset_name, file_name, path=download_path)
    # api.dataset_download_files(dataset_name, path=download_path if download_path else None, unzip=True)
    print(f"download_path: {download_path}")
    headers = {
        'User-Agent': 'Kaggle/1.5.12',
    }
    url = f"https://www.kaggle.com/api/v1/datasets/download/{dataset_name}/{file_name}"
    auth = (kaggle_api_token['username'], kaggle_api_token['key'])

    output_path = Path(download_path) / file_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, headers=headers, auth=auth, stream=True) as r:
        r.raise_for_status()  # Throw error if not 200
        with open(output_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                if chunk:  # Filter out keep-alive chunks
                    f.write(chunk)
    # print(f"✅ Downloaded: {file_name}")
    print("Dataset downloaded and unzipped.")



def save_to_kaggle(
    dataset_name: str,
    data_dir:str = None,
    result_neurons: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
    filename: Optional[Union[str, List[str]]] = None,
    is_update:bool =False,
    subdir_name:str = None
):
    
    os.environ['KAGGLE_USERNAME'] = kaggle_api_token['username']
    os.environ['KAGGLE_KEY'] = kaggle_api_token['key']

    api = KaggleApi()
    api.authenticate()
    subdir_name = subdir_name if subdir_name else ""
    target_dir = "kaggle/output"
    
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    os.makedirs(target_dir, exist_ok=True)

    base_dir = os.path.join(target_dir, "result")
    
    os.makedirs(base_dir, exist_ok=True)

    meta = {
        "id": f"{kaggle_api_token['username']}/{dataset_name}",
        "title": f"{dataset_name}",
        "isPrivate": True,
        "licenses": [{"name": "other"}]
    }

    metadata_path = os.path.join(base_dir, 'dataset-metadata.json')
    
    if is_update:
        download_dir = Path("kaggle/output/result").resolve()
        dataset_ref = f"{kaggle_api_token['username']}/{dataset_name}"
        api.dataset_download_files(dataset_ref, path=download_dir, unzip=True)
    with open(metadata_path, 'w') as f:
        json.dump(meta, f)
    
    sub_dir = os.path.join(base_dir, subdir_name)
    os.makedirs(base_dir, exist_ok=True)
    if data_dir:
        shutil.copytree(data_dir, sub_dir, dirs_exist_ok=True)
    elif result_neurons:
        if isinstance(result_neurons, torch.Tensor):
            assert isinstance(filename, str), "there must be the same number of filename and rensult neurons"
            torch.save(result_neurons, os.path.join(sub_dir, filename))
            
            # single tensor
        elif isinstance(result_neurons, list):
            assert len(result_neurons) == len(filename), "there must be the same number of filename and rensult neurons"
            for i in range (len(result_neurons)):
                torch.save(result_neurons[i], os.path.join(sub_dir, filename[i]))
                

    print("Files in the target directory:")
    for file_name in os.listdir(base_dir):
        print(f"- {file_name}")

    base_dir = Path("kaggle/output/result").resolve()
    save_dir = base_dir.as_posix() 
    print(f"save_dir: {save_dir}")
    try:
        dataset_ref = f"{kaggle_api_token['username']}/{dataset_name}"
        if is_update:
            print(f"Dataset {dataset_name} exists. Updating...")
            print(f"dataset_ref: {dataset_ref}")
            print("Files in directory before creating:")
            for f in os.listdir(base_dir):
                print("-", f)
            subprocess.run(["kaggle", "datasets", "version", "-p", base_dir,  "--dir-mode", "zip",  "-m", "Updating dataset"])

        else:
            print(f"Dataset {dataset_name} does not exist. Creating a new one...")
            print("Files in directory before creating:")
            for f in os.listdir(save_dir):
                print("-", f)
            result = subprocess.run(["kaggle", "datasets", "create", "-p", base_dir, "--dir-mode", "zip"],capture_output=True,text=True)
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            
    except Exception as e:
        print(f"Error while checking or creating dataset: {e}")

    shutil.rmtree(target_dir)


# b = torch.randn(2, 3, 116746)
# c = torch.randn(1,2,3,4)
# path_res = f"res"
# torch.save(b, f"{path_res}/test.pt")

# save_to_kaggle(
#     data_dir=path_res,
#     dataset_name= "testssssss",
#     is_update=True
# )
