import os
import numpy
import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision.transforms import ToTensor
from PIL import Image
import csv


'''
This file contains functions for loading (and if necessary downloading) the
relevant datasets.
'''

# Path to root folder for datasets
from workspace_path import home_path
root_folder = home_path / 'datasets'


def dataset_collector(dataset, split, **kwargs):
    '''
    Wrapper for all collectors, will use the datasets dict to see if a given
    dataset and split is available and return it using its specified collector
    Args:
        dataset (str): Key for the dataset in the datasets dict
        split (str): An available split for the given dataset
        **kwargs (dict): Any additional parameters
    Returns (torch.utils.data.Dataset)
    '''
    if dataset not in datasets:
        available_datasets = ', '.join(datasets.keys())
        raise ValueError(f'Unexpected value of dataset: {dataset}. '
                         f'Available datasets are {available_datasets}')
    dataset_info = datasets[dataset]
    if split not in dataset_info['split']:
        available_splits = ', '.join(dataset_info['split'])
        raise ValueError(
            f'Unexpected value of split: {split}. '
            f'Available splits for this dataset are {available_splits}')
    kwargs['dataset'] = dataset
    kwargs['split'] = split
    return dataset_info['source'](**kwargs)


def bapps_collector(split='train', subsplit='all', **kwargs):
    '''
    Function for collecting and, if necessary, downloading the BAPPS dataset.

    Args:
        split (str): Which split to collect ('train', 'val', 'jnd/val')
        subsplit (str): Which subsplit to collect ('all' collects all)
        **kwargs (dict): Any additional parameters for collection
    Returns:
        torch.utils.data.Dataset: The given split of the BAPPS dataset
    '''
    (root_folder/'BAPPS').mkdir(parents=True, exist_ok=True)
    if not (
        (root_folder/'BAPPS/train').exists() and
        (root_folder/'BAPPS/val').exists() and
        (root_folder/'BAPPS/jnd/val').exists()
    ):
        if 'download' in kwargs and not kwargs['download']:
            raise FileNotFoundError(
                'Files are missing. Set \'download\' to True to automatically '
                'download them')
        print('Downloading BAPPS dataset...')
        path = root_folder/'BAPPS'
        kaggle_download(
            'chaitanyakohli678/berkeley-adobe-perceptual-patch-similarity-bapps',
            path, unzip=True
        )
        (path/'dataset/2afc/train').rename(path/'train')
        (path/'dataset/2afc/val').rename(path/'val')
        (path/'dataset/jnd').rename(path/'jnd')
        (path/'dataset').rmdir()
    subsplits = os.listdir(root_folder/f'BAPPS/{split}')
    if subsplit == 'all':
        ret = []
        for subsplit in subsplits:
            dirs = os.listdir(root_folder/f'BAPPS/{split}/{subsplit}')
            paths = [root_folder/f'BAPPS/{split}/{subsplit}/{d}' for d in dirs]
            ret.append(MultipleFolderDataset(
                *paths, name=subsplit,
                image_transform=kwargs.get('image_transform')
            ))
        return ConcatDataset(ret)
    elif subsplit in subsplits:
        dirs = os.listdir(root_folder/f'BAPPS/{split}/{subsplit}')
        paths = [root_folder/f'BAPPS/{split}/{subsplit}/{d}' for d in dirs]
        return MultipleFolderDataset(
            *paths, name=subsplit,
            image_transform=kwargs.get('image_transform')
        )
    else:
        raise ValueError(
            f'Unexpected value of subsplit: {subsplit}. '
            f'Expected any of: all, {", ".join(subsplits+["all"])}'
        )

# Dictionary of available datasets and their attributes and parameters
datasets = {
    'BAPPS': {
        'full_name': 'Berkeley Adobe Perceptual Patch Similarity',
        'source': bapps_collector,
        'downloadable': True,
        'split': ['train', 'val', 'jnd/val'],
        'output_format': None  # TODO: annotate the format (eg 'onehot')
    }
}


def _kaggle_init():
    '''
    Prepares and returns a Kaggle API using keys in api_keys.csv

    Returns:
        KaggleApi: A prepared and authenticated Kaggle API
    '''
    keyfile = home_path/'api_keys.csv'
    if not keyfile.is_file():
        raise RuntimeError(
            f'Missing file {keyfile}. Create it as an empty file and rerun.'
        )
    kaggle_username = None
    kaggle_key = None
    with open(home_path/'api_keys.csv') as keyfile:
        key_reader = csv.reader(keyfile)
        for key, value in key_reader:
            if key == 'kaggle_username':
                kaggle_username = value
            elif key == 'kaggle_key':
                kaggle_key = value  
    if kaggle_username is None or kaggle_key is None:
        raise RuntimeError(
            f'Kaggle keys missing. Log in to '
            f'https://www.kaggle.com/<username>/account and download an API '
            f'token via "Create API Token" and paste the username and key '
            f'into {keyfile} as "kaggle_username,<username>" and '
            f'"kaggle_key,<key>"')
    os.environ['KAGGLE_USERNAME'] = kaggle_username
    os.environ['KAGGLE_KEY'] = kaggle_key
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


def kaggle_download(dataset, path, **kwargs):
    '''
    Downloads a given dataset from kaggle to a given folder.

    Args:
        dataset (str): Dataset to download
        path (str): Path to folder of dataset
        **kwargs (dict): Any additional parameters for downloading
    '''
    api = _kaggle_init()
    api.dataset_download_files(dataset, path, **kwargs)

class MultipleFolderDataset(Dataset):
    '''
    A dataset for loading data where data is contained in folders where each
    matching group of data has the same name (with possibly different
    file-endings)
    Allowed file types: .png, .tif, .tiff, .jpg, .jpeg, .bmp, .npy 
    Args:
        *args (str): Paths to the folders to extract from
        name (str): A name to be returned together with each datapoint
        image_transform (nn.Module): Transform applied when getting images
    '''
    def __init__(self, *args, name=None, image_transform=None):
        super().__init__()
        if len(args) < 1:
            raise RuntimeError('Must be given at least one path')
        self.name = name
        acceptable_endings = [
            'png', 'tif', 'tiff', 'jpg', 'jpeg', 'bmp', 'npy'
        ]
        folder_files = []
        for folder in args:
            files = os.listdir(folder)
            folder_files.append({
                f[:f.index('.')]: f[f.index('.') + 1:]
                for f in files if f[f.index('.') + 1:] in acceptable_endings
            })
        self.data_paths = []
        for filename, ending in folder_files[0].items():
            paths = [f'{args[0]}/{filename}.{ending}']
            for folder, arg in zip(folder_files[1:], args[1:]):
                if filename in folder:
                    paths.append(f'{arg}/{filename}.{folder[filename]}')
                else:
                    break
            if len(paths) != len(args):
                continue
            self.data_paths.append(paths)
        self.image_transform = image_transform
        if self.image_transform is None:
            self.image_transform = ToTensor()


    def __getitem__(self, index):
        image_endings = ['png', 'tif', 'tiff', 'jpg', 'jpeg', 'bmp']
        npy_endings = ['npy']

        ret = {}
        for path in self.data_paths[index]:
            ending = path[path.index('.') + 1:]
            folder = path.split('/')[-2]
            if ending in image_endings:
                image = Image.open(path).convert(mode='RGB')
                ret[folder] = self.image_transform(image)
            elif ending in npy_endings:
                ret[folder] = torch.from_numpy(numpy.load(path))
            else:
                raise RuntimeError('Loading from unsupported file type')
        if not self.name is None:
            ret['name'] = self.name
        return ret

    def __len__(self):
        return len(self.data_paths)