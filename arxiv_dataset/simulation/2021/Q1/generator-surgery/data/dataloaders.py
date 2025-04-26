import os
import pathlib

import numpy as np
import torch.utils.data


def get_dataloader(base_path, batch_size, n_train=-1, train=True):
    if train:
        path = os.path.join(base_path, 'train')
    else:
        path = os.path.join(base_path, 'test')

    if n_train > 0:
        dataset = torch.utils.data.Subset(FolderDataset(path),
                                          np.arange(n_train))
    else:
        dataset = FolderDataset(path)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=2,
                                             pin_memory=True)
    image_size = dataset[0].size(1)

    return dataloader, image_size


class FolderDataset(torch.utils.data.Dataset):
    """
    Simplified version of ImageFolder that does not require subfolders.
    - Used for loading images only (without any labels).
    - Assumes images are pre-processed into tensors and saved as '*.pt'
    for optimal loading speed.
    """
    def __init__(self, folder):
        self.folder = folder
        self.images = sorted(pathlib.Path(folder).rglob('*.pt'))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return torch.load(self.images[i])

    def __repr__(self):
        return self.__class__ + ":\n" + \
               f"Images folder: {self.folder}" + \
               f"Number of images: {self.__len__}"
