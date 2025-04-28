import torchvision
import torch
from PIL import Image

class MyMNIST(torchvision.datasets.MNIST):

    def __init__(self, rank=0, world_size=1, data_hete = False, **kwargs):
        """
        Args:
            rank: specify current process
            world_size: number of nodes
            data_hete: `= True`, then we sort the datat according to label, to generate heterogenous
        """
        super(MyMNIST, self).__init__(**kwargs)

        """
        generate the data heterogenous
        """
        self.data_hete = data_hete
        if self.data_hete:
            sorted_indices = sorted(range(len(self.targets)), key=lambda i:self.targets[i])
            self.data = self.data[sorted_indices]
            self.targets = self.targets[sorted_indices]

        """
        split data to the i-th piece, where i is the `rank` and there are `world_size` pieces
        """
        self.rank = rank
        self.world_size = world_size

        dataset_size = len(self.data)
        indices = list(range(dataset_size))
        split = dataset_size//self.world_size

        # calculate the start and end indices
        start_idx = self.rank*split
        end_idx = min(start_idx + split, dataset_size)

        if self.rank == self.world_size - 1:
            end_idx = dataset_size # ensure the last partition include the remainder

        self.data = self.data[start_idx:end_idx]
        self.targets = self.targets[start_idx:end_idx]


    def __getitem__(self, index):
        """
        Override the original method of the MNIST class.
        Args:
            index (int): the i-th sample
        Returns:
            tuple: (feature, target, index)
            target: one-hot label
            feature: the input value
        """  

        feature = self.data[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        feature = Image.fromarray(feature.numpy(), mode='L')

        if self.transform is not None:
            feature = self.transform(feature)

        return feature, self.targets[index], index