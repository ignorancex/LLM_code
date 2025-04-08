from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset

from . import Datasets, Filter
from . import Datasets
from .DataModule import DataModule


class DataFactory(object):
    def __init__(self, split=None, input_path=None, filenames=None, output_path=None, **args):
        super().__init__()
        print('...................................datafactory.init')
        self._split = split
        self._input_path = input_path
        self._output_path = output_path
        self._filenames = filenames
        
    def getDataset(self, config) -> Dataset:
        print('...............................................getfdatasetr',config.get('Type', 'Dataset'))
        cls = getattr(Datasets,config.get('Type', 'Dataset'))
        return cls(split=self._split, input_path=self._input_path, filenames=self._filenames, dataFactory=self, **config)

    def getFilter(self, config) -> Filter.Filter:
        print('...............................................getfilter',config['Type'])
        cls = getattr(Filter,config['Type'])
        return cls(dataFactory=self, input_path=self._input_path, output_path=self._output_path, **config)

    def getPipe(self, filters = None) -> Filter.Filter:
        print('...................................getpipe.datafactory')
        pipe = []
        for filter in filters or []:
            pipe.append(self.getFilter(filter))
        return pipe

def get(config) -> LightningDataModule:
    module = DataModule(DataFactory, **config)
    return module