from .Dataset import Dataset

class MultiDataset(Dataset):
    def __init__(self, datasets, **args):
        super().__init__(**args)
        self._datasets = [
            self.dataFactory.getDataset(dataset)
            for dataset in datasets
        ]

    def get(self, element, test=None):
        test = test or self.split == 'test'
        element = tuple(dataset.get(element, test) for dataset in self._datasets)
        for filter in self._pipe:
            print('GEET',element)
            element = filter(element, test)
        return element
