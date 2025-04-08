from shlex import split

import yaml
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

class DataModule(LightningDataModule):
    '''
    MODULO THE DATASET NECESARIO PARA EL ENTRENAMIENTO  CON PYTORCH LIGHTNING
    INDICO COMO SE LEVANTAN LAS IMAGENES Y GENERO LOS DATALODER TANTO PARA ENTRENAMIENTOS, VALIDACION Y TEST
    '''
    def __init__(self, DataFactory, splits, workers, batch_size, Dataset, val_batch = None, test_batch = None, ValDataset=None, TestDataset=None):
        '''
        INPUTS
        dataset: nombre del dataset utilizado
        workers: nmero de workers a utilizar
        num_workers: nmero de workers a utilizar
        '''
        super().__init__()
        self.DataFactory = DataFactory
        self._splits = splits
        self.num_workers = workers
        self.batch_size = batch_size
        self.val_batch = val_batch or batch_size
        self.test_batch = test_batch or batch_size
        self._dataset_config = Dataset
        self._val_dataset_config = ValDataset or {}
        self._test_dataset_config = TestDataset or {}
        
        self._valid_sample = None
        self.multiple_dataset_train=[]

    
    def prepare_data(self):
        '''Define steps that should be done on only one GPU, like getting data
        Usualmente se utiliza para el proceso de descagar el dataset'''
        return

    def setup(self, stage=None):
        '''
        Define steps that shouls be done in every GPU, like splitting data
        applying transforms, etc.
        Usually used to handle the task of loading the data'''
        
        with open(self._splits, 'r') as split_file:
            splits = yaml.load(split_file, Loader=yaml.FullLoader)
            path = splits['path']

        config = self._dataset_config
        val_config = {**config, **self._val_dataset_config}
        test_config = {**config, **self._test_dataset_config}

        self._train_data = self.DataFactory('training', path, splits['training']).getDataset(config)
        self._valid_data = self.DataFactory('validation', path, splits['validation']).getDataset(val_config)
        self._test_data = self.DataFactory('test', path, splits['test']).getDataset(test_config)
        # print("__init__setup_test",self._test_data.split,self._test_data.filenames, self._test_data.input_path)
        # print("__init__setup_validation",self._valid_data.split,self._valid_data.filenames, self._train_data.input_path)
        # print("__init__setup_traning",self._train_data.split,self._train_data.filenames, self._train_data.input_path)

    def train_dataloader(self):
        '''return Dataloader for Training data here'''
        return DataLoader(self._train_data, batch_size=self.batch_size, num_workers=self.num_workers)


    def val_dataloader(self):
        ''' return the DataLoader for Validation Data here'''
        return DataLoader(self._valid_data, batch_size=self.val_batch, num_workers=self.num_workers)

    def test_dataloader(self):
        '''Return DataLoader for Testing Data here'''
        return DataLoader(self._test_data, batch_size=self.test_batch, num_workers=self.num_workers)

    def val_sample(self, size):
        '''Return a sample from the validation set'''
        if self._valid_sample is None:
            self._valid_sample = self._valid_data.sample(size)
        return self._valid_sample
        
    def predict_dataloader(self):
        '''Return DataLoader for Testing Data here'''
        return DataLoader(self._test_data, batch_size=self.test_batch, num_workers=self.num_workers)
