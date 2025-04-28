import math
from typing import Dict, Generator, Optional, Tuple, Union
from enum import Enum

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from loader.augmentation import AugmentData
from loader.data_loader import DataLoaderSelector



class DatasetType(Enum):
    train = 'train'
    validation = 'validation'
    test = 'test'
    
    
    
class DataGenerator():
    
    def __init__(self, data_path: str, train_ratio: int = 0.6, validation_ratio: int = 0.2,
                 test_ratio: int = 0.2, contrastive: bool = False,
                 train_subset: Optional[float] = None, seed: Optional[int] = None):
        """
        Initialize the DataGenerator object.

        Args:
            data_path (str): The path to the data.
            train_ratio (int, optional): The ratio of training data. Defaults to 0.6.
            validation_ratio (int, optional): The ratio of validation data. Defaults to 0.2.
            test_ratio (int, optional): The ratio of test data. Defaults to 0.2.
            contrastive (bool, optional): Whether to use contrastive learning. Defaults to False.
            train_subset (float, optional): The ratio of training data to use as a subset. Defaults to None.
            seed (int, optional): The random seed. Defaults to None.

        Returns:
            None
        """
        data_loader_selector = DataLoaderSelector(data_path)
        X, y = data_loader_selector.get_data(gland='both')
        
        # Split the data to the three datasets
        (X_train, y_train,
         X_val, y_val,
         X_test, y_test) = self.data_split(X, y,
                                           train_ratio, validation_ratio, test_ratio,
                                           seed)
                                           
        if train_subset is not None:
            X_train, y_train = self.train_split(X_train, y_train,
                                                split_ratio=train_subset,
                                                seed=seed)
        
        self.X_train = X_train.reset_index(drop=True)
        self.X_val = X_val.reset_index(drop=True)
        self.X_test = X_test.reset_index(drop=True)
        self.y_train = y_train.reset_index(drop=True)
        self.y_val = y_val.reset_index(drop=True)
        self.y_test = y_test.reset_index(drop=True)
        self.X_size = self.X_train.shape[1:]
        self.y_size = self.y_train.shape[1:]
        self.max_data = max(self.X_train.shape[0],
                            self.X_val.shape[0],
                            self.X_test.shape[0])
        
        self.labels = list(self.X_train.columns)
        
        self.contrastive = contrastive
        
    
    @staticmethod
    def data_split(X: np.ndarray, y: np.ndarray, train_ratio: float,
                   validation_ratio: float, test_ratio: float, seed: int,
                   class_balance: bool = False) -> Tuple[np.ndarray]:
        """
        Split the data into training, validation, and test sets.

        Args:
            X (np.ndarray): The input features.
            y (np.ndarray): The target labels.
            train_ratio (float): The ratio of data to be used for training.
            validation_ratio (float): The ratio of data to be used for validation.
            test_ratio (float): The ratio of data to be used for testing.
            seed (int): The random seed for reproducibility.
            class_balance (bool, optional): Whether to balance the classes in the training set. Defaults to False.

        Returns:
            Tuple[np.ndarray]: A tuple containing the training features, training labels,
                               validation features, validation labels, test features, and test labels.
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=1-train_ratio,
                                                            random_state=seed,
                                                            shuffle=True,
                                                            stratify=y)
        
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test,
                                                        test_size=test_ratio/(test_ratio+validation_ratio),
                                                        random_state=seed,
                                                        shuffle=True,
                                                        stratify=y_test)
        
        if class_balance:
            # Note: Assumes class 1 is less than class 0
            class_0_size = len(X_train.loc[y_train['Cancer'] == 0])
            class_1_size = len(X_train.loc[y_train['Cancer'] == 1])
            
            X_train_0 = X_train.loc[y_train['Cancer'] == 0]
            y_train_0 = y_train.loc[y_train['Cancer'] == 0]
            X_train_1 = X_train.loc[y_train['Cancer'] == 1]
            y_train_1 = y_train.loc[y_train['Cancer'] == 1]
            
            X_train_1 = pd.concat([X_train_1] * math.ceil(class_0_size / class_1_size),
                                  ignore_index=True).iloc[:class_0_size]
            y_train_1 = pd.concat([y_train_1] * math.ceil(class_0_size / class_1_size),
                                  ignore_index=True).iloc[:class_0_size]
            
            X_train = pd.concat([X_train_0, X_train_1]).sample(frac=1.0, random_state=seed, ignore_index=True)
            y_train = pd.concat([y_train_0, y_train_1]).sample(frac=1.0, random_state=seed, ignore_index=True)        
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    
    @staticmethod
    def train_split(X: np.ndarray, y: np.ndarray, split_ratio: float, seed: int) -> Tuple[np.ndarray]:
        """
        Split the input data into training and validation sets.

        Args:
            X (np.ndarray): The input data array.
            y (np.ndarray): The target labels array.
            split_ratio (float): The ratio to split the data into training and validation sets.
            seed (int): The random seed.

        Returns:
            Tuple[np.ndarray]: A tuple containing the training data and labels arrays.
        """
        X_train, _, y_train, _ = train_test_split(X, y,
                                                  test_size=1-split_ratio,
                                                  random_state=seed,
                                                  shuffle=True,
                                                  stratify=y)

        return X_train, y_train
        
        
    @staticmethod
    def to_structure(X: np.ndarray, y: np.ndarray, contrastive: bool = False,
                     add_batch_axis: bool = False) -> Dict[str, np.ndarray]:
        """
        Convert input data and labels into a structured dictionary.

        Args:
            X (np.ndarray): The input data.
            y (np.ndarray): The labels.
            contrastive (bool, optional): Whether to include contrastive labels. Defaults to False.
            add_batch_axis (bool, optional): Whether to add a batch axis to the data and labels. Defaults to False.

        Returns:
            Dict[str, np.ndarray]: A dictionary containing the structured data.

        """
        if add_batch_axis:
            X = np.expand_dims(X, axis=0) 
            y = np.expand_dims(y, axis=0)
            
        data = ({'base_input': X},
                {'base_output': y})
        
        if contrastive:
            data[1]['base_contrastive'] = y    
        
        return data
    
    
    def data_generator(self, X: pd.DataFrame, y: pd.DataFrame,
                       verbose: int = 0) -> Generator[Tuple[Dict[str, np.ndarray]], None, None]:
        """
        Generate data for training or evaluation.

        Args:
            X (pd.DataFrame): The input features.
            y (pd.DataFrame): The target labels.
            verbose (int, optional): Verbosity level. Defaults to 0.

        Yields:
            Generator[Tuple[Dict[str, np.ndarray]], None, None]: A generator that yields a tuple containing a
                dictionary of input features and target labels.
        """
        for index, row in X.iterrows():
            if verbose > 0:
                print('Generating patient: ', index)
            
            X_row = X.iloc[[index]].to_numpy()[0]
            y_row = y.iloc[[index]].to_numpy()[0]
            
            data = self.to_structure(X_row, y_row,
                                     contrastive=self.contrastive,
                                     add_batch_axis=False)
            
            yield data

    
    def data_generator_index(self, index: int, dataset: Union[DatasetType, int],
                             verbose: int = 0) -> Generator[Tuple[Dict[str, np.ndarray]], None, None]:
        """
        Generate data for a specific index and dataset.

        Args:
            index (int): The index of the data to generate.
            dataset (Union[DatasetType, int]): The dataset to generate data from.
            verbose (int, optional): Verbosity mode. Defaults to 0.

        Yields:
            Generator[Tuple[Dict[str, np.ndarray]], None, None]: A generator that yields a tuple containing
                a dictionary of arrays.

        Raises:
            ValueError: If an invalid dataset type is provided.

        """
        if dataset == DatasetType.train or dataset == 0:
            X = self.X_train[index: index+1]
            y = self.y_train[index: index+1]
        elif dataset == DatasetType.validation or dataset == 1:
            X = self.X_val[index: index+1]
            y = self.y_val[index: index+1]
        elif dataset == DatasetType.test or dataset == 2:
            X = self.X_test[index: index+1]
            y = self.y_test[index: index+1]
        else:
            raise ValueError('Invalid dataset type provided.')

        yield from self.data_generator(X, y, verbose=verbose)
            
            
    def dataset_generator(self, dataset: DatasetType, verbose: int = 0) -> Generator[Tuple[Dict[str, np.ndarray]], None, None]:
        """
        Generate data batches for a specific dataset.

        Args:
            dataset (DatasetType): The type of dataset to generate batches for.
            verbose (int, optional): Verbosity level. Defaults to 0.

        Yields:
            Generator[Tuple[Dict[str, np.ndarray]], None, None]: A generator that yields data.

        Raises:
            ValueError: If an invalid dataset type is provided.
        """
        if dataset == DatasetType.train or dataset == 0:
            X = self.X_train
            y = self.y_train
        elif dataset == DatasetType.validation or dataset == 1:
            X = self.X_val
            y = self.y_val
        elif dataset == DatasetType.test or dataset == 2:
            X = self.X_test
            y = self.y_test
        else:
            raise ValueError('Invalid dataset type provided.')

        yield from self.data_generator(X, y, verbose=verbose)
                
                
    def train_generator(self, verbose: int = 0) -> Generator[Tuple[Dict[str, np.ndarray]], None, None]:
        """
        Generate training data batches.

        Args:
            verbose (int, optional): Verbosity level. Defaults to 0.

        Yields:
            Generator[Tuple[Dict[str, np.ndarray]], None, None]: A generator that yields batches of training data.
        """
        yield from self.dataset_generator(DatasetType.train, verbose=verbose)
        
    
    def validation_generator(self, verbose: int = 0) -> Generator[Tuple[Dict[str, np.ndarray]], None, None]:
        """
        Generate data for validation.

        Args:
            verbose (int, optional): Verbosity mode. Defaults to 0.

        Yields:
            Generator[Tuple[Dict[str, np.ndarray]], None, None]:  generator that yields batches of validation data.
        """
        yield from self.dataset_generator(DatasetType.validation, verbose=verbose)
            
    
    def test_generator(self, verbose: int = 0) -> Generator[Tuple[Dict[str, np.ndarray]], None, None]:
        """
        Generate test data batches.

        Args:
            verbose (int, optional): Verbosity level. Defaults to 0.

        Yields:
            Generator[Tuple[Dict[str, np.ndarray]], None, None]: A generator that yields batches of test data.
        """
        yield from self.dataset_generator(DatasetType.test, verbose=verbose)
        
        
        
class DataGlobalGenerator(DataGenerator):
    
    
    def __init__(self, data_path: str, train_ratio: int = 0.6, validation_ratio: int = 0.2,
                 test_ratio: int = 0.2, contrastive: bool = False,
                 train_subset: Optional[float] = None, seed: Optional[int] = None):
        """
        Initialize the DataGenerator object for the Global model.

        Args:
            data_path (str): The path to the data.
            train_ratio (int, optional): The ratio of training data. Defaults to 0.6.
            validation_ratio (int, optional): The ratio of validation data. Defaults to 0.2.
            test_ratio (int, optional): The ratio of test data. Defaults to 0.2.
            contrastive (bool, optional): Whether to use contrastive learning. Defaults to False.
            train_subset (float, optional): The ratio of training data to use as a subset. Defaults to None.
            seed (int, optional): The random seed. Defaults to None.

        Returns:
            None
        """
        data_loader_selector = DataLoaderSelector(data_path)
        
        X, y = data_loader_selector.get_data(gland='both', surface='both',
                                             use_ref_values=True,
                                             use_axillary_values=True)
        
        # Split the data to the three datasets
        (X_train, y_train,
         X_val, y_val,
         X_test, y_test) = self.data_split(X, y,
                                           train_ratio, validation_ratio, test_ratio,
                                           seed)
        
        if train_subset is not None:
            X_train, y_train = self.train_split(X_train, y_train,
                                                split_ratio=train_subset,
                                                seed=seed)
                                           
        self.X_train = X_train.reset_index(drop=True)
        self.X_val = X_val.reset_index(drop=True)
        self.X_test = X_test.reset_index(drop=True)
        self.y_train = y_train.reset_index(drop=True)
        self.y_val = y_val.reset_index(drop=True)
        self.y_test = y_test.reset_index(drop=True)
        
        self.X_size = self.X_train.shape[1:]
        self.y_size = self.y_train.shape[1:]
        self.max_data = max(self.X_train.shape[0],
                            self.X_val.shape[0],
                            self.X_test.shape[0])
        
        self.augment = AugmentData()
        
        self.contrastive = contrastive
    
        
    @staticmethod
    def to_structure(X: np.ndarray, X_flipped: np.ndarray, y: np.ndarray, contrastive: bool = False,
                     add_batch_axis: bool = False) -> Tuple[Dict[str, np.ndarray]]:
        """
        Convert input data to a structured format.

        Args:
            X (np.ndarray): The input data.
            X_flipped (np.ndarray): The flipped input data.
            y (np.ndarray): The output data.
            contrastive (bool, optional): Whether to include contrastive data. Defaults to False.
            add_batch_axis (bool, optional): Whether to add a batch axis to the data. Defaults to False.

        Returns:
            Tuple[Dict[str, np.ndarray]]: A tuple for the input and output containing a dictionary of arrays.
        """
        if add_batch_axis:
            X = np.expand_dims(X, axis=0)
            X_flipped = np.expand_dims(X_flipped, axis=0)
            y = np.expand_dims(y, axis=0)
            
        data = ({'global_input': X,
                 'global_input_flipped': X_flipped},
                {'global_output': y})
        
        if contrastive:
            data[1]['global_contrastive'] = y
        
        return data
    
    
    def data_generator(self, X: pd.DataFrame, y: pd.DataFrame,
                       verbose: int = 0) -> Generator[Tuple[Dict[str, np.ndarray]], None, None]:
        """
        Generate data for training.

        Args:
            X (pd.DataFrame): The input data.
            y (pd.DataFrame): The target data.
            verbose (int, optional): Verbosity level. Defaults to 0.

        Yields:
            Generator[Tuple[Dict[str, np.ndarray]], None, None]: A generator that yields data.

        """
        X_flipped = self.augment.flip(data=X.copy(deep=True))
        X_flipped = self.augment.rotate(data=X_flipped, rotate_by=4)
        
        for index, row in X.iterrows():
            if verbose > 0:
                print('Generating patient: ', index)
                
            X_row = X.iloc[[index]].to_numpy()[0]
            X_row_flipped = X_flipped.iloc[[index]].to_numpy()[0]
            y_row = y.iloc[[index]].to_numpy()[0]
            
            
            data = self.to_structure(X_row, X_row_flipped, y_row,
                                     contrastive=self.contrastive,
                                     add_batch_axis=False)
            
            yield data
            
            
    def data_generator_index(self, index: int, dataset: Union[DatasetType, int],
                             verbose: int = 0) -> Generator[Tuple[Dict[str, np.ndarray]], None, None]:
        """
        Generate data for a specific index and dataset.

        Args:
            index (int): The index of the data to generate.
            dataset (Union[DatasetType, int]): The dataset to generate data from.
            verbose (int, optional): Verbosity mode. Defaults to 0.

        Yields:
            Generator[Tuple[Dict[str, np.ndarray]], None, None]: A generator that yields a tuple containing a dictionary of arrays.

        Raises:
            ValueError: If an invalid dataset type is provided.
        """
        if dataset == DatasetType.train or dataset == 0:
            X = self.X_train[index: index+1]
            y = self.y_train[index: index+1]
        elif dataset == DatasetType.validation or dataset == 1:
            X = self.X_val[index: index+1]
            y = self.y_val[index: index+1]
        elif dataset == DatasetType.test or dataset == 2:
            X = self.X_test[index: index+1]
            y = self.y_test[index: index+1]
        else:
            raise ValueError('Invalid dataset type provided.')

        yield from self.data_generator(X, y, verbose=verbose)
            
            
    def dataset_generator(self, dataset: DatasetType, verbose: int = 0) -> Generator[Tuple[Dict[str, np.ndarray]], None, None]:
        """
        Generate data batches for a specific dataset.

        Args:
            dataset (DatasetType): The type of dataset to generate batches for.
            verbose (int, optional): Verbosity level. Defaults to 0.

        Yields:
            Generator[Tuple[Dict[str, np.ndarray]], None, None]: A generator that yields batches of data.

        Raises:
            ValueError: If an invalid dataset type is provided.
        """
        if dataset == DatasetType.train or dataset == 0:
            X = self.X_train
            y = self.y_train
        elif dataset == DatasetType.validation or dataset == 1:
            X = self.X_val
            y = self.y_val
        elif dataset == DatasetType.test or dataset == 2:
            X = self.X_test
            y = self.y_test
        else:
            raise ValueError('Invalid dataset type provided.')

        yield from self.data_generator(X, y, verbose=verbose)


    
class DataRegionalGenerator(DataGenerator):
    
    def __init__(self, data_path: str, train_ratio: int = 0.6, validation_ratio: int = 0.2,
                 test_ratio: int = 0.2, contrastive: bool = False,
                 train_subset: Optional[float] = None, seed: Optional[int] = None):
        """
        Initialize the DataGenerator object for the Regional model.

        Args:
            data_path (str): The path to the data.
            train_ratio (int, optional): The ratio of training data. Defaults to 0.6.
            validation_ratio (int, optional): The ratio of validation data. Defaults to 0.2.
            test_ratio (int, optional): The ratio of test data. Defaults to 0.2.
            contrastive (bool, optional): Whether to use contrastive learning. Defaults to False.
            train_subset (float, optional): The ratio of training data to use as a subset. Defaults to None.
            seed (int, optional): The random seed. Defaults to None.

        Returns:
            None
        """
        data_loader_selector = DataLoaderSelector(data_path)
        
        X_l, y = data_loader_selector.get_data(gland='left')
        X_r, _ = data_loader_selector.get_data(gland='right')
        
        # Split the data to the three datasets
        (X_l_train, y_l_train,
         X_l_val, y_l_val,
         X_l_test, y_l_test) = self.data_split(X_l, y,
                                               train_ratio, validation_ratio, test_ratio,
                                               seed)
                                               
        (X_r_train, y_r_train,
         X_r_val, y_r_val,
         X_r_test, y_r_test) = self.data_split(X_r, y,
                                               train_ratio, validation_ratio, test_ratio,
                                               seed)
        
        if train_subset is not None:
            X_l_train, y_l_train = self.train_split(X_l_train, y_l_train,
                                                    split_ratio=train_subset,
                                                    seed=seed)
            X_r_train, y_r_train = self.train_split(X_r_train, y_r_train,
                                                    split_ratio=train_subset,
                                                    seed=seed)
                                               
        assert y_l_train.equals(y_r_train)
        assert y_l_val.equals(y_r_val)
        assert y_l_test.equals(y_r_test)
                        
        self.X_l_train = X_l_train.reset_index(drop=True)
        self.X_r_train = X_r_train.reset_index(drop=True)
        self.X_l_val = X_l_val.reset_index(drop=True)
        self.X_r_val = X_r_val.reset_index(drop=True)
        self.X_l_test = X_l_test.reset_index(drop=True)
        self.X_r_test = X_r_test.reset_index(drop=True)
        self.y_train = y_l_train.reset_index(drop=True)
        self.y_val = y_l_val.reset_index(drop=True)
        self.y_test = y_l_test.reset_index(drop=True)
        
        self.X_size = self.X_l_train.shape[1:]
        self.y_size = self.y_train.shape[1:]
        self.max_data = max(self.X_l_train.shape[0],
                            self.X_l_val.shape[0],
                            self.X_l_test.shape[0])
        
        self.contrastive = contrastive
        
    
    @staticmethod
    def to_structure(X_l: np.ndarray, X_r: np.ndarray, y: np.ndarray,
                     contrastive: bool = False, add_batch_axis: bool = False) -> Tuple[Dict[str, np.ndarray]]:
        """
        Convert input data into a structured format.

        Args:
            X_l (np.ndarray): Left input data.
            X_r (np.ndarray): Right input data.
            y (np.ndarray): Output data.
            contrastive (bool, optional): Flag indicating whether to include contrastive data. Defaults to False.
            add_batch_axis (bool, optional): Flag indicating whether to add a batch axis. Defaults to False.

        Returns:
            Tuple[Dict[str, np.ndarray]]: A tuple for the input and output containing a dictionary of arrays.
        """
        if add_batch_axis:
            X_l = np.expand_dims(X_l, axis=0)
            X_r = np.expand_dims(X_r, axis=0)
            y = np.expand_dims(y, axis=0)
            
        data = ({'regional_input_l': X_l,
                 'regional_input_r': X_r},
                {'regional_output': y})
        
        if contrastive:
            data[1]['regional_contrastive'] = y
        
        return data
    
    
    def data_generator(self, X_l: pd.DataFrame, X_r: pd.DataFrame, y: pd.DataFrame,
                       verbose: int = 0) -> Generator[Tuple[Dict[str, np.ndarray]], None, None]:
        """
        Generate data batches for training or evaluation.

        Args:
            X_l (pd.DataFrame): Left input data.
            X_r (pd.DataFrame): Right input data.
            y (pd.DataFrame): Target data.
            verbose (int, optional): Verbosity level. Defaults to 0.

        Yields:
            Generator[Tuple[Dict[str, np.ndarray]], None, None]: A generator that yields data batches.

        """
        for index, row in X_l.iterrows():
            if verbose > 0:
                print('Generating patient: ', index)
            
            X_l_row = X_l.iloc[[index]].to_numpy()[0]
            X_r_row = X_r.iloc[[index]].to_numpy()[0]
            y_row = y.iloc[[index]].to_numpy()[0]
            
            data = self.to_structure(X_l_row, X_r_row, y_row,
                                     contrastive=self.contrastive,
                                     add_batch_axis=False)
            
            yield data
            
            
    def data_generator_index(self, index: int, dataset: Union[DatasetType, int],
                             verbose: int = 0) -> Generator[Tuple[Dict[str, np.ndarray]], None, None]:
        """
        Generate data for a specific index and dataset.

        Args:
            index (int): The index of the data to generate.
            dataset (Union[DatasetType, int]): The dataset to generate data from.
            verbose (int, optional): Verbosity mode. Defaults to 0.

        Yields:
            Generator[Tuple[Dict[str, np.ndarray]], None, None]: A generator that yields a tuple containing a dictionary of arrays.

        Raises:
            ValueError: If the dataset is not a valid DatasetType or integer value.
        """
        if dataset == DatasetType.train or dataset == 0:
            X_l = self.X_l_train[index: index+1]
            X_r = self.X_r_train[index: index+1]
            y = self.y_train[index: index+1]
        elif dataset == DatasetType.validation or dataset == 1:
            X_l = self.X_l_val[index: index+1]
            X_r = self.X_r_val[index: index+1]
            y = self.y_val[index: index+1]
        elif dataset == DatasetType.test or dataset == 2:
            X_l = self.X_l_test[index: index+1]
            X_r = self.X_r_test[index: index+1]
            y = self.y_test[index: index+1]
        else:
            raise ValueError("Invalid dataset type provided.")

        yield from self.data_generator(X_l, X_r, y, verbose=verbose)
            
            
    def dataset_generator(self, dataset: DatasetType, verbose: int = 0) -> Generator[Tuple[Dict[str, np.ndarray]], None, None]:
        """
        Generate data for a specific dataset.

        Args:
            dataset (DatasetType): The type of dataset to generate data for.
            verbose (int, optional): Verbosity level. Defaults to 0.

        Yields:
            Generator[Tuple[Dict[str, np.ndarray]], None, None]: A generator that yields a tuple containing a dictionary of arrays.

        Raises:
            ValueError: If an invalid dataset type is provided.
        """
        if dataset == DatasetType.train or dataset == 0:
            X_l = self.X_l_train
            X_r = self.X_r_train
            y = self.y_train
        elif dataset == DatasetType.validation or dataset == 1:
            X_l = self.X_l_val
            X_r = self.X_r_val
            y = self.y_val
        elif dataset == DatasetType.test or dataset == 2:
            X_l = self.X_l_test
            X_r = self.X_r_test
            y = self.y_test
        else:
            raise ValueError('Invalid dataset type provided.')

        yield from self.data_generator(X_l, X_r, y, verbose=verbose)
        
        
        
class DataLocalGenerator(DataGenerator):
    
    def __init__(self, data_path: str, train_ratio: int = 0.6, validation_ratio: int = 0.2,
                 test_ratio: int = 0.2, contrastive: bool = False,
                 train_subset: Optional[float] = None, seed: Optional[int] = None):
        """
        Initializes a DataGenerator object for the Local model.

        Args:
            data_path (str): The path to the data.
            train_ratio (int, optional): The ratio of training data. Defaults to 0.6.
            validation_ratio (int, optional): The ratio of validation data. Defaults to 0.2.
            test_ratio (int, optional): The ratio of test data. Defaults to 0.2.
            contrastive (bool, optional): Whether to use contrastive learning. Defaults to False.
            train_subset (float, optional): The ratio of training data to use as a subset. Defaults to None.
            seed (int, optional): The random seed for data splitting. Defaults to None.
        
        Returns:
            None
        """
        data_loader_selector = DataLoaderSelector(data_path)
        
        X_s, y = data_loader_selector.get_data(gland='both', surface='skin',
                                               use_ref_values=False, use_axillary_values=False,
                                               use_age=False)
        X_d, _ = data_loader_selector.get_data(gland='both', surface='depth',
                                               use_ref_values=False, use_axillary_values=False,
                                               use_age=False)
        
        
        # Split the data to the three datasets
        (X_s_train, y_s_train,
         X_s_val, y_s_val,
         X_s_test, y_s_test) = self.data_split(X_s, y,
                                               train_ratio, validation_ratio, test_ratio,
                                               seed)
                                               
        (X_d_train, y_d_train,
         X_d_val, y_d_val,
         X_d_test, y_d_test) = self.data_split(X_d, y,
                                               train_ratio, validation_ratio, test_ratio,
                                               seed)
        
        if train_subset is not None:
            X_s_train, y_s_train = self.train_split(X_s_train, y_s_train,
                                                    split_ratio=train_subset,
                                                    seed=seed)
            X_d_train, y_d_train = self.train_split(X_d_train, y_d_train,
                                                    split_ratio=train_subset,
                                                    seed=seed)
                                               
        assert y_s_train.equals(y_d_train)
        assert y_s_val.equals(y_d_val)
        assert y_s_test.equals(y_d_test)
                        
        self.X_s_train = X_s_train.reset_index(drop=True)
        self.X_d_train = X_d_train.reset_index(drop=True)
        self.X_s_val = X_s_val.reset_index(drop=True)
        self.X_d_val = X_d_val.reset_index(drop=True)
        self.X_s_test = X_s_test.reset_index(drop=True)
        self.X_d_test = X_d_test.reset_index(drop=True)
        self.y_train = y_s_train.reset_index(drop=True)
        self.y_val = y_s_val.reset_index(drop=True)
        self.y_test = y_s_test.reset_index(drop=True)
        
        self.X_input_size = self.X_s_train.shape[1:][0]
        self.X_size = (2,)
        self.y_size = self.y_train.shape[1:]
        self.max_data = max(self.X_s_train.shape[0],
                            self.X_s_val.shape[0],
                            self.X_s_test.shape[0])
        
        self.labels = list(self.X_d_train.columns)
        
        self.contrastive = contrastive
        
        
    @staticmethod
    def to_structure(X_s: np.ndarray, X_d: np.ndarray, y: np.ndarray,
                     contrastive: bool = False, add_batch_axis: bool = False) -> Tuple[Dict[str, np.ndarray]]:
        """
        Convert input data into a structured format.

        Args:
            X_s (np.ndarray): The skin temperature input data.
            X_d (np.ndarray): The depth temperature input data.
            y (np.ndarray): The output data.
            contrastive (bool, optional): Whether to include contrastive data. Defaults to False.
            add_batch_axis (bool, optional): Whether to add a batch axis to the input data. Defaults to False.

        Returns:
            Tuple[Dict[str, np.ndarray]]: A tuple for the input and output containing a dictionary of arrays.
        """
        # Merge together
        X = np.asarray([X_s, X_d])
        # Transpose so that skin and depth values are paired individually
        X = X.T
        
        input_data = {}
        for i in range(X.shape[0]):
            X_value = X[i]
            
            if add_batch_axis:
                X_value = np.expand_dims(X_value, axis=0)
                
            input_data[f'local_input_{i}'] = X_value
            
        if add_batch_axis:
            y = np.expand_dims(y, axis=0)
            
        data = (input_data,
                {'local_output': y})
        
        if contrastive:
            data[1]['local_contrastive'] = y
        
        return data
    
    
    def data_generator(self, X_s: pd.DataFrame, X_d: pd.DataFrame, y: pd.DataFrame,
                       verbose: int = 0) -> Generator[Tuple[Dict[str, np.ndarray]], None, None]:
        """
        Generate data batches for training or evaluation.

        Args:
            X_s (pd.DataFrame): The skin temperature input data.
            X_d (pd.DataFrame): The depth temperature input data.
            y (pd.DataFrame): The target output data.
            verbose (int, optional): Verbosity level. Defaults to 0.

        Yields:
            Generator[Tuple[Dict[str, np.ndarray]], None, None]: A generator that yields data batches.
        """
        for index, row in X_s.iterrows():
            if verbose > 0:
                print('Generating patient: ', index)
            
            X_s_row = X_s.iloc[[index]].to_numpy()[0]
            X_d_row = X_d.iloc[[index]].to_numpy()[0]
            y_row = y.iloc[[index]].to_numpy()[0]
            
            data = self.to_structure(X_s_row, X_d_row, y_row,
                                     contrastive=self.contrastive,
                                     add_batch_axis=False)
            
            yield data
            
            
    def data_generator_index(self, index: int, dataset: Union[DatasetType, int],
                             verbose: int = 0) -> Generator[Tuple[Dict[str, np.ndarray]], None, None]:
        """
        Generate data for a specific index and dataset.

        Args:
            index (int): The index of the data to generate.
            dataset (Union[DatasetType, int]): The dataset to generate data from.
            verbose (int, optional): Verbosity mode. Defaults to 0.

        Yields:
            Generator[Tuple[Dict[str, np.ndarray]], None, None]: A generator that yields a tuple containing a dictionary of arrays.

        Raises:
            ValueError: If an invalid dataset type is provided.
        """
        if dataset == DatasetType.train or dataset == 0:
            X_s = self.X_s_train[index: index+1]
            X_d = self.X_d_train[index: index+1]
            y = self.y_train[index: index+1]
        elif dataset == DatasetType.validation or dataset == 1:
            X_s = self.X_s_val[index: index+1]
            X_d = self.X_d_val[index: index+1]
            y = self.y_val[index: index+1]
        elif dataset == DatasetType.test or dataset == 2:
            X_s = self.X_s_test[index: index+1]
            X_d = self.X_d_test[index: index+1]
            y = self.y_test[index: index+1]
        else:
            raise ValueError('Invalid dataset type provided.')

        yield from self.data_generator(X_s, X_d, y, verbose=verbose)
            
            
    def dataset_generator(self, dataset: DatasetType, verbose: int = 0) -> Generator[Tuple[Dict[str, np.ndarray]], None, None]:
        """
        Generate data batches for a given dataset.

        Args:
            dataset (DatasetType): The type of dataset to generate batches for.
            verbose (int, optional): Verbosity level. Defaults to 0.

        Yields:
            Generator[Tuple[Dict[str, np.ndarray]], None, None]: A generator that yields batches of data.

        Raises:
            ValueError: If an invalid dataset type is provided.
        """
        if dataset == DatasetType.train or dataset == 0:
            X_s = self.X_s_train
            X_d = self.X_d_train
            y = self.y_train
        elif dataset == DatasetType.validation or dataset == 1:
            X_s = self.X_s_val
            X_d = self.X_d_val
            y = self.y_val
        elif dataset == DatasetType.test or dataset == 2:
            X_s = self.X_s_test
            X_d = self.X_d_test
            y = self.y_test
        else:
            raise ValueError('Invalid dataset type provided.')

        yield from self.data_generator(X_s, X_d, y, verbose=verbose)
        
        

class DataJointGenerator(DataGenerator):
    
    def __init__(self, data_path: str, train_ratio: int = 0.6, validation_ratio: int = 0.2,
                 test_ratio: int = 0.2, contrastive: bool = False,
                 train_subset: Optional[float] = None, seed: Optional[int] = None):
        """
        Initialize the DataGenerator object for the Joint model.

        Args:
            data_path (str): The path to the data.
            train_ratio (int, optional): The ratio of training data. Defaults to 0.6.
            validation_ratio (int, optional): The ratio of validation data. Defaults to 0.2.
            test_ratio (int, optional): The ratio of test data. Defaults to 0.2.
            contrastive (bool, optional): Whether to use contrastive learning. Defaults to False.
            train_subset (float, optional): The ratio of training data to use as a subset. Defaults to None.
            seed (int, optional): The random seed. Defaults to None.

        Returns:
            None
        """
        # Generate a random seed if None, so that each individual generator
        # has the same data between the 3 data sets
        if seed is None:
            rng = np.random.default_rng()
            seed = rng.integers(1000, size=1).item()

        self.local_generator = DataLocalGenerator(data_path=data_path,
                                                  train_ratio=train_ratio,
                                                  validation_ratio=validation_ratio,
                                                  test_ratio=test_ratio,
                                                  train_subset=train_subset,
                                                  seed=seed)

        self.regional_generator = DataRegionalGenerator(data_path=data_path,
                                                        train_ratio=train_ratio,
                                                        validation_ratio=validation_ratio,
                                                        test_ratio=test_ratio,
                                                        train_subset=train_subset,
                                                        seed=seed)

        self.global_generator = DataGlobalGenerator(data_path=data_path,
                                                    train_ratio=train_ratio,
                                                    validation_ratio=validation_ratio,
                                                    test_ratio=test_ratio,
                                                    train_subset=train_subset,
                                                    seed=seed)

        assert self.global_generator.y_train.equals(self.regional_generator.y_train)
        assert self.global_generator.y_train.equals(self.local_generator.y_train)
        assert self.global_generator.y_val.equals(self.regional_generator.y_val)
        assert self.global_generator.y_val.equals(self.local_generator.y_val)
        assert self.global_generator.y_test.equals(self.regional_generator.y_test)
        assert self.global_generator.y_test.equals(self.local_generator.y_test)

        self.y_train = self.global_generator.y_train
        self.y_val = self.global_generator.y_val
        self.y_test = self.global_generator.y_test

        self.X_local_input_size = self.local_generator.X_input_size
        self.X_sizes = {'local': self.local_generator.X_size,
                        'regional': self.regional_generator.X_size,
                        'global': self.global_generator.X_size}
        self.y_size = self.global_generator.y_size
        self.max_data = self.global_generator.max_data

        self.contrastive = contrastive
        

    def to_structure(self, global_data: Tuple[Dict[str, np.ndarray]], regional_data: Tuple[Dict[str, np.ndarray]],
                     local_data: Tuple[Dict[str, np.ndarray]], y: np.ndarray, contrastive: bool = False,
                     add_batch_axis: bool = False) -> Tuple[Dict[str, np.ndarray]]:
        """
        Convert input data into a structured format.

        Args:
            global_data (Tuple[Dict[str, np.ndarray]]): A tuple containing dictionaries of global data.
            regional_data (Tuple[Dict[str, np.ndarray]]): A tuple containing dictionaries of regional data.
            local_data (Tuple[Dict[str, np.ndarray]]): A tuple containing dictionaries of local data.
            y (np.ndarray): The target output data.
            contrastive (bool, optional): Flag indicating whether to include contrastive data. Defaults to False.
            add_batch_axis (bool, optional): Flag indicating whether to add a batch axis to the input data. Defaults to False.

        Returns:
            Tuple[Dict[str, np.ndarray]]: A tuple for the input and output containing a dictionary of arrays.
        """
        input_data = {}
        # Update keys
        for key, item in global_data[0].items():
            if add_batch_axis:
                item = np.expand_dims(item, axis=0)
            input_data['joint_' + key] = item
            
        for key, item in regional_data[0].items():
            if add_batch_axis:
                item = np.expand_dims(item, axis=0)
            input_data['joint_' + key] = item
            
        for key, item in local_data[0].items():
            if add_batch_axis:
                item = np.expand_dims(item, axis=0)
            input_data['joint_' + key] = item
            
        if add_batch_axis:
            y = np.expand_dims(y, axis=0)
            
        data = (input_data,
                {'local_output': y,
                 'regional_output': y,
                 'global_output': y,
                 'joint_output': y})
        
        if contrastive:
            data[1]['joint_local_contrastive'] = y
            data[1]['joint_regional_contrastive'] = y
            data[1]['joint_global_contrastive'] = y
        
        return data
    
    
    def data_generator(self, X: pd.DataFrame, X_l: pd.DataFrame, X_r: pd.DataFrame, X_s: pd.DataFrame,
                           X_d: pd.DataFrame, y: pd.DataFrame, verbose: int = 0) -> Generator[Tuple[Dict[str, np.ndarray]], None, None]:
        """
        Generate data batches for training or evaluation.

        Args:
            X (pd.DataFrame): The input data for the global model.
            X_l (pd.DataFrame): The left temperature input data for the regional model.
            X_r (pd.DataFrame): The right remperature input data for the regional model.
            X_s (pd.DataFrame): The skin temperature input data for the local model.
            X_d (pd.DataFrame): The depth temperature input data for the local model.
            y (pd.DataFrame): The target data.
            verbose (int, optional): Verbosity level. Defaults to 0.

        Yields:
            Generator[Tuple[Dict[str, np.ndarray]], None, None]: A generator that yields a tuple of dictionaries
                containing the input data and target data for each batch.
        """
        for g, r, l in zip(self.global_generator.data_generator(X, y, verbose),
                            self.regional_generator.data_generator(X_l, X_r, y, verbose),
                            self.local_generator.data_generator(X_s, X_d, y, verbose)):
            data = self.to_structure(g, r, l, g[1]['global_output'],
                                        contrastive=self.contrastive,
                                        add_batch_axis=False)

            yield data
            
            
    def data_generator_index(self, index: int, dataset: Union[DatasetType, int],
                             verbose: int = 0) -> Generator[Tuple[Dict[str, np.ndarray]], None, None]:
        """
        Generate data for a specific index and dataset.

        Args:
            index (int): The index of the data to generate.
            dataset (Union[DatasetType, int]): The dataset to generate data from.
            verbose (int, optional): Verbosity level. Defaults to 0.

        Yields:
            Generator[Tuple[Dict[str, np.ndarray]], None, None]: A generator that yields a tuple of dictionaries containing the generated data.

        Raises:
            ValueError: If an invalid dataset type is provided.
        """
        if dataset == DatasetType.train or dataset == 0:
            X = self.global_generator.X_train[index: index+1]
            X_l = self.regional_generator.X_l_train[index: index+1]
            X_r = self.regional_generator.X_r_train[index: index+1]
            X_s = self.local_generator.X_s_train[index: index+1]
            X_d = self.local_generator.X_d_train[index: index+1]
            y = self.y_train[index: index+1]
        elif dataset == DatasetType.validation or dataset == 1:
            X = self.global_generator.X_val[index: index+1]
            X_l = self.regional_generator.X_l_val[index: index+1]
            X_r = self.regional_generator.X_r_val[index: index+1]
            X_s = self.local_generator.X_s_val[index: index+1]
            X_d = self.local_generator.X_d_val[index: index+1]
            y = self.y_val[index: index+1]
        elif dataset == DatasetType.test or dataset == 2:
            X = self.global_generator.X_test[index: index+1]
            X_l = self.regional_generator.X_l_test[index: index+1]
            X_r = self.regional_generator.X_r_test[index: index+1]
            X_s = self.local_generator.X_s_test[index: index+1]
            X_d = self.local_generator.X_d_test[index: index+1]
            y = self.y_test[index: index+1]
        else:
            raise ValueError('Invalid dataset type provided.')
            
        yield from self.data_generator(X, X_l, X_r, X_s, X_d, y, verbose=verbose)
            
            
    def dataset_generator(self, dataset: DatasetType, verbose: int = 0) -> Generator[Tuple[Dict[str, np.ndarray]], None, None]:
        """
        Generate data for a specific dataset.

        Args:
            dataset (DatasetType): The type of dataset to generate data for.
            verbose (int, optional): Verbosity mode. Defaults to 0.

        Yields:
            Generator[Tuple[Dict[str, np.ndarray]], None, None]: A generator that yields a tuple containing a dictionary of arrays.

        Raises:
            ValueError: If an invalid dataset type is provided.
        """
        if dataset == DatasetType.train or dataset == 0:
            X = self.global_generator.X_train
            X_l = self.regional_generator.X_l_train
            X_r = self.regional_generator.X_r_train
            X_s = self.local_generator.X_s_train
            X_d = self.local_generator.X_d_train
            y = self.y_train
        elif dataset == DatasetType.validation or dataset == 1:
            X = self.global_generator.X_val
            X_l = self.regional_generator.X_l_val
            X_r = self.regional_generator.X_r_val
            X_s = self.local_generator.X_s_val
            X_d = self.local_generator.X_d_val
            y = self.y_val
        elif dataset == DatasetType.test or dataset == 2:
            X = self.global_generator.X_test
            X_l = self.regional_generator.X_l_test
            X_r = self.regional_generator.X_r_test
            X_s = self.local_generator.X_s_test
            X_d = self.local_generator.X_d_test
            y = self.y_test
        else:
            raise ValueError('Invalid dataset type provided.')

        yield from self.data_generator(X, X_l, X_r, X_s, X_d, y, verbose=verbose)



class DataContrastiveGenerator(DataGenerator):
    
    @staticmethod
    def to_structure(X: np.ndarray, y: np.ndarray, add_batch_axis: bool = False) -> Tuple[Dict[str, np.ndarray]]:
        """
        Convert input data and labels into a structured format.

        Args:
            X (np.ndarray): The input data.
            y (np.ndarray): The labels.
            add_batch_axis (bool, optional): Whether to add a batch axis to the data and labels. Defaults to False.

        Returns:
            Tuple[Dict[str, np.ndarray]]: A tuple for the input and output containing a dictionary of arrays.
        """
        if add_batch_axis:
            X = np.expand_dims(X, axis=0) 
            y = np.expand_dims(y, axis=0)

        data = ({'contrastive_input': X},
                {'contrastive_loss': y,
                 'contrastive_output': y})

        return data
