from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from loader.data_loader import DataLoader


class Augmentation():
    
    def __init__(self):
        """
        Initializes the Augmentation class.

        Args:
            None

        Returns:
            None
        """
        data_loader = DataLoader(data_path=None)
        
        # Get the expected label names of the dataset
        self.cancer = data_loader.cancer
        self.l_values = data_loader.l_values
        self.r_values = data_loader.r_values
        self.l_skin_values = data_loader.l_skin_values
        self.r_skin_values = data_loader.r_skin_values
        self.l_axillary_value = data_loader.l_axillary_value
        self.r_axillary_value = data_loader.r_axillary_value
        self.l_axillary_skin_value = data_loader.l_axillary_skin_value
        self.r_axillary_skin_value = data_loader.r_axillary_skin_value
        self.ref_values = data_loader.ref_values
        self.ref_skin_values = data_loader.ref_skin_values
        
    
    @staticmethod
    def rotate(data: pd.DataFrame, labels: List[str], rotate_by: int) -> pd.DataFrame:
        """
        Rotate (circular-list) the columns of a DataFrame by a given number of positions.

        Args:
            data (pd.DataFrame): The input DataFrame.
            labels (List[str]): The list of column labels to rotate.
            rotate_by (int): The number of positions to rotate the columns by.

        Returns:
            pd.DataFrame: The DataFrame with rotated columns.
        """
        columns = data[labels].to_numpy()
        shifted_columns = np.roll(columns, rotate_by, axis=-1)
        data[labels] = shifted_columns

        return data
    
    
    @staticmethod
    def reverse(data: pd.DataFrame, labels: List[str]) -> pd.DataFrame:
        """
        Reverse the order of columns in a DataFrame.

        Args:
            data (pd.DataFrame): The input DataFrame.
            labels (List[str]): The list of column labels to reverse.

        Returns:
            pd.DataFrame: The DataFrame with reversed columns.
        """
        columns = data[labels].to_numpy()
        reverse_columns = columns[..., ::-1]
        data[labels] = reverse_columns
        
        return data
    
    
    @staticmethod
    def flip(data: pd.DataFrame, labels_a: List[str], labels_b: List[str]) -> pd.DataFrame:
        """
        Flip the values between two sets of labels in a DataFrame.

        Args:
            data (pd.DataFrame): The input DataFrame.
            labels_a (List[str]): The labels of the first set of columns to flip.
            labels_b (List[str]): The labels of the second set of columns to flip.

        Returns:
            pd.DataFrame: The DataFrame with the values between the two sets of labels flipped.
        """
        columns_a = data[labels_a].to_numpy()
        columns_b = data[labels_b].to_numpy()
        data[labels_a] = columns_b
        data[labels_b] = columns_a
        
        return data
        


class AugmentData(Augmentation):
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the Augmentation class.

        Args:
            seed (int, optional): The seed value for the random number generator. Defaults to None.
        """
        super().__init__()
        
        self.random_generator = np.random.RandomState(seed)
        self.rotate_range = [0, len(self.l_values) - 1]
    
    
    def rotate(self, data: pd.DataFrame, rotate_by: int) -> pd.DataFrame:
        """
        Rotate the given data by the specified amount.

        Args:
            data (pd.DataFrame): The input data to be rotated.
            rotate_by (int): The amount by which to rotate the data.

        Returns:
            pd.DataFrame: The rotated data.
        """
        data = Augmentation.rotate(data, self.l_values, rotate_by)
        data = Augmentation.rotate(data, self.r_values, rotate_by)
        data = Augmentation.rotate(data, self.l_skin_values, rotate_by)
        data = Augmentation.rotate(data, self.r_skin_values, rotate_by)
        
        return data
        
        
    def random_rotate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Randomly rotates the given data by a random step within the specified range.

        Args:
            data (pd.DataFrame): The input data to be rotated.

        Returns:
            pd.DataFrame: The rotated data.
        """
        rotate_by = self.random_generator.randint(low=self.rotate_range[0],
                                                  high=self.rotate_range[1] + 1,
                                                  dtype=np.int32)
        
        data = self.rotate(data, rotate_by)
        
        return data

    
    def random_reverse(self, data: pd.DataFrame, independent: bool) -> pd.DataFrame:
        """
        Randomly reverses the given data.

        Args:
            data (pd.DataFrame): The input data to be reversed.
            independent (bool): Flag indicating whether the reversal should be independent for each side.

        Returns:
            pd.DataFrame: The reversed data.

        """
        should_reverse = self.random_generator.choice([False, True],
                                                      replace=False)
        
        if should_reverse:
            data = self.reverse(data, self.l_values)
            data = self.reverse(data, self.l_skin_values)
        
        if independent:
            should_reverse = self.random_generator.choice([False, True],
                                                          replace=False)
        
        if should_reverse:
            data = self.reverse(data, self.r_values)
            data = self.reverse(data, self.r_skin_values)
        
        
        return data
    
    
    def flip(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Flips the data between left and right side.

        Args:
            data (pd.DataFrame): The input data to be flipped.

        Returns:
            pd.DataFrame: The flipped data.
        """
        data = Augmentation.flip(data, self.l_values, self.r_values)
        data = Augmentation.flip(data, self.l_skin_values, self.r_skin_values)
        data = Augmentation.flip(data, self.l_axillary_value, self.r_axillary_value)
        data = Augmentation.flip(data, self.l_axillary_skin_value, self.r_axillary_skin_value)
    
        return data


    def random_flip(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Randomly flips the given DataFrame.

        Args:
            data (pd.DataFrame): The input DataFrame to be flipped.

        Returns:
            pd.DataFrame: The flipped DataFrame.
        """
        should_flip = self.random_generator.choice([False, True],
                                                    replace=False)
        
        if should_flip:
            data = self.flip(data)
        
        return data
    
    
    
class AugmentConstrastiveData(Augmentation):
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the Augmentation class.

        Args:
            seed (int, optional): The seed value for the random number generator. Defaults to None.
        """
        super().__init__()
        self.random_generator = np.random.RandomState(seed)
        
        self.rotate_range = [0, len(self.l_values) - 1]
    
    
    def random_rotate(self, data_l: pd.DataFrame, data_r: pd.DataFrame,
                      independent: bool) -> Tuple[pd.DataFrame]:
        """
        Randomly rotates the given DataFrames.

        Args:
            data_l (pd.DataFrame): The left data frame to be rotated.
            data_r (pd.DataFrame): The right data frame to be rotated.
            independent (bool): Specifies whether the rotation should be independent between the two DataFrame.

        Returns:
            Tuple[pd.DataFrame]: A tuple containing the rotated left and right DataFrames.
        """
        rotate_by = self.random_generator.integers(low=self.rotate_range[0],
                                                   high=self.rotate_range[1],
                                                   dtype=np.int32,
                                                   endpoint=True)
        
        data_l = self.rotate(data_l, self.l_values, rotate_by)
        data_l = self.rotate(data_l, self.l_skin_values, rotate_by)
        
        # If rotation is independent between the two
        if independent:
            rotate_by = self.random_generator.integers(low=self.rotate_range[0],
                                                       high=self.rotate_range[1],
                                                       dtype=np.int32,
                                                       endpoint=True)
        
        data_r = self.rotate(data_r, self.r_values, rotate_by)
        data_r = self.rotate(data_r, self.r_skin_values, rotate_by)
        
        return data_l, data_r
    
    
    def random_reverse(self, data_l: pd.DataFrame, data_r: pd.DataFrame,
                       independent: bool) -> Tuple[pd.DataFrame]:
        """
        Randomly reverses the DataFrames.

        Args:
            data_l (pd.DataFrame): The left dataframe to be reversed.
            data_r (pd.DataFrame): The right dataframe to be reversed.
            independent (bool): Flag indicating whether the reversal should be independent for DataFrame.

        Returns:
            Tuple[pd.DataFrame]: A tuple containing the reversed left and right DataFrames.
        """
        need_reverse = self.random_generator.choice([False, True],
                                                    replace=False)
        
        if need_reverse:
            data_l = self.reverse(data_l, self.l_values)
            data_l = self.reverse(data_l, self.l_skin_values)
        
        if independent:
            need_reverse = self.random_generator.choice([False, True],
                                                        replace=False)
        
        if need_reverse:
            data_r = self.reverse(data_r, self.r_values)
            data_r = self.reverse(data_r, self.r_skin_values)
        
        
        return data_l, data_r
    
    
    def random_flip(self, data_l: pd.DataFrame, data_r: pd.DataFrame) -> Tuple[pd.DataFrame]:
        """
        Randomly flips the data in the left and right DataFrames.

        Args:
            data_l (pd.DataFrame): The left DataFrame.
            data_r (pd.DataFrame): The right DataFrame.

        Returns:
            Tuple[pd.DataFrame]: A tuple containing the modified left and right DataFrames.
        """
        should_flip = self.random_generator.choice([False, True],
                                                    replace=False)
        
        if should_flip:
            labels_l = data_l.columns
            labels_r = data_r.colunms
            tmp = data_l[labels_l].to_numpy().copy()
            data_l[labels_l] = data_r[labels_r].to_numpy()
            data_r[labels_r] = tmp
        
        return data_l, data_r