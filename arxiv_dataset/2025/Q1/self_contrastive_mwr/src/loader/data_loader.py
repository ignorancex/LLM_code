import re
from typing import Dict, Tuple, Optional

import pandas as pd
import numpy as np

from scipy.optimize import curve_fit


class DataLoader():
    
    def __init__(self, data_path: str):
        """
        Initialize the DataLoader object.

        Args:
            data_path (str): The path to the data file.

        Returns:
            None
        """
        if not data_path is None:
            self.data = pd.read_csv(data_path)
            self.data = DataLoader.prepare_data(self.data)
            self.data = DataLoader.drop_nans(self.data)
            self.data = DataLoader.cast_types(self.data)
            
        self.cancer = 'Cancer'
        self.l_values = ['L{}'.format(i) for i in range(9)]
        self.r_values = ['R{}'.format(i) for i in range(9)]
        self.l_skin_values = ['Skin {}'.format(i) for i in self.l_values]
        self.r_skin_values = ['Skin {}'.format(i) for i in self.r_values]
        self.l_axillary_value = 'L9'
        self.r_axillary_value = 'R9'
        self.l_axillary_skin_value = 'Skin ' + self.l_axillary_value
        self.r_axillary_skin_value = 'Skin ' + self.r_axillary_value
        self.ref_values = ['T{}'.format(i) for i in range(2)]
        self.ref_skin_values = ['Skin {}'.format(i) for i in self.ref_values]
        
    
    @staticmethod
    def prepare_data(df: pd.DataFrame, show_menopause: bool = True) -> pd.DataFrame:
        """
        Preprocesses the given DataFrame by filling missing values and transforming the 'Cycle' and 'Day of Cycle' columns.

        Args:
            df (pd.DataFrame): The input DataFrame to be processed.
            show_menopause (bool, optional): Whether to show menopause feature. Defaults to True.

        Returns:
            pd.DataFrame: The processed DataFrame.
        """
        df['Ambient'] = df['Ambient'].fillna((df['Ambient'].mean()))
        df['Breast Diameter'] = df['Breast Diameter'].fillna((df['Breast Diameter'].mean()))

        pattern = re.compile("^([0-9]+-[0-9]+)+$")

        for index, row in df.iterrows():
            # Cycle
            if pd.isnull(row['Cycle']) or row['Cycle'] == '' or row['Cycle'] == '-':
                if show_menopause:
                    df.at[index, 'Cycle'] = -1
                else:
                    df.at[index, 'Cycle'] = 28
            elif pattern.match(row['Cycle']):
                values = row['Cycle'].split('-')
                average = int((int(values[0]) + int(values[1])) * 0.5)
                df.at[index, 'Cycle'] = average

            # Day of Cycle
            if pd.isnull(row['Day of Cycle']) or row['Day of Cycle'] == '' or row['Day of Cycle'] == '-':
                if show_menopause:
                    df.at[index, 'Day of Cycle'] = -1
                else:
                    df.at[index, 'Day of Cycle'] = 5

        return df
            
    
    @staticmethod
    def cast_types(df: pd.DataFrame) -> pd.DataFrame:
        """
        Casts the data types of specific columns in the DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: The DataFrame with updated data types.
        """
        df = df.astype({'Age': int,
                        'Cancer': int,
                        'Breast Diameter': int,
                        'Cycle': int,
                        'Day of Cycle': int,
                        #'Cycle_Stage': int,
                        'Ambient': int})
            
        return df
        
    
    @staticmethod
    def compute_cycle(df: pd.DataFrame, groups: float) -> pd.DataFrame:
        """
        Compute the cycle stage for each row in the given DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            groups (float): The number of groups to divide the cycle stage into.

        Returns:
            pd.DataFrame: The DataFrame with an additional column 'Cycle_Stage' representing the cycle stage for each row.
        """
        def compute(cycle: int, cycle_day: int) -> float:
            if cycle < 0 or cycle_day < 0:
                return -1 
            
            # For percentage purposes do not let the cycle day exceed the expected cycle length
            cycle_day = min(cycle, cycle_day)
            
            cycle_stage = cycle_day / cycle
            if groups:
                cycle_stage = int(cycle_stage // groups)
                
            return cycle_stage
        
        df['Cycle_Stage'] = df.apply(lambda x: compute(int(x['Cycle']), int(x['Day of Cycle'])), axis=1)
        
        return df
    
        
    @staticmethod
    def noramalise(df: pd.DataFrame, label_tag: str, ref_label: str) -> pd.DataFrame:
        """
        Normalize the temperature data in the given DataFrame using a linear transformation
        based on the reference temperature values.

        Args:
            df (pd.DataFrame): The DataFrame containing the temperature data.
            label_tag (str): The tag used to identify the temperature columns in the DataFrame.
            ref_label (str): The label of the reference temperature column.

        Returns:
            pd.DataFrame: The DataFrame with the normalized temperature data.
        """
        def line_function(x, A, B):
            return A * x + B

        def transform(temperature, A, refAvg, ref):
            return temperature + A * (refAvg - ref)
        
        
        ref_mean = df[ref_label].mean(axis=0)
        
        for i in range(10):
            label = label_tag + str(i)
            A, B = curve_fit(line_function, df[ref_label].values, df[label].values)[0]
            df[label] = np.vectorize(transform)(df[label], A, ref_mean, df[ref_label])
            
        return df
    
    
    @staticmethod
    def select_cancer_cases(df: pd.DataFrame, has_cancer: bool) -> pd.DataFrame:
        """
        Selects cases from the given DataFrame based on the presence or absence of cancer.

        Args:
            df (pd.DataFrame): The DataFrame containing the cases.
            has_cancer (bool): A boolean value indicating whether to select cases with cancer (True) or without cancer (False).

        Returns:
            pd.DataFrame: A new DataFrame containing the selected cases.
        """
        cancer_label = 1 if has_cancer else 0
        return df.loc[df['Cancer'] == cancer_label]
    
    
    @staticmethod
    def drop_nans(df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop rows with missing values from a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to drop rows from.

        Returns:
            pd.DataFrame: The DataFrame with rows containing missing values dropped.
        """
        return df.dropna(axis=0, how='any')
        
    
    def select_columns(self, df: pd.DataFrame, l_values: bool = False,
                       r_values: bool = False, l_skin_values: bool = False,
                       r_skin_values: bool = False, l_axil_value: bool = False,
                       r_axil_value: bool = False, l_axil_skin_value: bool = False,
                       r_axil_skin_value: bool = False, age: bool = False,
                       ref_values: bool = False, ref_skin_values: bool = False,
                       cancer_values: bool = False) -> pd.DataFrame:
        """
        Selects specific columns from a DataFrame based on the provided arguments.

        Args:
            df (pd.DataFrame): The input DataFrame.
            l_values (bool, optional): Whether to include the left depth breast temperatures. Defaults to False.
            r_values (bool, optional): Whether to include the right depth breast temperatures. Defaults to False.
            l_skin_values (bool, optional): Whether to include the left skin breast temperatures. Defaults to False.
            r_skin_values (bool, optional): Whether to include the right skin breast temperatures. Defaults to False.
            l_axil_value (bool, optional): Whether to include the left axillary temperature. Defaults to False.
            r_axil_value (bool, optional): Whether to include the right axillary temperature. Defaults to False.
            l_axil_skin_value (bool, optional): Whether to include the left axillary skin temperature. Defaults to False.
            r_axil_skin_value (bool, optional): Whether to include the right axillary skin temperature. Defaults to False.
            age (bool, optional): Whether to include the age. Defaults to False.
            ref_values (bool, optional): Whether to include the reference temperatures. Defaults to False.
            ref_skin_values (bool, optional): Whether to include the reference skin temperatures. Defaults to False.
            cancer_values (bool, optional): Whether to include the 'cancer' column. Defaults to False.

        Returns:
            pd.DataFrame: A new DataFrame containing only the selected columns.
        """
        column_selection = []
        
        if age:
            column_selection.append('Age')
        if l_values:
            column_selection.extend(self.l_values)
        if r_values:
            column_selection.extend(self.r_values)
        if l_skin_values:
            column_selection.extend(self.l_skin_values)
        if r_skin_values:
            column_selection.extend(self.r_skin_values)
        if l_axil_value:
            column_selection.append(self.l_axillary_value)
        if r_axil_value:
            column_selection.append(self.r_axillary_value)
        if l_axil_skin_value:
            column_selection.append(self.l_axillary_skin_value)
        if r_axil_skin_value:
            column_selection.append(self.r_axillary_skin_value)
        if ref_values:
            column_selection.extend(self.ref_values)
        if ref_skin_values:
            column_selection.extend(self.ref_skin_values)
        if cancer_values:
            column_selection.append(self.cancer)
        
        return df[column_selection]
    
    @staticmethod
    def shuffle(df: pd.DataFrame, seed: Optional[int] = None) -> pd.DataFrame:
        """
        Shuffle the rows of a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to be shuffled.
            seed (int, optional): The random seed for shuffling. Defaults to None.

        Returns:
            pd.DataFrame: The shuffled DataFrame.
        """
        df = df.shuffle(frac=1).reset_index(drop=True)
        return df
    
    
    
class DataLoaderSelector():
    
    def __init__(self, data_path: str = 'dataset.csv'):
        """
        Initialize the DataLoader class.

        Args:
            data_path (str, optional): The path to the dataset file. Defaults to 'dataset.csv'.
        
        Returns:
            None
        """
        self.data_loader = DataLoader(data_path)
    
    
    @staticmethod
    def _select_data(gland: str = 'both', surface: str = 'both',
                     use_ref_values: bool = True, use_axillary_values: bool = True,
                     use_age: bool = False) -> Dict[str, bool]:
        """
        Returns the appropriate data selection criteria based on the input.

        Args:
            gland (str, optional): The gland to select data from. Valid values are 'both' (default),
                                   'left' or 'l', and 'right' or 'r'.
            surface (str, optional): The surface to select data from. Valid values are 'both' (default),
                                     'skin' or 's', and 'depth' or 'd'.
            use_ref_values (bool, optional): Whether to use reference values. Defaults to True.
            use_axillary_values (bool, optional): Whether to use axillary values. Defaults to True.
            use_age (bool, optional): Whether to use age. Defaults to False.

        Raises:
            ValueError: If invalid values are provided for the 'gland' or 'surface' arguments.

        Returns:
            Dict[str, bool]: A dictionary containing the selected data criteria.
        """
        input_keys = dict(locals())
        
        gland = gland.lower()
        surface = surface.lower()
        
        if surface == 'skin' or surface == 's':
            l_values = False
            r_values = False
            ref_values = False
            l_axil_value = False
            r_axil_value = False
            l_skin_values = True
            r_skin_values = True
            ref_skin_values = True
            l_axil_skin_value = True
            r_axil_skin_value = True
        elif surface == 'depth' or surface == 'd':
            l_values = True
            r_values = True
            ref_values = True
            l_axil_value = True
            r_axil_value = True
            l_skin_values = False
            r_skin_values = False
            ref_skin_values = False
            l_axil_skin_value = False
            r_axil_skin_value = False
        elif surface == 'both' or surface == 'b':
            l_values = True
            r_values = True
            ref_values = True
            l_axil_value = True
            r_axil_value = True
            l_skin_values = True
            r_skin_values = True
            ref_skin_values = True
            l_axil_skin_value = True
            r_axil_skin_value = True
        else:
            raise ValueError("f{surface} is not a valid input for 'surface' argument") 
        
        
        if gland == 'both' or gland == 'b': # values will be as set by surface condition
            pass
        elif gland == 'left' or gland == 'l':
            r_values = False
            r_skin_values = False
            r_axil_value = False
            r_axil_skin_value = False
        elif gland == 'right' or gland == 'r':
            l_values = False
            l_skin_values = False
            l_axil_value = False
            l_axil_skin_value = False
        else:
            ValueError("f{gland} is not a valid input for 'gland' argument") 
        
        if not use_ref_values:
            ref_values = False
            ref_skin_values = False
        
        if not use_axillary_values:
            l_axil_value = False
            l_axil_skin_value = False
            r_axil_value = False
            r_axil_skin_value = False
        
        if use_age:
            age = True
        else:
            age = False
        
        selections = dict(locals())
        # Remove input variables
        for key in input_keys:
            del selections[key]
        del selections['input_keys']
        
        return selections
    
    
    def get_data(self, gland: str = 'both', surface: str = 'both',
                 use_ref_values: bool = True, use_axillary_values: bool = True,
                 use_age: bool = False) -> Tuple[pd.DataFrame]:
        """
        Get the data based on the specified parameters.

        Args:
            gland (str, optional): The type of gland to include in the data. Defaults to 'both'.
            surface (str, optional): The type of surface to include in the data. Defaults to 'both'.
            use_ref_values (bool, optional): Whether to include reference values in the data. Defaults to True.
            use_axillary_values (bool, optional): Whether to include axillary values in the data. Defaults to True.
            use_age (bool, optional): Whether to include age in the data. Defaults to False.

        Returns:
            Tuple[pd.DataFrame]: A tuple containing the X and y dataframes.
        """
        selections = DataLoaderSelector._select_data(gland=gland, surface=surface,
                                                        use_ref_values=use_ref_values,
                                                        use_axillary_values=use_axillary_values,
                                                        use_age=use_age)
        
        X = self.data_loader.select_columns(self.data_loader.data,
                                            **selections,
                                            cancer_values=False)
        
        y = self.data_loader.select_columns(self.data_loader.data, l_values=False,
                                            r_values=False, l_skin_values=False, r_skin_values=False,
                                            age=False, ref_values=False, ref_skin_values=False,
                                            cancer_values=True)
            
        return X, y
