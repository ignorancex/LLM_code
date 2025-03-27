""" This class contains utilities to provide the predictions file.
"""
import pandas as pd
import os


class CsvResults:
    def __init__(self, cols_headers=None):
        """ Initializes class fields.
        """
        if cols_headers is None:
            cols_headers = ['Configuration ID', 'Container capacity', 'Container mass', \
                        'Filling mass', 'None', 'Pasta', 'Rice', 'Water', 'Filling type', 'Empty', \
                        'Half-full', 'Full', 'Filling level', 'Width at the top', 'Width at the bottom', \
                        'Height', 'Object safety', 'Distance', 'Angle difference', 'Execution time']
        self.pred_dict = {key: [] for key in cols_headers}

    def fill_entry(self, col_name, value):
        """ Fills an entry of the csv file.

        :param col_name: name of the column of the csv file
        :param value: value to fill the entry
        :return: None
        """
        self.pred_dict[col_name].append(value)
        return

    def fill_other_entries(self, col_list, value):
        """ Fills an entry of the csv file.

        :param col_list: list with name of the columns of the csv file
        :param value: value to fill the entries
        :return: None
        """
        for col_name in col_list:
            self.pred_dict[col_name].append(value)
        return
    
    def save_csv(self, path_to_dest):
        """ Saves csv file.

        :param path_to_dest: path to the destination file (including the name of the file)
        :param gt_json: dataframe containing data of the eventual csv file
        :return: None
        """
        df = pd.DataFrame(self.pred_dict)
        df.to_csv(path_to_dest, index=False)
        print("{} successfully created!!".format(os.path.basename(path_to_dest)))
