"""This Module is related to io_helper_class"""
#pylint: disable=import-error
import pandas as pd
import os
import sys
cur_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(cur_dir)
import src.sneak.config.configparams as cfg


class InputOutput:
    """
    The InputOutput is used to perform the below mentioned IO stream related operations:
    1. Convert dimacs input to CNF format.
    2. Read the question text input from the csv file.
    """
    @staticmethod
    def read_dimacs(filename):
        """
        Function: read_dimacs
        Description: This function is created to read dimac format input and convert it to CNF form
        Input:
            - filename : File
        Output:
            - names : Array of features
            - cnf : Solution in CNF form
        """
        file = open(filename)
        lines = file.readlines()
        names = []
        for line in lines[:cfg.whunparams["NUM_FEATURES"]]:
            names.append(line.split(' ')[2][:-1])
        dimacs = lines[cfg.whunparams["NUM_FEATURES"] + 1:]
        cnf = [[int(s) for s in line.split(' ') if int(s) != 0]
               for line in dimacs]
        return names, cnf

    @staticmethod
    def get_question_text(filename, column):
        """
        Function : get_question_text
        Description : This function is created to read text input regarding questions
        Input:
            - filename : File
            - column : int
        Output:
            - column of dataframe : DataFrame
        """
        df = pd.read_csv(filename)
        return df[column].tolist()
