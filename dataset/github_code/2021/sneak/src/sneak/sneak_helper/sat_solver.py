"""This module is related to SAT Solver Class"""
# pylint: disable=import-error,invalid-name
import os
import sys
cur_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(cur_dir)
from csv import reader
import pandas as pd
from src.sneak.sneak_helper.item import Item

sys.path.append('/whun_helper')


class SatSolver:
    """This class is used for getting Items"""
    @staticmethod
    def get_solutions(cnf, eval_file, prefix_path=""):
        """
        Function: get_solutions
        Description: Takes CNF and evaluation metrics and returns list of Item class objects
        Inputs:
            -cnf:String
            -eval_file:String
        Output:
            -items:Item
        """
        #global folder
        #global eval_file
        evals = pd.read_csv(prefix_path + eval_file).to_numpy()

        with open(cnf, 'r') as read_obj:
            binary_solutions = [[int(x) for x in rec]
                                for rec in reader(read_obj, delimiter=',')]
            items = []
            for i, item in enumerate(binary_solutions):
                items.append(Item(item, evals[i]))
            return items
