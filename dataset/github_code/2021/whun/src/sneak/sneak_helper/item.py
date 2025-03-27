"""This module is related to item_helper_class"""
# pylint: disable=import-error,invalid-name,too-many-instance-attributes
import os
import sys
cur_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(cur_dir)
import math
import secrets
import random
import time
import numpy as np
from src.sneak.config import configparams as cfg



class Item:
    """This class has the structure for each solution with all required parameters"""
    max_features = -math.inf
    min_features = math.inf
    # max_totalcost = -math.inf
    # min_totalcost = math.inf
    # max_known = -math.inf
    # min_known = math.inf
    max_featuresused = -math.inf
    min_featuresused = math.inf
    # costs = [secrets.randbelow(10) for _ in range(cfg.whunparams["NUM_FEATURES"])]
    # defective = [bool(secrets.randbelow(2)) for _ in range(cfg.whunparams["NUM_FEATURES"])]
    # used = [bool(secrets.randbelow(2)) for _ in range(cfg.whunparams["NUM_FEATURES"])]

    def __init__(self, item, eval):
        """
        Function : __init__
        Description : This is the constructor for item_helper_class class
        Input :
            - item : item
            - eval : Array
        """
        self.r = -1
        self.d = -1
        self.theta = -1
        self.item = item
        self.score = 0
        self.features = sum(item)
        # self.selectedpoints = 0
        # self.totalcost = sum(np.multiply(item, self.costs))
        # self.knowndefects = sum(np.multiply(item, self.defective))
        # self.featuresused = sum(np.multiply(item, self.used))
        self.n_mae = eval[0]
        self.n_mse = eval[1]
        self.n_rmse = eval[2]
        self.n_mape = eval[3]
        self.n_acc = eval[4]
        self.mae = eval[5]
        self.mse = eval[6]
        self.rmse = eval[7]
        self.mape = eval[8]
        self.acc = eval[9]
        self.n_estimators = eval[10]
        self.criterion = eval[11]
        self.min_samples_leaf = eval[12]
        self.min_impurity_decrease = eval[13]
        self.max_depth = eval[14]

    def better(self, other):
        east_cols = [self.n_mae, self.n_mse, self.n_rmse, self.n_mape,
                     self.n_acc]
        west_cols = [other.n_mae, other.n_mse, other.n_rmse, other.n_mape,
                     other.n_acc]
        s1, s2, n = 0, 0, len(east_cols)
        i = 0
        for e_col, b_col in zip(east_cols, west_cols):
            a = e_col
            b = b_col
            if i >= n-1:
                s1 -= math.e**(1 * (a - b) / n)
                s2 -= math.e**(1 * (b - a) / n)
            else:
                s1 -= math.e**(-1 * (a - b) / n)
                s2 -= math.e**(-1 * (b - a) / n)
            i += 1
        # To simulate a 1 second or more eval function add line below
        # time.sleep(1)
        return s1 / n < s2 / n

    def __lt__(self, other):
        return self.better(other)

    # @staticmethod
    # def calc_staticfeatures(items):
    #     """
    #     Function : calc_staticfeatures
    #     Description : This function updates the parameters related to static features
    #     Input:
    #         - items : item[]
    #     Output:
    #         - none
    #     """
    #     for x in items:
    #         if x.features > Item.max_features:
    #             Item.max_features = x.features
    #         if x.features < Item.min_features:
    #             Item.min_features = x.features
    #         if x.totalcost > Item.max_totalcost:
    #             Item.max_totalcost = x.totalcost
    #         if x.totalcost < Item.min_totalcost:
    #             Item.min_totalcost = x.totalcost
    #         if x.knowndefects > Item.max_known:
    #             Item.max_known = x.knowndefects
    #         if x.knowndefects < Item.min_known:
    #             Item.min_known = x.knowndefects
    #         if x.featuresused > Item.max_featuresused:
    #             Item.max_featuresused = x.featuresused
    #         if x.featuresused < Item.min_featuresused:
    #             Item.min_featuresused = x.featuresused

    @staticmethod
    def rank_features(items, names):
        """
        Function : rank_features
        Description :  This function is used to update the ranking parameters of all the features
        Input:
            - items : item[]
            - names : Array of attribute names
        Output:
            - count : int
            - rank : int
       """
        count = np.zeros(len(items[0].item))
        for item in items:
            count = np.add(count, item.item)
        rank = np.zeros(len(count))
        for i, v in enumerate(count):
            if v == 0:
                rank[i] = -1
                print("No", names[i])
            if v == (len(items)):
                rank[i] = -1
                print("All", names[i])
        return count, rank
