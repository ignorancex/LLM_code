"""This module is related to TreeNode class"""
from itertools import count

import numpy as np


class TreeNode:
    """ This class is to initialise the TreeNode and to perform difference operations """
    _ids = count(0)

    def __init__(self, east, west, east_node, west_node, leaf):
        """
        Function: __init__
        Description: Initialises the attributes of the TreeNode
        Inputs:
                east :Item
                west :Item
                east_node :TreeNode
                west_node  :treeNode
                leaf : boolean
        Output:
                self initialised with attributes
        """
        self.id = next(self._ids)
        self.east = east
        self.west = west
        self.east_id = -1
        self.west_id = -1
        self.east_node = east_node
        self.west_node = west_node
        self.score = 0
        self.weight = 1
        self.asked = 0
        self.leaf = leaf

    def difference(self):
        """
        Function: difference
        Description: Returns the difference of east and west items
        Inputs:
                self :TreeNode
        Output:
                np.sum(res) :Sum of elements in res,Numpy array
        """
        w = np.array(self.west.item)
        e = np.array(self.east.item)
        res = np.logical_xor(w, e)
        return np.sum(res)

    def diff_array(self):
        """
        Function: diff_array
        Description: Returns the difference array of east and west items
        Inputs:
                self :TreeNode
        Output:
                res :Numpy Array
        """
        w = np.array(self.west.item)
        e = np.array(self.east.item)
        res = np.logical_xor(w, e)
        return res
