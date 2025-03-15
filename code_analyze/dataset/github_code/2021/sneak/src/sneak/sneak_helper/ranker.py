"""This module is related to Ranker Class"""
# pylint: disable=import-error,invalid-name
import numpy as np


class Ranker:
    """This class is used for the following tasks:
    1. Ranking all the solutions
    2. Finding the current best node to ask further questions to the user
    3. Checking for the best solutions"""

    @staticmethod
    def level_rank_features(root, weights):
        """
            Function: level_rank_features
            Description: Function to build a tree of all the solutions
            Inputs:
                -root: TreeNode
                -weights: Array of weights from Method class object.
            Output:
                -items_rank : Solutions tree based on the ranking
        """
        if not root:
            return None
        items_rank = np.zeros(len(root.west.item))
        q = [[root, 1]]
        q_len = len(q)
        while q_len:
            p = q[0]
            q.pop(0)
            if p[0].west is not None and p[0].east is not None:
                diff = p[0].diff_array()
                for i, d in enumerate(diff):
                    if d and items_rank[i] == 0:
                        items_rank[i] = p[1] * weights[i]
            if p[0].west_node:
                q.append([p[0].west_node, p[1] + 1])
            if p[0].east_node:
                q.append([p[0].east_node, p[1] + 1])
            q_len = len(q)
        print(int(np.sum([1 for x in items_rank if x > 0])),
              "Total number of important questions")
        return items_rank

    @staticmethod
    def rank_nodes(root, rank):
        """
            Function: rank_nodes
            Description: Function to find out the current best node to ask human preferences
            Inputs:
                -root: TreeNode
                -rank: Rank value from Method class object.
            Output:
                -largest : Largest score among the questions
        """
        if not root:
            return None
        largest = -100000000
        q = [[root, 1]]
        q_len = len(q)
        while q_len:
            p = q[0]
            q.pop(0)
            if p[0].west is not None and p[0].east is not None:
                diff = p[0].diff_array()
                res = np.multiply(diff, rank)
                p[0].score = (np.sum(res) * p[0].weight) / np.sum(diff)
                if p[0].score > largest:
                    largest = p[0].score
            if p[0].west_node:
                q.append([p[0].west_node, p[1] + 1])
            if p[0].east_node:
                q.append([p[0].east_node, p[1] + 1])
            q_len = len(q)
        return largest

    @staticmethod
    def pr_level(root):
        """
            Function: pr_level
            Description: Function to find the preference level
            Inputs:
                -root: TreeNode
            Output:
                -tree_lvl : Array
        """
        tree_lvl = []
        if not root:
            return None
        q = [[root, 1]]
        q_len = len(q)
        while q_len:
            p = q[0]
            q.pop(0)
            if p[0].west is not None and p[0].east is not None:
                tree_lvl.append((p[1], p[0].difference()))
            if p[0].west_node:
                q.append([p[0].west_node, p[1] + 1])
            if p[0].east_node:
                q.append([p[0].east_node, p[1] + 1])
            q_len = len(q)
        return tree_lvl

    @staticmethod
    def check_solution(root):
        """
            Function: check_solution
            Description: Function to check the solution
            Inputs:
                -root: TreeNode
            Output:
                -number : Return 1/-1 values
        """
        count = 0
        if not root:
            return None
        q = [[root, 1]]
        q_len = len(q)
        while q_len:
            p = q[0]
            q.pop(0)
            if p[0].weight == 1 and p[0].leaf:
                count += 1
            if p[0].west_node:
                q.append([p[0].west_node, p[1] + 1])
            if p[0].east_node:
                q.append([p[0].east_node, p[1] + 1])
            q_len = len(q)
        print("Count =", count)
        if count == 1:
            return 1
        if count < 1:
            return -1
        return None
