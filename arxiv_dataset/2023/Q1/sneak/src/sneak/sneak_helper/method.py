"""This module is related to Method Class"""
# pylint: disable=import-error
import os
import sys
cur_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(cur_dir)
import scipy.stats as st
import numpy as np
import math
from src.sneak.config import configparams as cfg
from src.sneak.utils.utils import sway
from src.sneak.sneak_helper.sat_solver import SatSolver
from src.sneak.sneak_helper.search import Search
from src.sneak.sneak_helper.ranker import Ranker
from src.sneak.sneak_helper.input_output import InputOutput
cur_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(cur_dir)


class Method:
    """
    This class is used store the sat solver solutions in form of a tree,
    ranking of each node down to its feature and then handles the weight adjustments,
    re-ranking tree based on the user preferences and finding the corresponding
    solutions.
    """

    def __init__(self, filename, eval_file, prefix_path=""):
        try:
            sys.setrecursionlimit(cfg.whunparams["RECURSION_LIMIT"])
            self.items = SatSolver.get_solutions(filename, eval_file, prefix_path)
            self.weights = [1] * len(self.items)
            self.tree = sway(self.items, 100)
            self.names = []
            self.rank = Ranker.level_rank_features(self.tree, self.weights)
            self.cur_best_node = Ranker.rank_nodes(self.tree, self.rank)
            # IO.get_question_text('terms_sentence_map.csv', 'sentence')
            # self.questions = InputOutput.get_question_text(
            #     cur_dir + '/' + cfg.whunparams["FOLDER"] + 'terms_sentence_map.csv', 'sentence')
        except Exception as e:
            print(e)
            raise e

    def find_node(self):
        """
        Function: find_node
        Description: creates a queue of trees and returns path_id and
        node based on tree and current best node
        Inputs:
        Output:
            -path_id: path
            -node: last node in the path
        """
        return Search.bfs(self.tree, self.cur_best_node)

    def pick_questions(self, node):
        """
        Function: pick_questions
        Description: returns questions associated with this node
        Inputs:
            -self: method object
            -node: item node
        Output:
            -questions: array of questions
        """
        diff = node.diff_array()
        ranked_ranks = sorted(self.rank, reverse=True)
        ranks = [x for x in ranked_ranks if x != 0]
        ranks.reverse()
        ranks = set(ranks)
        return self.get_index(diff, ranks)

    def adjust_tree(self, node, q_idx):
        """
        Function: adjust_tree
        Description: adjusts the current node of tree based on picked questions
        Inputs:
            -self: method object
            -node: item node
            -q_idx: picked questions
        Output:
        """
        if node.leaf:
            return
        if node.west:
            for i in q_idx:
                if i in node.west.item:
                    node.west_node.weight = 0
                    self.adjust_tree(node.west_node, q_idx)
        if node.east:
            for i in q_idx:
                if i in node.east.item:
                    node.east_node.weight = 0
                    self.adjust_tree(node.east_node, q_idx)

    def get_index(self, diff, ranks):
        """
        Function: get_index
        Description:
        Inputs:
            -self: method object
            -diff:
            -ranks:
        Output:
            questions: questions specific to this nodes
        """
        questions = []
        for rank in ranks:
            for i, value in enumerate(diff):
                if rank == self.rank[i] and value:
                    questions.append(i)
        return questions

    def process_options(self, left_branch, right_branch):
        """
            Function: process_options
            Description: Given east and west branch, the method will check if both the branches have valid nodes or not.
            If one of the branch doesn't have valid nodes it will return the other brancch as selectedd preference.
            Inputs:
                -self: method object
                -left_branch: East Branch
                -right_branch: West Branch
            Output:
                - 0: If east branch doesn't have any valid nodes, we send 0 signifying that west branch needs to be selected.
                - 1: If west branch doesn't have any valid nodes, we send 1 signifying that east branch needs to be selected.
                - -1: If both the branches have valid nodes, we send -1 signifying that human needs to select his preference.
        """
        count = 0
        for item in left_branch:
            if len(item) == item.count(" "):
                count += 1
        if count == len(left_branch):
            return 0
        count = 0
        for item in right_branch:
            if len(item) == item.count(" "):
                count += 1
        if count == len(right_branch):
            return 1
        return -1

    def ask_questions(self, q_idx, node, ui_obj):
        """
       Function: ask_questions
       Description: Function to take human preference on whether to select east branch or west branch from the picked set of questions.
       Inputs:
           -self: method object
           -q_idx: picked questions
           -node: item node
       Output:
            None
       """
        east_options, west_options = [], []
        for _, value in enumerate(q_idx):
            if node.east.item[value]:
                east_options.append(self.questions[value])
            elif node.west.item[value]:
                west_options.append(self.questions[value])

        len_east = len(east_options)
        len_west = len(west_options)
        diff = abs(len_east - len_west)
        if len_east > len_west:
            for _ in range(diff):
                west_options.append('           ')
        else:
            for _ in range(diff):
                east_options.append('           ')
        processed_value = self.process_options(east_options, west_options)
        if processed_value >= 0:
            return processed_value
        return ui_obj.show_options_dialog(east_options, west_options)

    def adjust_weights(self, node, picked, q_idx):
        """
        Function: adjust_weights
        Description: adjusts the weights in tree based on picked question
        Inputs:
            -self: method object
            -node: item node
            -picked: picked question
            -q_idx: all questions related to node
        Output:
        """
        node.asked += 1
        east_options, west_options = [], []
        for _, value in enumerate(q_idx):
            if node.east.item[value]:
                east_options.append(value)
            elif node.west.item[value]:
                west_options.append(value)
        if picked == 0:  # EAST
            for _, value in enumerate(q_idx):
                self.weights[value] = 0
            self.adjust_down(node.west_node)
            self.adjust_tree(self.tree, west_options)
        if picked == 1:  # WEST
            for _, value in enumerate(q_idx):
                self.weights[value] = 0
            self.adjust_down(node.east_node)
            self.adjust_tree(self.tree, east_options)

    def adjust_down(self, node, depth=0):
        """
        Function: increment the depth of east_node and west_node if its not None
        Description:
        Inputs:
            -self: method object
            -node: item node
            -depth: Integer, picked question
        Output:
        """
        # weight = 0
        node.weight = 0
        if node.east_node is not None and node.west_node is not None:
            self.adjust_down(node.east_node, depth + 1)
            self.adjust_down(node.west_node, depth + 1)

    def re_rank(self):
        """
        Function: re_rank
        Description: rerank the tree based on current state of the method object
        Inputs:
            -self: method object
        Output:
        """
        self.rank = Ranker.level_rank_features(self.tree, self.weights)
        self.cur_best_node = Ranker.rank_nodes(self.tree, self.rank)

    def check_solution(self):
        """
        Function: check_solution
        Description: finds all the possible solutions in the tree
        Inputs:
            -self: method object
        Output:
        """
        if sum(self.rank) == 0:
            return Search.get_all_items(self.tree)
        value = Ranker.check_solution(self.tree)
        if value is None:
            return None
        if value == -1:
            return -1
        return Search.get_all_items(self.tree)

    def get_item(self, path):
        """
        Function: get_item
        Description: return item based on the input path
        Inputs:
            -self: method object
            -path: item node
        Output:
            -item: item object
        """
        return Search.get_item(self.tree, path)

    def pick_best(self, solutions, evaluations):
        """
        Function: pick_best
        Description: picks the best solution among all the possible solutions
        Inputs:
            -self: method object
            -solutions: item node
        Output:
            solution:
        """
        all_items = Search.get_all_leaves(self.tree)
        solutions.sort()
        evaluations += (len(solutions)*math.log(len(solutions), 2))
        all_items.sort()
        best = solutions[0]
        best.score = 1
        for index, item in enumerate(all_items):
            if best == item:
                best.score = index/float(len(all_items))

        return best, evaluations

    def calculate_score(self, selected_solution):
        """
        Function: calculate_score
        Description: calculates the score of a randomly selected solution
        Inputs:
            -self: method object
            -solutions: item nodes
            -selected_solution: item node
        Output:
            selected_solution: item node with caculated score
        """
        all_items = Search.get_all_leaves(self.tree)
        all_items.sort()
        selected_solution.score = 1
        for index, item in enumerate(all_items):
            if selected_solution == item:
                selected_solution.score = index/float(len(all_items))
        return selected_solution
