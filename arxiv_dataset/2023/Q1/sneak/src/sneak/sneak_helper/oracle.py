"""This module is related to Oracle class"""
# pylint: disable=import-error,invalid-name,too-few-public-methods
import random


class Oracle:
    """
    This class is used to perform the human interaction automatically.
    Oracle is presented with the questions and the preferences for those questions.
    It will randomly choose the preferences everytime. So it is used to replace human interactions.
    """

    def __init__(self, size):
        """
            Function: constructor
            Description: Initializes the class object attributes with initial values
            Inputs:
                -size: number
            Output:
                None
        """
        self.picked = [0] * size

    def pick(self, q_idx, node):
        """
            Function: pick
            Description: Function to find a random preference value for a question inorder to do the human interaction automatically
            Inputs:
                -q_idx: List of indices of questions
                -node: TreeNode
            Output:
                -selected : Either 1 or 0 based on the condition if the preference is selected or not.
        """
        west_points = 0
        east_points = 0
        # Check how many of these questions I have picked before
        q_idx_len = len(q_idx)
        for i in range(q_idx_len):
            if node.east.item[q_idx[i]] and self.picked[q_idx[i]] == 1:
                east_points += 1
            elif node.west.item[q_idx[i]] and self.picked[q_idx[i]] == 1:
                west_points += 1
        # Random selection favoring the side i like the most
        if east_points + west_points > 0:
            weighted_selection = west_points / (east_points + west_points)
        else:
            weighted_selection = 0.5

        if random.random() < weighted_selection:
            selected = 0
        else:
            selected = 1
        # Update my vector of picked options
        self.update_picked_array(selected, q_idx, node)
        # Return selected {0 = East, 1 = West}
        return selected

    def update_picked_array(self, selected, q_idx, node):
        """
            Function: update_picked_array
            Description: Function to update picked array based on the corresponding attributes that are selected.
            Inputs:
                -selected: Value that determines if the east branch is selected or the west branch is selected.
                -q_idx: List of indices of questions
                -node: TreeNode
            Output:
                -selected : Either 1 or 0 based on the condition if the preference is selected or not.
        """
        # Update my vector of picked options
        for i in range(min(len(q_idx), 4)):
            if selected and self.picked[q_idx[i]] == 0:
                self.picked[q_idx[i]] = node.east.item[q_idx[i]]
            elif not selected and self.picked[q_idx[i]] == 0:
                self.picked[q_idx[i]] = node.west.item[q_idx[i]]
        return self.picked

    def evalItems(self, east, west):
        """
            Function: evalItems
            Description: Function to evaluate the items based on their scores.
            Inputs:
                -east: Represents the east branch of the tree
                -west: Represents the west branch of the tree
            Output:
                -selected : Either 1 or 0 based on which branch is better.
        """
        return west > east
