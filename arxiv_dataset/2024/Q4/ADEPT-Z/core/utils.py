"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 1969-12-31 18:00:00
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2025-01-21 18:37:35
"""

"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-12-26 19:57:31
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-01-15 15:49:13
"""

import torch.nn as nn


def hidden_register_hook(m, input, output):
    m._recorded_hidden = output


def register_hidden_hooks(model):
    for name, m in model.named_modules():
        if isinstance(m, (nn.ReLU, nn.GELU, nn.Hardswish, nn.ReLU6)):
            m.register_forward_hook(hidden_register_hook)


def combination_sum(candidates, target):
    """
    given candidate integers as a list and a target integer,\
    find all unique combinations that sum to the target using dynamic programming
    """
    # Sort the candidates to help avoid duplicates in the result
    candidates.sort()

    # Initialize a 2D DP table to store combinations for each target value
    dp = [[] for _ in range(target + 1)]

    # For a target of 0, there is one empty combination
    dp[0] = [[]]

    # Loop through the candidates and build up the DP table
    for candidate in candidates:
        for i in range(candidate, target + 1):
            for combo in dp[i - candidate]:
                dp[i].append(combo + [candidate])

    # Return the combinations for the target value
    return dp[target]
