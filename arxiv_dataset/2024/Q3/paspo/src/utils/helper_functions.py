"""Module providing the various helper functions."""

import logging
import logging.config
import os
from typing import List, Dict, Tuple
import numpy as np
import gym
import pandas as pd
from ray.rllib.utils.spaces.simplex import Simplex
import matplotlib.pyplot as plt
from utils.custom_keys import Postprocessing_Custom

logger = logging.getLogger(os.getenv("LOGGER_NAME"))



def calculate_action_space_dim(
    action_space, action_space_distribution: str = None
) -> int:
    """
    Calculate the size of the action space
    :param action_space:
    :return:
    """
    if isinstance(action_space, gym.spaces.Dict):
        space_output_dim_total = 0
        for space_name, space in action_space.spaces.items():
            space_output_dim = None
            if isinstance(space, gym.spaces.Discrete):
                space_output_dim = 1
            elif isinstance(space, gym.spaces.MultiDiscrete) and space is not None:
                space_output_dim = int(np.prod(space.shape))
            elif isinstance(space, gym.spaces.Box) and space is not None:
                space_output_dim = 2 * int(
                    np.sum(space.shape)
                )  # only valid for one dimensional .Box
            elif isinstance(space, Simplex) and space is not None:
                space_output_dim = int(np.sum(space.shape))
            else:
                error = ValueError(f"Unknown space type {space}")
                logger.error(error)
                raise error
            space_output_dim_total += space_output_dim
    else:
        space = action_space
        space_output_dim_total = 0
        if isinstance(space, gym.spaces.Discrete):
            space_output_dim = 1
        elif isinstance(space, gym.spaces.MultiDiscrete) and space is not None:
            space_output_dim = int(np.prod(space.shape))
        elif isinstance(space, gym.spaces.Box) and space is not None:
            if action_space_distribution == "TorchDirichletCustom" or action_space_distribution == "TorchDirichletCustomStable":
                space_output_dim = int(
                    np.sum(space.shape)
                )  # case that we use a dirichlet distribution which has one parameter per resource
            else:
                raise NotImplementedError(f"Please specify a action space distribution")
            # if action_space_distribution is None:
            #    space_output_dim = 2 * int(
            #        np.sum(space.shape)
            #    )  # only valid for one dimensional .Box #*2 because of two parameters, mean+var
            #    for normal distribution
        elif isinstance(space, Simplex) and space is not None:
            space_output_dim = int(np.sum(space.shape))
        else:
            error = ValueError(f"Unknown space type {space}")
            logger.error(error)
            raise error
        space_output_dim_total += space_output_dim

    return space_output_dim_total

