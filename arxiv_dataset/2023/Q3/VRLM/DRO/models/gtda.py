#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    GT-DA: https://ieeexplore.ieee.org/document/9054056
"""

from __future__ import print_function
import argparse
import os
import time
import torch
import numpy
import typing
from mpi4py import MPI

# Custom classes
from models.base import BaseDL


# Declare main class
class GTDA(BaseDL):
    def __init__(self,
                 params: typing.Dict,
                 mixing_matrix: numpy.array,
                 training_data,
                 training_labels,
                 testing_data,
                 testing_labels,
                 comm_world,
                 comm_size,
                 current_rank
                 ):

        super().__init__(params,
                 mixing_matrix,
                 training_data,
                 training_labels,
                 testing_data,
                 testing_labels,
                 comm_world,
                 comm_size,
                 current_rank,
                 "gtda",
                 f"comm_rounds{params['comm_rounds']}")

        # Save comm_rounds
        self.comm_rounds = params['comm_rounds']

        # Check problem instance
        if self.problem == 'plgame':
            self.Dx, self.Dy = self.get_mega_grads_pl(self.X, self.Y)
        else:
            self.Dx, self.Dy = self.get_mega_grad_robust(self.X, self.Y)

        # Save gradient tracking terms
        self.Vx = [self.Dx[ind].detach().clone() for ind in range(len(self.Dx))]
        self.prevDx = [self.Dx[ind].detach().clone() for ind in range(len(self.Dx))]

    def one_step(self, iteration: int) -> tuple:

        # ----- PERFORM COMMUNICATION ----- #
        self.X, comm_time1x = self.communicate_with_neighbors(self.X)
        # ---------------------------------- #

        # TIME THIS EPOCH
        time_i = time.time()

        # UPDATE WEIGHTS
        self.X = [self.X[k] - self.lrX * self.Vx[k] for k in range(len(self.X))]

        # Perform the second update using the temporary terms
        for _ in range(self.comm_rounds):

            if self.problem == 'plgame':
                _, self.Dy = self.get_mega_grads_pl(self.X, self.Y)
            else:
                _, self.Dy = self.get_mega_grad_robust(self.X, self.Y)

            # Perform an update
            self.Y = [self.Y[k] + self.lrY * self.Dy[k] for k in range(len(self.Y))]

        # Compute the most recent gradient
        if self.problem == 'plgame':
            self.Dx, self.Dy = self.get_mega_grads_pl(self.X, self.Y)
        else:
            self.Dx, self.Dy = self.get_mega_grad_robust(self.X, self.Y)

        # ----- PERFORM COMMUNICATION ----- #
        self.Vx, comm_time2x = self.communicate_with_neighbors(self.Vx)
        # ---------------------------------- #

        # Update gradient tracking
        self.Vx = [self.Vx[j] + self.Dx[j] - self.prevDx[j] for j in range(len(self.Vx))]
        self.prevDx = [self.Dx[k].detach().clone() for k in range(len(self.Dx))]

        # Update epoch counter
        self.epochs += (self.size * self.mega_batch) / self.num_train

        # END time
        time_i_end = time.time()

        # SAVE TIMES
        comp_time = round(time_i_end - time_i, 4)
        comm_time = comm_time1x + comm_time2x

        return comp_time, comm_time