#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    DM-HSGD:
    https://proceedings.neurips.cc/paper/2021/file/d994e3728ba5e28defb88a3289cd7ee8-Paper.pdf
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
class DMHSGD(BaseDL):
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
                 "dmhsgd",
                 f"minibatch{params['mini_batch']}_megabatch{params['mega_batch']}_betaX{params['betaX']}_betaY{params['betaY']}")

        # Extract parameters
        if 'betaX' in params:
            self.betaX = params['betaX']
        else:
            self.betaX = 1e-2
        if 'betaY' in params:
            self.betaY = params['betaY']
        else:
            self.betaY = 1e-2

        # Initialize the extra variables
        self.prevX = [self.X[ind].detach().clone() for ind in range(len(self.X))]
        self.prevY = [self.Y[ind].detach().clone() for ind in range(len(self.Y))]

        # Check problem instance
        self.Dx, self.Dy = self.get_mega_grad(self.X, self.Y)

        # Save gradients
        self.Vx = [self.Dx[ind].detach().clone() for ind in range(len(self.Dx))]
        self.Vy = [self.Dy[ind].detach().clone() for ind in range(len(self.Dy))]
        self.prevVx = [self.Dx[ind].detach().clone() for ind in range(len(self.Dx))]
        self.prevVy = [self.Dy[ind].detach().clone() for ind in range(len(self.Dy))]

        # Update epoch counter
        self.epochs += (self.size * self.mega_batch) / self.num_train

    def one_step(self, iteration: int) -> tuple:

        # TIME THIS EPOCH
        time_i = time.time()

        if iteration == 1:
            pass
        else:

            x_grad_diff, y_grad_diff = self.get_stoch_grad_difference(self.X, self.prevX, self.Y, self.prevY, scaleX=(1 - self.betaX), scaleY=(1 - self.betaY))
            self.Vx = [x_grad_diff[j] + (1 - self.betaX) * self.prevVx[j] for j in range(len(x_grad_diff))]
            self.Vy = [y_grad_diff[j] + (1 - self.betaY) * self.prevVy[j] for j in range(len(y_grad_diff))]

        # Update gradient tracking terms
        self.Dx = [self.Dx[j] + self.Vx[j] - self.prevVx[j] for j in range(len(self.Dx))]
        self.Dy = [self.Dy[j] + self.Vy[j] - self.prevVy[j] for j in range(len(self.Dy))]
        self.prevVx = [self.Vx[k].detach().clone() for k in range(len(self.Vx))]
        self.prevVy = [self.Vy[k].detach().clone() for k in range(len(self.Vy))]
        self.prevX = [self.X[k].detach().clone() for k in range(len(self.X))]
        self.prevY = [self.Y[k].detach().clone() for k in range(len(self.Y))]

        # END TIME
        time_i1 = time.time()

        # ----- PERFORM COMMUNICATION OF GRADIENT TRACKING TERMS ----- #
        self.comm.Barrier()
        self.Dx, comm_time2x = self.communicate_with_neighbors(self.Dx)
        self.Dy, comm_time2y = self.communicate_with_neighbors(self.Dy)
        self.comm.Barrier()
        # ---------------------------------- #

        # Start time
        time_i2 = time.time()

        # UPDATE WEIGHTS
        self.X = self.y_obj.projection_x([self.X[k] - self.lrX * self.Dx[k] for k in range(len(self.X))], self.l1 * self.lrX)
        self.Y = self.y_obj.projection_y([self.Y[k] + self.lrY * self.Dy[k] for k in range(len(self.Y))])

        # End time
        time_i_end = time.time()

        # ----- PERFORM COMMUNICATION ----- #
        self.comm.Barrier()
        self.X, comm_time1x = self.communicate_with_neighbors(self.X)
        self.Y, comm_time1y = self.communicate_with_neighbors(self.Y)
        self.comm.Barrier()
        # ---------------------------------- #

        # SAVE TIMES
        comp_time = round(time_i_end - time_i2 + time_i1 - time_i, 4)
        comm_time = comm_time1x + comm_time1y + comm_time2x + comm_time2y

        # Update epoch counter
        self.epochs += (self.size * self.mini_batch) / self.num_train

        return comp_time, comm_time