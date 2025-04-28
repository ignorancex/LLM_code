#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    New Method
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


# DECLARE THE SPIDER TYPE ESTIMATOR
class ProposedSPIDER(BaseDL):
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
                 "proposed_spider",
                 f"minibatch{params['mini_batch']}_megabatch{params['mega_batch']}_frequency{params['frequency']}_lrLam1{params['lrLam1']}_lrLam2{params['lrLam2']}")

        # Get Lambda learning rates
        if 'lrLam1' in params:
            self.lrLam1 = params['lrLam1']
        else:
            self.lrLam1 = 1e-2
        if 'lrLam2' in params:
            self.lrLam2 = params['lrLam2']
        else:
            self.lrLam2 = 1e-2

        # Initialize the extra variables
        self.prevX = [self.X[ind].detach().clone() for ind in range(len(self.X))]
        self.prevY = [self.Y[ind].detach().clone() for ind in range(len(self.Y))]

        # Check problem instance
        self.Vx, self.Vy = self.get_mega_grad(self.X, self.Y)

        # Save Lambda - Init at 0
        self.Lambda = [torch.zeros(self.Y[ind].shape) for ind in range(len(self.Y))]

        # Save gradients
        self.Dx = [self.Vx[ind].detach().clone() for ind in range(len(self.Vx))]
        self.Dy = [self.Vy[ind].detach().clone() for ind in range(len(self.Vy))]
        self.prevDx = [self.Vx[ind].detach().clone() for ind in range(len(self.Vx))]
        self.prevDy = [self.Vy[ind].detach().clone() for ind in range(len(self.Vy))]

    def one_step(self, iteration: int) -> tuple:

        # TIME THIS EPOCH
        time_i = time.time()

        # Mega batch
        if iteration % self.frequency == 0:

                self.Dx, self.Dy = self.get_mega_grad(self.X, self.Y)

        elif iteration == 1:
            pass

        # Mini-batch
        else:

            self.Dx, self.Dy = self.get_stoch_grad_difference(self.X, self.prevX, self.Y, self.prevY)

            # Add the previous term to the update - SPIDER-type update
            self.Dx = [self.Dx[k].detach().clone() + self.prevDx[k].detach().clone() for k in range(len(self.Dx))]
            self.Dy = [self.Dy[k].detach().clone() + self.prevDy[k].detach().clone() for k in range(len(self.Dy))]

        # Save the pre-communicated values for Y and Lambda
        pre_comm_y = [self.Y[k].detach().clone() for k in range(len(self.Y))]
        pre_comm_lam = [self.Lambda[k].detach().clone() for k in range(len(self.Lambda))]
        self.Vx = [self.Vx[k] + self.Dx[k] - self.prevDx[k] for k in range(len(self.Vx))]

        # Stop compute time
        int_time1 = time.time()

        # ----- PERFORM COMMUNICATION ----- #
        self.X, comm_time1x = self.communicate_with_neighbors(self.X)
        self.Y, comm_time1y = self.communicate_with_neighbors(self.Y)
        self.Vx, comm_time1vx = self.communicate_with_neighbors(self.Vx)
        self.Lambda, comm_time1lam = self.communicate_with_neighbors(self.Lambda)
        # ---------------------------------- #

        # Start time
        int_time2 = time.time()

        # Y update
        post_comm_y = [self.Y[k].detach().clone() for k in range(len(self.Y))]
        self.Vy = [self.Dy[k] - self.lrLam1 * (self.Lambda[k] - pre_comm_lam[k]) for k in range(len(self.Vy))]
        self.Y = self.y_obj.projection_y([self.Y[k] + self.lrY * self.Vy[k] for k in range(len(self.Y))])

        # Lambda update
        self.Lambda = [self.Lambda[k] + self.lrLam1 * self.lrLam2 * (post_comm_y[k] - pre_comm_y[k]) for k in range(len(self.Lambda))]

        # X update
        self.X = self.y_obj.projection_x([self.X[k] - self.lrX * self.Vx[k] for k in range(len(self.X))], self.l1 * self.lrX)

        # Save values
        self.prevDx = [self.Dx[k].detach().clone() for k in range(len(self.Dx))]
        self.prevDy = [self.Dy[k].detach().clone() for k in range(len(self.Dy))]
        self.prevX = [self.X[ind].detach().clone() for ind in range(len(self.X))]
        self.prevY = [self.Y[ind].detach().clone() for ind in range(len(self.Y))]

        # Update epoch counter
        self.epochs += (self.size * (self.mega_batch if iteration % self.frequency == 0 or iteration == 1 else self.mini_batch)) / self.num_train

        # END time
        time_i_end = time.time()

        # SAVE TIMES
        comp_time = round(time_i_end - int_time2 + int_time1 - time_i, 4)
        comm_time = comm_time1x + comm_time1y + comm_time1vx + comm_time1lam

        return comp_time, comm_time


# DECLARE STORM TYPE ESTIMATOR
class ProposedSTORM(BaseDL):
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
                 "proposed_storm",
                 f"minibatch{params['mini_batch']}_megabatch{params['mega_batch']}_lrLam1{params['lrLam1']}_lrLam2{params['lrLam2']}_betaX{params['betaX']}_betaY{params['betaY']}")

        # Get Lambda learning rates
        if 'lrLam1' in params:
            self.lrLam1 = params['lrLam1']
        else:
            self.lrLam1 = 1e-2
        if 'lrLam2' in params:
            self.lrLam2 = params['lrLam2']
        else:
            self.lrLam2 = 1e-2
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
        self.Vx, self.Vy = self.get_mega_grad(self.X, self.Y)

        # Save Lambda - Init at 0
        self.Lambda = [torch.zeros(self.Y[ind].shape) for ind in range(len(self.Y))]

        # Save gradients
        self.Dx = [self.Vx[ind].detach().clone() for ind in range(len(self.Vx))]
        self.Dy = [self.Vy[ind].detach().clone() for ind in range(len(self.Vy))]
        self.prevDx = [self.Vx[ind].detach().clone() for ind in range(len(self.Vx))]
        self.prevDy = [self.Vy[ind].detach().clone() for ind in range(len(self.Vy))]

    def one_step(self, iteration: int) -> tuple:

        # TIME THIS EPOCH
        time_i = time.time()

        if iteration == 1:
            pass

        else:

            x_grad_diff, y_grad_diff = self.get_stoch_grad_difference(self.X, self.prevX, self.Y,
                                                                                     self.prevY,
                                                                                     scaleX=(1 - self.betaX),
                                                                                     scaleY=(1 - self.betaY))
            self.Dx = [x_grad_diff[j] + (1 - self.betaX) * self.prevDx[j] for j in range(len(x_grad_diff))]
            self.Dy = [y_grad_diff[j] + (1 - self.betaY) * self.prevDy[j] for j in range(len(y_grad_diff))]

        # Save the pre-communicated values for Y and Lambda
        pre_comm_y = [self.Y[k].detach().clone() for k in range(len(self.Y))]
        pre_comm_lam = [self.Lambda[k].detach().clone() for k in range(len(self.Lambda))]
        self.Vx = [self.Vx[k] + self.Dx[k] - self.prevDx[k] for k in range(len(self.Vx))]

        # Stop compute time
        int_time1 = time.time()

        # ----- PERFORM COMMUNICATION ----- #
        self.X, comm_time1x = self.communicate_with_neighbors(self.X)
        self.Y, comm_time1y = self.communicate_with_neighbors(self.Y)
        self.Vx, comm_time1vx = self.communicate_with_neighbors(self.Vx)
        self.Lambda, comm_time1lam = self.communicate_with_neighbors(self.Lambda)
        # ---------------------------------- #

        # Start time
        int_time2 = time.time()

        # Y update
        post_comm_y = [self.Y[k].detach().clone() for k in range(len(self.Y))]
        self.Vy = [self.Dy[k] - self.lrLam1 * (self.Lambda[k] - pre_comm_lam[k]) for k in range(len(self.Vy))]
        self.Y = [self.Y[k] + self.lrY * self.Vy[k] for k in range(len(self.Y))]
        self.Y = self.y_obj.projection_y(self.Y)

        # Lambda update
        self.Lambda = [self.Lambda[k] + self.lrLam1 * self.lrLam2 * (post_comm_y[k] - pre_comm_y[k]) for k in range(len(self.Lambda))]

        # X update
        self.X = self.y_obj.projection_x([self.X[k] - self.lrX * self.Vx[k] for k in range(len(self.X))], self.l1 * self.lrX)

        # Save values
        self.prevDx = [self.Dx[k].detach().clone() for k in range(len(self.Dx))]
        self.prevDy = [self.Dy[k].detach().clone() for k in range(len(self.Dy))]
        self.prevX = [self.X[ind].detach().clone() for ind in range(len(self.X))]
        self.prevY = [self.Y[ind].detach().clone() for ind in range(len(self.Y))]

        # Update epoch counter
        self.epochs += (self.size * (self.mega_batch if iteration % self.frequency == 0 or iteration == 1 else self.mini_batch)) / self.num_train

        # END time
        time_i_end = time.time()

        # SAVE TIMES
        comp_time = round(time_i_end - int_time2 + int_time1 - time_i, 4)
        comm_time = comm_time1x + comm_time1y + comm_time1vx + comm_time1lam

        return comp_time, comm_time