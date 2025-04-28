#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Base Class for deep learning with MPI
"""

from __future__ import print_function
import argparse
import os
import time
import torch
import numpy
import typing
import numpy as np
from mpi4py import MPI

# Custom classes
from models.dro import DRO
from models.lenet import LENET
from models.linear import LINEAR
from models.fully_connected import FC
from models.replace_weights import Opt


# Declare main class
class BaseDL:
    '''
        Custom base class for the deep learning problems
    '''

    def __init__(self,
                 params: typing.Dict,
                 mixing_matrix: numpy.array,
                 training_data: torch.Tensor,
                 training_labels: torch.Tensor,
                 testing_data: torch.Tensor,
                 testing_labels: torch.Tensor,
                 comm_world,
                 comm_size,
                 current_rank,
                 method_name: str,
                 path_extension: str = ''):

        # SAVE MPI THINGS
        self.comm = comm_world
        self.size = comm_size
        self.rank = current_rank

        # GATHER COMMUNICATION INFORMATION FROM THE MIXING MATRIX
        self.mixing_matrix = mixing_matrix.float()

        # PARSE COMMUNICATION GRAPH TO GET PEERS AND CORRESPONDING WEIGHTS
        self.peers = torch.where(self.mixing_matrix[self.rank, :] != 0)[0].tolist()
        self.peers.remove(self.rank) # remove yourself from the list

        # Get the weights
        self.peer_weights = self.mixing_matrix[self.rank, self.peers].tolist()

        # Get weights
        self.my_weight = self.mixing_matrix[self.rank, self.rank].item()

        # PARSE INPUT/TRAINING PARAMETERS
        self.method_name = method_name
        self.path_extension = path_extension
        self.agent_indices = params['index']
        if 'l1' in params:
            self.l1 = params['l1']
        else:
            self.l1 = 0.0
        if 'lrX' in params:
            self.lrX = params['lrX']
        else:
            self.lrX = 1e-2
        if 'lrY' in params:
            self.lrY = params['lrY']
        else:
            self.lrY = 1e-2
        if 'frequency' in params:
            self.frequency = params['frequency']
        else:
            self.frequency = 32
        if 'mini_batch' in params:
            self.mini_batch = params['mini_batch']
        else:
            self.mini_batch = 32
        if 'mega_batch' in params:
            self.mega_batch = params['mega_batch']
        else:
            self.mega_batch = 32
        if 'report' in params:
            self.report = params['report']
        else:
            self.report = 20
        if 'trial' in params:
            self.trial = params['trial']
        else:
            self.trial = 1
        if 'eta' in params:
            self.eta = params['eta']
        else:
            self.eta = 1.0
        if 'dataset_name' in params:
            self.dataset_name = params['dataset_name']
        else:
            self.dataset_name = "mnist"
        if 'model' in params:
            self.model_name = params['model']
        else:
            self.model_name = "fc"
        if 'problem' in params:
            self.problem = params['problem']
        else:
            self.problem = "dro"

        # GET THE CUDA DEVICE
        self.device = torch.device(f'cuda:{self.rank % 8}')

        # Initialize the models
        # Save channels:
        self.num_classes = 10
        self.channels = 1
        self.num_train = len(training_data[0]) if self.problem == 'plgame' else 60000

        # Set the model
        # SAVE THE DATA
        self.training_data = training_data
        self.training_labels = training_labels
        self.testing_data = testing_data
        self.testing_labels = testing_labels

        if self.model_name == 'fc':
            self.model = FC(self.num_classes, self.channels, 784).to(self.device)
        elif self.model_name == 'linear':
            self.model = LINEAR(self.num_classes, 784).to(self.device)
        else:
            self.model = LENET(self.num_classes).to(self.device)
        self.y_obj = DRO(self.training_data.shape[0], self.size, self.eta, self.l1).to(self.device)
        self.loss_function = torch.nn.NLLLoss(reduction='none')  # Returns a vector

        # Initialize the updating weights rule, training loss
        self.replace_weights_model = Opt(self.model.parameters(), lr=0.1)
        self.replace_weights_y_obj = Opt(self.y_obj.parameters(), lr=0.1)

        # LOAD VARIABLES ON SEPARATE GPUS
        # X = min variable
        # Y = max variable
        try:

            # Load from file
            self.X = [torch.tensor(numpy.load(os.path.join(os.getcwd(),
                                                           f'init_weights_X/{self.problem}/{self.dataset_name}/{self.model_name}/trial{self.trial}/rank{self.rank}/layer{ind}.dat'),
                                              allow_pickle=True)) for ind, p in
                      enumerate(self.model.parameters())]
            self.Y = [torch.tensor(numpy.load(os.path.join(os.getcwd(),
                                                           f'init_weights_Y/{self.problem}/{self.dataset_name}/{self.model_name}/trial{self.trial}/rank{self.rank}/layer{ind}.dat'),
                                              allow_pickle=True)) for ind, p in enumerate(self.y_obj.parameters())]

        except:

            # No weights
            if self.rank == 0:
                print(f"[INFO] Creating initial weights for trial {self.trial}...")

            # Create random weights
            self.X = [p.cpu().detach().clone() for p in self.model.parameters()]
            self.Y = [p.cpu().detach().clone() for p in self.y_obj.parameters()]

            # Save these
            for ind in range(len(self.X)):

                # Verify the path
                try:
                    os.mkdir(os.path.join(os.getcwd(), f'init_weights_X/'))
                except:
                    # Main storage already exists
                    pass
                try:
                    os.mkdir(os.path.join(os.getcwd(), f'init_weights_X/{self.problem}'))
                except:
                    # Main storage already exists
                    pass
                try:
                    os.mkdir(os.path.join(os.getcwd(), f'init_weights_X/{self.problem}/{self.dataset_name}'))
                except:
                    # Main storage already exists
                    pass
                try:
                    os.mkdir(os.path.join(os.getcwd(), f'init_weights_X/{self.problem}/{self.dataset_name}/{self.model_name}'))
                except:
                    # Main storage already exists
                    pass
                try:
                    os.mkdir(os.path.join(os.getcwd(), f'init_weights_X/{self.problem}/{self.dataset_name}/{self.model_name}/trial{self.trial}'))
                except:
                    # Main storage already exists
                    pass
                try:
                    os.mkdir(
                        os.path.join(os.getcwd(), f'init_weights_X/{self.problem}/{self.dataset_name}/{self.model_name}/trial{self.trial}/rank{self.rank}'))
                except:
                    # Main storage already exists
                    pass

                # Actually save the weights
                self.X[ind].numpy().dump(
                    f"init_weights_X/{self.problem}/{self.dataset_name}/{self.model_name}/trial{self.trial}/rank{self.rank}/layer{ind}.dat")

            for ind in range(len(self.Y)):

                # Verify the path
                try:
                    os.mkdir(os.path.join(os.getcwd(), f'init_weights_Y/'))
                except:
                    # Main storage already exists
                    pass
                try:
                    os.mkdir(os.path.join(os.getcwd(), f'init_weights_Y/{self.problem}'))
                except:
                    # Main storage already exists
                    pass
                try:
                    os.mkdir(os.path.join(os.getcwd(), f'init_weights_Y/{self.problem}/{self.dataset_name}'))
                except:
                    # Main storage already exists
                    pass
                try:
                    os.mkdir(os.path.join(os.getcwd(), f'init_weights_Y/{self.problem}/{self.dataset_name}/{self.model_name}'))
                except:
                    # Main storage already exists
                    pass
                try:
                    os.mkdir(os.path.join(os.getcwd(), f'init_weights_Y/{self.problem}/{self.dataset_name}/{self.model_name}/trial{self.trial}'))
                except:
                    # Main storage already exists
                    pass
                try:
                    os.mkdir(
                        os.path.join(os.getcwd(), f'init_weights_Y/{self.problem}/{self.dataset_name}/{self.model_name}/trial{self.trial}/rank{self.rank}'))
                except:
                    # Main storage already exists
                    pass

                # Save the weights
                self.Y[ind].numpy().dump(
                    f"init_weights_Y/{self.problem}/{self.dataset_name}/{self.model_name}/trial{self.trial}/rank{self.rank}/layer{ind}.dat")

        # Save norm histories and consensus histories
        self.ConsensusViolationX = []
        self.ConsensusViolationY = []
        self.GradientViolation = []
        self.TrainLoss = []
        self.TrainAcc = []
        self.TestLoss = []
        self.TestAcc = []
        self.AvgNNZ = []
        self.compute_time = []
        self.communication_time = []
        self.total_time = []
        self.epochs = 0

        # Print the initial information
        if self.rank == 0:
            print("{:<10} | {:<7} | {:<15} | {:<15} | {:<15} | {:<15} | {:<15} | {:<6}".format("Iteration", "Epoch",
                                                                                           "Consensus",
                                                                                           "Gradient",
                                                                                           "Train Loss",
                                                                                           "Test Acc",
                                                                                           "Density",
                                                                                           "Time"))

    def compute_errors(self, iteration: int, epoch: float, initial_time: float) -> None:
        """Save the relevant errors to the above-mentioned quantities"""

        # Impose a barrier
        self.comm.Barrier()

        # Save first errors
        # GET AVERAGE POINT
        avgX = self.get_average_param(self.X)
        avgY = self.get_average_param(self.Y)

        # Compute violations
        consX, consY, grad_norm, nnz = self.compute_optimality_criteria(avgX, self.X, avgY, self.Y)
        self.ConsensusViolationX.append(consX)
        self.ConsensusViolationY.append(consY)
        self.GradientViolation.append(grad_norm)
        self.AvgNNZ.append(nnz)

        # Compute accuracies
        train_loss, train_acc = self.test(avgX, avgY, self.training_data[self.agent_indices], self.training_labels[self.agent_indices])
        self.TrainLoss.append(train_loss)
        self.TrainAcc.append(train_acc)

        # Testing errors
        test_loss, test_acc = self.test(avgX, avgY, self.testing_data, self.testing_labels)
        self.TestLoss.append(test_loss)
        self.TestAcc.append(test_acc)

        # Print information
        if self.rank == 0:
            print("{:<10} | {:<7} | {:<15} | {:<15} | {:<15} | {:<15} | {:<15} | {:<6}".format(iteration,
                                    round(epoch, 2),
                                    round(consX + consY, 4),
                                    round(grad_norm, 4),
                                    round(train_loss, 4),
                                    round(test_acc, 4),
                                    round(nnz, 4),
                                    round(time.time() - initial_time, 1)))

        # End barrier
        self.comm.Barrier()

        return None

    def save_values(self):
        """Save relevant values as a numpy array"""

        self.comm.Barrier()

        # Rank 0's solutions are the ones to use
        if self.rank == 0:

            # First, make a home for the results
            # Make directory for both the dataset and the method and the model
            try:
                os.mkdir(os.path.join(os.getcwd(), f'results/'))
            except:
                # Main storage already exists
                pass
            try:
                os.mkdir(os.path.join(os.getcwd(), f'results/{self.problem}'))
            except:
                # Main storage already exists
                pass
            try:
                os.mkdir(os.path.join(os.getcwd(), f'results/{self.problem}/{self.dataset_name}'))
            except:
                # Main storage already exists
                pass
            try:
                os.mkdir(os.path.join(os.getcwd(), f'results/{self.problem}/{self.dataset_name}/{self.model_name}'))
            except:
                # Main storage already exists
                pass
            try:
                os.mkdir(os.path.join(os.getcwd(), f'results/{self.problem}/{self.dataset_name}/{self.model_name}/{self.method_name}'))
            except:
                # Main storage already exists
                pass
            try:
                os.mkdir(os.path.join(os.getcwd(), f'results/{self.problem}/{self.dataset_name}/{self.model_name}/{self.method_name}/{self.trial}'))
            except:
                # Main storage already exists
                pass

            # Save the main path
            path =os.path.join(os.getcwd(), f'results/{self.problem}/{self.dataset_name}/{self.model_name}/{self.method_name}/{self.trial}')

            # Save results
            # Consensus violation
            numpy.savetxt(
                f'{path}/consensusX_lrX{self.lrX}_lrY{self.lrY}_{self.path_extension}.txt',
                self.ConsensusViolationX, fmt='%.7f')
            numpy.savetxt(
                f'{path}/consensusY_lrX{self.lrX}_lrY{self.lrY}_{self.path_extension}.txt',
                self.ConsensusViolationY, fmt='%.7f')
            numpy.savetxt(
                f'{path}/gradient_lrX{self.lrX}_lrY{self.lrY}_{self.path_extension}.txt',
                self.GradientViolation, fmt='%.7f')
            numpy.savetxt(
                f'{path}/total_lrX{self.lrX}_lrY{self.lrY}_{self.path_extension}.txt',
                numpy.array(self.GradientViolation) + numpy.array(self.ConsensusViolationX)
                + numpy.array(self.ConsensusViolationY), fmt='%.7f')

            # Computing times
            numpy.savetxt(
                f'{path}/total_time_lrX{self.lrX}_lrY{self.lrY}_{self.path_extension}.txt',
                self.total_time, fmt='%.7f')
            numpy.savetxt(
                f'{path}/comm_time_lrX{self.lrX}_lrY{self.lrY}_{self.path_extension}.txt',
                self.communication_time, fmt='%.7f')
            numpy.savetxt(
                f'{path}/comp_time_lrX{self.lrX}_lrY{self.lrY}_{self.path_extension}.txt',
                self.compute_time, fmt='%.7f')

            # Loss
            numpy.savetxt(
                f'{path}/train_loss_lrX{self.lrX}_lrY{self.lrY}_{self.path_extension}.txt',
                self.TrainLoss, fmt='%.7f')
            numpy.savetxt(
                f'{path}/test_loss_lrX{self.lrX}_lrY{self.lrY}_{self.path_extension}.txt',
                self.TestLoss, fmt='%.7f')
            numpy.savetxt(
                f'{path}/avgnnz_lrX{self.lrX}_lrY{self.lrY}_{self.path_extension}.txt',
                self.AvgNNZ, fmt='%.7f')

            # Loss
            numpy.savetxt(
                f'{path}/train_acc_lrX{self.lrX}_lrY{self.lrY}_{self.path_extension}.txt',
                self.TrainAcc, fmt='%.7f')
            numpy.savetxt(
                f'{path}/test_acc_lrX{self.lrX}_lrY{self.lrY}_{self.path_extension}.txt',
                self.TestAcc, fmt='%.7f')

        self.comm.Barrier()

    def communicate_with_neighbors(self, params_to_communicate: typing.List[torch.Tensor]) -> tuple:
        """Communicate parameters with neighbors"""

        # TIME IT
        self.comm.Barrier()
        time0 = MPI.Wtime()

        # ----- LOOP OVER PARAMETERS ----- #
        for pa in range(len(params_to_communicate)):

            # DEFINE VARIABLE TO SEND
            send_data = params_to_communicate[pa].cpu().detach().numpy()
            recv_data = numpy.empty(shape=((len(self.peers),) + params_to_communicate[pa].shape), dtype=numpy.float32)

            # SET UP REQUESTS TO INSURE CORRECT SENDS/RECVS
            recv_request = [MPI.REQUEST_NULL for _ in range(int(2 * len(self.peers)))]

            # SEND THE DATA
            for ind, peer_id in enumerate(self.peers):
                # Send the data
                recv_request[ind + len(self.peers)] = self.comm.Isend(send_data, dest=peer_id)

            # RECEIVE THE DATA
            for ind, peer_id in enumerate(self.peers):
                # Receive the data
                recv_request[ind] = self.comm.Irecv(recv_data[ind, :], source=peer_id)

            # HOLD UNTIL ALL COMMUNICATIONS COMPLETE
            MPI.Request.waitall(recv_request)

            # SCALE CURRENT WEIGHTS
            params_to_communicate[pa] = self.my_weight * params_to_communicate[pa]

            # Update global variables
            for ind in range(len(self.peers)):
                params_to_communicate[pa] += (self.peer_weights[ind] * torch.tensor(recv_data[ind, :]))

        self.comm.Barrier()

        return params_to_communicate, round(MPI.Wtime() - time0, 4)

    def get_average_param(self, list_of_params: typing.List[torch.Tensor]) -> typing.List[torch.Tensor]:
        '''Perform ALLREDUCE of neighbor parameters'''

        # Save information to blank list
        output_list_of_parameters = [None] * len(list_of_params)

        # ----- LOOP OVER PARAMETERS ----- #
        for pa in range(len(list_of_params)):
            # Prep data to be sent
            send_data = list_of_params[pa].cpu().detach().numpy()

            # Prep reception location
            recv_data = numpy.empty(shape=(list_of_params[pa].shape), dtype=numpy.float32)

            # Barrier
            self.comm.Barrier()

            # Perform ALLREDUCE
            self.comm.Allreduce(send_data, recv_data)

            # Barrier
            self.comm.Barrier()

            # Save information
            output_list_of_parameters[pa] = (1 / self.size) * torch.tensor(recv_data)

        return output_list_of_parameters

    def compute_optimality_criteria(self,
                                    avgX: typing.List[torch.Tensor],
                                    localX: typing.List[torch.Tensor],
                                    avgY: typing.List[torch.Tensor],
                                    localY: typing.List[torch.Tensor]) -> tuple:
        '''
            Compute the relevant metrics for this problem
        '''

        # Compute consensus for this agent - both X and Y
        local_violationX = sum([numpy.linalg.norm(
            localX[i].cpu().numpy().flatten() - avgX[i].cpu().numpy().flatten(), ord=2) ** 2
                                for i in range(len(localX))])

        local_violationY = sum([numpy.linalg.norm(
            localY[i].cpu().numpy().flatten() - avgY[i].cpu().numpy().flatten(), ord=2) ** 2
                                for i in range(len(localY))])

        # Compute the losses
        self.replace_weights_model.step(avgX, self.device)
        self.replace_weights_y_obj.step(avgY, self.device)
        self.model.eval()
        self.y_obj.eval()

        # Zero out gradients
        self.replace_weights_model.zero_grad()
        self.replace_weights_y_obj.zero_grad()

        # Convert data to CUDA if possible
        data, target = self.training_data.to(self.device), self.training_labels.to(self.device)

        # Get the optimal gradient
        gradX = self.y_obj.get_optimal_y(data, target, self.model, self.loss_function)

        # Get the norm
        grad_norm = torch.norm(gradX, p=2) ** 2

        # Compute the avg nnz
        _, avg_nnz = self.y_obj.number_non_zeros(localX)

        # Perform all-reduce to have sum of local violations, i.e. Frobenius norm of consensus
        array_to_send = numpy.array([local_violationX, local_violationY, avg_nnz])
        recv_array = numpy.empty(shape=array_to_send.shape)
        self.comm.Barrier()
        self.comm.Allreduce(array_to_send, recv_array)  # Operation here is summation
        self.comm.Barrier()

        # return consensus at X, consensus at Y, average loss
        return recv_array[0] / self.size, recv_array[1] / self.size, grad_norm.item(), 100 * recv_array[2] / self.size

    def get_stoch_grad(self,
                                current_x: typing.List[torch.Tensor],
                                current_y: typing.List[torch.Tensor]) -> tuple:
        """Compute a batched gradient with respect to the current iterate"""

        # Put the current weights on the devices and in the models
        self.replace_weights_model.step(current_x, self.device)
        self.replace_weights_y_obj.step(current_y, self.device)

        # Set to training
        self.model.train()
        self.y_obj.train()

        # Get a minibatch
        batch = np.random.choice(self.agent_indices, self.mini_batch)

        # Put the data on the device
        data, target = self.training_data[batch].to(self.device), self.training_labels[batch].to(self.device)

        # Reset the gradients
        self.replace_weights_model.zero_grad()
        self.replace_weights_y_obj.zero_grad()

        # Forward pass using just the adversary since this will compute both gradients
        out = self.y_obj(data, target, self.model, self.loss_function, batch)
        out.backward()  # compute gradients with proper scaling

        # Save the gradients and return to CPU
        gradsX = [p.grad.detach().cpu() for i, p in enumerate(self.model.parameters())]
        gradsY = [p.grad.detach().cpu() for i, p in enumerate(self.y_obj.parameters())]

        return gradsX, gradsY

    def get_mega_grad(self,
                             current_x: typing.List[torch.Tensor],
                             current_y: typing.List[torch.Tensor]) -> tuple:
        """Compute a batched gradient with respect to the current iterate"""

        # Put the current weights on the devices and in the models
        self.replace_weights_model.step(current_x, self.device)
        self.replace_weights_y_obj.step(current_y, self.device)

        # Set to training
        self.model.train()
        self.y_obj.train()

        # Get a minibatch
        batch = np.random.choice(self.agent_indices, self.mega_batch)

        # Put the data on the device
        data, target = self.training_data[batch].to(self.device), self.training_labels[batch].to(self.device)

        # Reset the gradients
        self.replace_weights_model.zero_grad()
        self.replace_weights_y_obj.zero_grad()

        # Forward pass using just the adversary since this will compute both gradients
        out = self.y_obj(data, target, self.model, self.loss_function, batch)
        out.backward()  # compute gradients with proper scaling

        # Save the gradients and return to CPU
        gradsX = [p.grad.detach().cpu() for i, p in enumerate(self.model.parameters())]
        gradsY = [p.grad.detach().cpu() for i, p in enumerate(self.y_obj.parameters())]

        return gradsX, gradsY

    def get_stoch_grad_difference(self,
                                current_x: typing.List[torch.Tensor],
                                prev_x: typing.List[torch.Tensor],
                                current_y: typing.List[torch.Tensor],
                                prev_y: typing.List[torch.Tensor],
                                scaleX: float = 1.0,
                                scaleY: float = 1.0) -> tuple:
        """Compute a batched gradient with respect to the current gradient
        and the previous gradient to get the gradient difference"""

        # Put the current weights on the devices and in the models
        self.replace_weights_model.step(current_x, self.device)
        self.replace_weights_y_obj.step(current_y, self.device)

        # Set to training
        self.model.train()
        self.y_obj.train()

        # Get a minibatch
        batch = np.random.choice(self.agent_indices, self.mini_batch)

        # Put the data on the device
        data, target = self.training_data[batch].to(self.device), self.training_labels[batch].to(self.device)

        # Reset the gradients
        self.replace_weights_model.zero_grad()
        self.replace_weights_y_obj.zero_grad()

        # Forward pass using just the adversary since this will compute both gradients
        out = self.y_obj(data, target, self.model, self.loss_function, batch)
        out.backward()  # compute gradients with proper scaling

        # Save the gradients and return to CPU
        gradsX = [p.grad.detach().cpu() for i, p in enumerate(self.model.parameters())]
        gradsY = [p.grad.detach().cpu() for i, p in enumerate(self.y_obj.parameters())]

        # Update parameters with previous values and compute the difference
        self.replace_weights_model.step(prev_x, self.device)
        self.replace_weights_y_obj.step(prev_y, self.device)

        # Reset the gradients
        self.replace_weights_model.zero_grad()
        self.replace_weights_y_obj.zero_grad()

        # Forward pass using just the adversary since this will compute both gradients
        out = self.y_obj(data, target, self.model, self.loss_function, batch)
        out.backward()  # compute gradients with proper scaling

        # Save the gradients and return to CPU
        gradsX = [gradsX[i] - scaleX * p.grad.detach().cpu() for i, p in enumerate(self.model.parameters())]
        gradsY = [gradsY[i] - scaleY * p.grad.detach().cpu() for i, p in enumerate(self.y_obj.parameters())]

        return gradsX, gradsY

    def test(self,
             avgX: typing.List[torch.Tensor],
             avgY: typing.List[torch.Tensor],
             testing_data: torch.Tensor,
             testing_labels: torch.Tensor) -> tuple:
        '''Test the data using the average weights'''

        # Replace the weights
        self.replace_weights_model.step(avgX, self.device)
        self.replace_weights_y_obj.step(avgY, self.device)
        self.model.eval()
        self.y_obj.eval()

        # Zero out the gradients
        self.replace_weights_model.zero_grad()
        self.replace_weights_y_obj.zero_grad()

        # Create separate testing loss for testing data
        loss_function = torch.nn.NLLLoss(reduction='sum')

        # Allocate space for testing loss and accuracy
        test_loss = 0
        correct = 0

        # Do not compute gradient with respect to the testing data
        with torch.no_grad():

            data, target = testing_data.to(self.device), testing_labels.to(self.device)

            # Evaluate the model on the testing data - perturbed by the adversary
            output = self.model(data)
            test_loss += loss_function(output, target).item()

            # Gather predictions on testing data
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

        # Compute number of testing data points
        num_test_points = len(testing_data)

        # PERFORM ALL REDUCE TO HAVE AVERAGE
        array_to_send = numpy.array([correct, num_test_points, test_loss])
        recv_array = numpy.empty(shape=array_to_send.shape)

        # Barrier
        self.comm.Barrier()
        self.comm.Allreduce(array_to_send, recv_array)
        self.comm.Barrier()

        # Save loss and accuracy
        test_loss = recv_array[2] / recv_array[1]
        testing_accuracy = 100 * recv_array[0] / recv_array[1]

        return test_loss, testing_accuracy

    def solve(self, outer_iterations: int) -> float:
        '''Perform the algorithm updates for outer_iterations iterations'''

        # Call the report procedure
        self.compute_errors(iteration=0, epoch=self.epochs, initial_time=time.time())

        # TIME IT
        t0 = time.time()

        # Barrier communication at beginning of run
        self.comm.Barrier()

        # Loop over algorithm updates
        for i in range(1, outer_iterations + 1):

            # Perform one step
            comp_time, comm_time = self.one_step(i)

            # Barrier at end of iteration
            self.comm.Barrier()

            # Save information
            if i % self.report == 0 or i == 1:

                # Call the report procedure
                self.compute_errors(iteration=i, epoch=self.epochs, initial_time=t0)

            # APPEND TIMING INFORMATION
            self.compute_time.append(comp_time)
            self.communication_time.append(comm_time)
            self.total_time.append(comp_time + comm_time)

        # END TIME
        t1 = time.time() - t0

        # Save the results
        self.save_values()

        # Return the training time
        return t1

    def one_step(self, iteration: int) -> tuple:
        """Each method will update this equation"""

        raise NotImplementedError