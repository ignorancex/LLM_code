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
from mpi4py import MPI

# Custom classes
from models.allcnn import AllCNN
from models.fixup_resnet import fixup_resnet20
from models.fully_connected import FC
from models.adversary import Adversary
from models.replace_weights import Opt
from models.fair_classification import FairTerm


# Declare main class
class BaseDL:
    '''
        Custom base class for the deep learning problems
    '''

    def __init__(self,
                 params: typing.Dict,
                 mixing_matrix: numpy.array,
                 training_data: torch.utils.data.DataLoader,
                 mega_batch_train_loader: torch.utils.data.DataLoader,
                 full_batch_train_loader: torch.utils.data.DataLoader,
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
            self.eta = 0.1
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
            self.problem = "robust_nn"

        # GET THE CUDA DEVICE
        self.device = torch.device(f'cuda:{self.rank % 8}')

        # SAVE THE DATA
        self.data_loader = training_data
        self.mega_loader = mega_batch_train_loader
        self.full_data_loader = full_batch_train_loader

        # Initialize the models
        # Save channels:
        self.num_classes = 10
        self.channels = 3 if self.dataset_name == 'cifar' else 1
        self.num_train = 50000 if self.dataset_name == 'cifar' else 60000

        # Set the model
        if self.model_name == 'allcnn':
            self.model = AllCNN(self.num_classes).to(self.device)
        elif self.model_name == 'resnet':
            self.model = fixup_resnet20().to(self.device)
        else:
            self.model = FC(self.num_classes, self.channels, 784).to(self.device)

        # Set the y objective based on which problem setting we are considering
        if self.problem == 'fair_class':
            self.y_obj = FairTerm(self.num_classes, self.eta).to(self.device)
            self.loss_function = torch.nn.NLLLoss(reduction='none')  # Returns a vector
        else:
            self.y_obj = Adversary(self.channels, 32 if self.dataset_name == 'cifar' else 28).to(self.device) # Assume we perform a projection to 32 x 32
            self.loss_function = torch.nn.NLLLoss(reduction='mean')  # Returns a vector

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
        self.compute_time = []
        self.communication_time = []
        self.total_time = []
        self.epochs = 0

        # Print the initial information
        if self.rank == 0:
            print("{:<10} | {:<7} | {:<15} | {:<15} | {:<15} | {:<15} | {:<6}".format("Iteration", "Epoch",
                                                                                           "Consensus",
                                                                                           "Gradient",
                                                                                           "Train Loss",
                                                                                           "Test Acc",
                                                                                           "Time"))

    def compute_errors(self, testing_data: torch.utils.data.DataLoader, iteration: int, epoch: float, initial_time: float) -> None:
        """Save the relevant errors to the above-mentioned quantities"""

        # Impose a barrier
        self.comm.Barrier()

        # Save first errors
        # GET AVERAGE POINT
        avgX = self.get_average_param(self.X)
        avgY = self.get_average_param(self.Y)

        # Compute violations
        consX, consY, grad_norm = self.compute_optimality_criteria_fair_class(avgX, self.X, avgY, self.Y) if self.problem == 'fair_class' else self.compute_optimality_criteria_robust(avgX, self.X, avgY, self.Y)
        self.ConsensusViolationX.append(consX)
        self.ConsensusViolationY.append(consY)
        self.GradientViolation.append(grad_norm)

        # Compute accuracies
        train_loss, train_acc = self.test_fair_class(avgX, self.data_loader) if self.problem == 'fair_class' else self.test_robust(avgX, avgY, self.data_loader)
        self.TrainLoss.append(train_loss)
        self.TrainAcc.append(train_acc)

        # Testing errors
        test_loss, test_acc = self.test_fair_class(avgX, testing_data) if self.problem == 'fair_class' else self.test_robust(avgX, avgY, testing_data)
        self.TestLoss.append(test_loss)
        self.TestAcc.append(test_acc)

        # Print information
        if self.rank == 0:
            print("{:<10} | {:<7} | {:<15} | {:<15} | {:<15} | {:<15} | {:<6}".format(iteration,
                                    round(epoch, 2),
                                    round(consX + consY, 4),
                                    round(grad_norm, 4),
                                    round(train_loss, 4),
                                    round(test_acc, 4),
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

    def compute_optimality_criteria_robust(self,
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

        # Choose one random sample
        for batch_idx, (data, target) in enumerate(self.full_data_loader):

            # Zero out gradients
            self.replace_weights_model.zero_grad()
            self.replace_weights_y_obj.zero_grad()

            # Convert data to CUDA if possible
            data, target = data.to(self.device), target.to(self.device)

            # Perform a forward pass of the model
            out = self.y_obj(data, target, self.model, self.loss_function)
            out.backward()  # compute gradients with proper scaling

        # Save the gradient
        opt_grad_x = torch.cat([p.grad.data.cpu().detach().clone().flatten() for p in self.model.parameters()]).flatten()
        opt_grad_y = torch.cat([p.grad.data.cpu().detach().clone().flatten() for p in self.y_obj.parameters()]).flatten()
        opt_grad = [torch.cat((opt_grad_x, opt_grad_y))]

        # Get the average gradient
        self.comm.Barrier()
        avg_grad = self.get_average_param(opt_grad)
        grad_norm = torch.norm(avg_grad[0], p="fro") ** 2
        self.comm.Barrier()

        # Perform all-reduce to have sum of local violations, i.e. Frobenius norm of consensus
        array_to_send = numpy.array([local_violationX, local_violationY])
        recv_array = numpy.empty(shape=array_to_send.shape)
        self.comm.Barrier()
        self.comm.Allreduce(array_to_send, recv_array)  # Operation here is summation
        self.comm.Barrier()

        # return consensus at X, consensus at Y, average loss
        return recv_array[0] / self.size, recv_array[1] / self.size, grad_norm.item()

    def compute_optimality_criteria_fair_class(self,
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

        # Choose one random sample
        for batch_idx, (data, target) in enumerate(self.full_data_loader):

            # Zero out gradients
            self.replace_weights_model.zero_grad()
            self.replace_weights_y_obj.zero_grad()

            # Convert data to CUDA if possible
            data, target = data.to(self.device), target.to(self.device)

            # Perform a forward pass of the model
            outX = self.model(data)
            outX = self.loss_function(outX, target)  # [batchself.size x 1]

            # Now we group each loss value by the target
            outs = []  # concat to this
            for i in range(self.num_classes):

                # Get the subset of data points which have loss i
                subset_i = outX[target == i]  # These are the data points which have label i

                # Append this loss
                outs.append(subset_i.mean())

            # Compute the optimal Y term - this uses the Sherman-Moorison formula
            outs = torch.stack(outs).flatten()
            optY = (1 / self.eta) * outs
            optY = self.y_obj.projection([optY])[0]

            # Use this to compute the final gradient
            final_loss = (outs * optY).sum()
            final_loss.backward()

        # Save the gradient
        opt_grad = [torch.cat([p.grad.data.cpu().detach().clone().flatten() for p in self.model.parameters()]).flatten()]

        # Get the average gradient
        self.comm.Barrier()
        avg_grad = self.get_average_param(opt_grad)
        grad_norm = torch.norm(avg_grad[0], p="fro") ** 2
        self.comm.Barrier()

        # Perform all-reduce to have sum of local violations, i.e. Frobenius norm of consensus
        array_to_send = numpy.array([local_violationX, local_violationY])
        recv_array = numpy.empty(shape=array_to_send.shape)
        self.comm.Barrier()
        self.comm.Allreduce(array_to_send, recv_array)  # Operation here is summation
        self.comm.Barrier()

        # return consensus at X, consensus at Y, average loss
        return recv_array[0] /self.size, recv_array[1] /self.size, grad_norm.item()

    def get_stoch_grads_fair_class(self,
                        current_x: typing.List[torch.Tensor],
                        current_y: typing.List[torch.Tensor]) -> tuple:
        """Get the deterministic gradients at this local point"""

        # Replace the current parameters
        self.replace_weights_model.step(current_x, self.device)
        self.replace_weights_y_obj.step(current_y, self.device)

        # Set to train
        self.model.train()
        self.y_obj.train()

        # Choose one random sample
        for batch_idx, (data, target) in enumerate(self.data_loader):

            # Zero out gradients
            self.replace_weights_model.zero_grad()
            self.replace_weights_y_obj.zero_grad()

            # Convert data to CUDA if possible
            data, target = data.to(self.device), target.to(self.device)

            # Perform a forward pass of the model
            outX = self.model(data)
            outX = self.loss_function(outX, target)  # [batchself.size x 1]

            # Now we group each loss value by the target
            outs = []  # concat to this
            for i in range(self.num_classes):

                # Get the subset of data points which have loss i
                subset_i = outX[target == i]  # These are all of the data points which have label i

                # Generate the scaling factor
                scale = 1 / (len(subset_i) *self.size) if len(subset_i) > 0 else 0.0

                # Append this loss
                outs.append(scale * subset_i.sum())

            # Perform concatenation
            outs = torch.stack(outs).flatten()
            outY = self.y_obj(outs)

            # Backward pass - this also populates the gradients with respect to the X term
            outY.backward()

            # Save gradients
            gradX = [p.grad.data.cpu().detach().clone() for ind, p in enumerate(self.model.parameters())]
            gradY = [p.grad.data.cpu().detach().clone() for ind, p in enumerate(self.y_obj.parameters())]

            # Return once the passes have been complete
            return gradX, gradY

    def get_mega_grads_fair_class(self,
                                   current_x: typing.List[torch.Tensor],
                                   current_y: typing.List[torch.Tensor]) -> tuple:
        """Get the deterministic gradients at this local point"""

        # Replace the current parameters
        self.replace_weights_model.step(current_x, self.device)
        self.replace_weights_y_obj.step(current_y, self.device)

        # Set to train
        self.model.train()
        self.y_obj.train()
        num_visited = 0

        # Zero out gradients
        self.replace_weights_model.zero_grad()
        self.replace_weights_y_obj.zero_grad()

        # Save outs
        outs = [0.0] * self.num_classes  # concat to this
        scales = torch.tensor([0.0] * self.num_classes)

        # Choose one random sample
        for batch_idx, (data, target) in enumerate(self.data_loader):

            # Check the slice size
            slice_to_use = torch.arange(len(target) if num_visited + len(target) <= self.mega_batch else abs(self.mega_batch - num_visited))

            # Convert data to CUDA if possible
            data, target = data[slice_to_use].to(self.device), target[slice_to_use].to(self.device)

            # Perform a forward pass of the model
            outX = self.model(data)
            outX = self.loss_function(outX, target)  # [batchself.size x 1]

            # Now we group each loss value by the target
            for i in range(self.num_classes):

                # Get the subset of data points which have loss i
                subset_i = outX[target == i]  # These are all of the data points which have label i

                # Generate the scaling factor
                scales[i] += len(subset_i)

                # Append this loss
                outs[i] += subset_i.sum()

            # Perform concatenation
            num_visited += len(target)

            # Break criteria
            if num_visited >= self.mega_batch:
                break

        # Backward pass - this also populates the gradients with respect to the X term
        # Component-wise scaling of the outputs
        scale = [1 / (scales[i] * self.size) if scales[i] > 0.0 else 0.0 for i in range(len(scales))]
        outs = torch.stack(outs).flatten() * torch.stack(scale).flatten().to(self.device)
        outY = self.y_obj(outs)
        outY.backward()

        # Save gradients
        gradX = [p.grad.data.cpu().detach().clone() for ind, p in enumerate(self.model.parameters())]
        gradY = [p.grad.data.cpu().detach().clone() for ind, p in enumerate(self.y_obj.parameters())]

        # Return once the passes have been complete
        return gradX, gradY

    def get_stoch_grad_difference_fair_class(self,
                       current_x: typing.List[torch.Tensor],
                       prev_x: typing.List[torch.Tensor],
                       current_y: typing.List[torch.Tensor],
                       prev_y: typing.List[torch.Tensor],
                       scaleX: float = 1.0,
                       scaleY: float = 1.0) -> tuple:
        """Get the deterministic gradients at this local point"""

        # Replace the current parameters
        self.replace_weights_model.step(current_x, self.device)
        self.replace_weights_y_obj.step(current_y, self.device)

        # Set to train
        self.model.train()
        self.y_obj.train()

        # Choose one random sample
        for batch_idx, (data, target) in enumerate(self.data_loader):

            # Zero out gradients
            self.replace_weights_model.zero_grad()
            self.replace_weights_y_obj.zero_grad()

            # Convert data to CUDA if possible
            data, target = data.to(self.device), target.to(self.device)

            # Perform a forward pass of the model
            outX = self.model(data)
            outX = self.loss_function(outX, target)  # [batchself.size x 1]

            # Now we group each loss value by the target
            outs = []  # concat to this
            for i in range(self.num_classes):

                # Get the subset of data points which have loss i
                subset_i = outX[target == i]  # These are all of the data points which have label i

                # Generate the scaling factor
                scale = 1 / (len(subset_i) * self.size) if len(subset_i) > 0 else 0.0

                # Append this loss
                outs.append(scale * subset_i.sum())

            # Perform concatenation
            outs = torch.stack(outs).flatten()
            outY = self.y_obj(outs)

            # Backward pass
            outY.backward()

            # Save gradients
            gradX = [p.grad.data.cpu().detach().clone() for ind, p in
                     enumerate(self.model.parameters())]
            gradY = [p.grad.data.cpu().detach().clone() for ind, p in
                     enumerate(self.y_obj.parameters())]

            # Use the previous weights
            self.replace_weights_model.step(prev_x, self.device)
            self.replace_weights_y_obj.step(prev_y, self.device)

            # Zero out gradients
            self.replace_weights_model.zero_grad()
            self.replace_weights_y_obj.zero_grad()

            # Perform a forward pass of the model
            outX = self.model(data)
            outX = self.loss_function(outX, target)  # [batchself.size x 1]

            # Now we group each loss value by the target
            outs = []  # concat to this
            for i in range(self.num_classes):
                # Get the subset of data points which have loss i
                subset_i = outX[target == i]  # These are all of the data points which have label i

                # Generate the scaling factor
                scale = 1 / (len(subset_i) * self.size) if len(subset_i) > 0 else 0.0

                # Append this loss
                outs.append(scale * subset_i.sum())

            # Perform concatenation
            outs = torch.stack(outs).flatten()
            outY = self.y_obj(outs)

            # Backward pass
            outY.backward()

            # Save gradients
            gradX = [gradX[ind] - scaleX * p.grad.data.cpu().detach().clone() for ind, p in
                     enumerate(self.model.parameters())]
            gradY = [gradY[ind] - scaleY * p.grad.data.cpu().detach().clone() for ind, p in
                     enumerate(self.y_obj.parameters())]

            # Return the sample difference
            return gradX, gradY

    def get_stoch_grad_robust(self,
                                current_x: typing.List[torch.Tensor],
                                current_y: typing.List[torch.Tensor]) -> tuple:
        """Compute a batched gradient with respect to the current iterate"""

        # Put the current weights on the devices and in the models
        self.replace_weights_model.step(current_x, self.device)
        self.replace_weights_y_obj.step(current_y, self.device)

        # Set to training
        self.model.train()
        self.y_obj.train()

        # Loop over the data
        for batch_idx, (data, target) in enumerate(self.data_loader):
            # Put the data on the device
            data, target = data.to(self.device), target.to(self.device)

            # Reset the gradients
            self.replace_weights_model.zero_grad()
            self.replace_weights_y_obj.zero_grad()

            # Forward pass using just the adversary since this will compute both gradients
            out = (1 / self.size) * self.y_obj(data, target, self.model, self.loss_function)
            out.backward()  # compute gradients with proper scaling

            # Save the gradients and return to CPU
            gradsX = [p.grad.detach().cpu() for i, p in enumerate(self.model.parameters())]
            gradsY = [p.grad.detach().cpu() for i, p in enumerate(self.y_obj.parameters())]

            return gradsX, gradsY

    def get_mega_grad_robust(self,
                             current_x: typing.List[torch.Tensor],
                             current_y: typing.List[torch.Tensor]) -> tuple:
        """Compute a batched gradient with respect to the current iterate"""

        # Put the current weights on the devices and in the models
        self.replace_weights_model.step(current_x, self.device)
        self.replace_weights_y_obj.step(current_y, self.device)

        # Set to training
        self.model.train()
        self.y_obj.train()

        # Loop over the data
        for batch_idx, (data, target) in enumerate(self.mega_loader):
            # Put the data on the device
            data, target = data.to(self.device), target.to(self.device)

            # Reset the gradients
            self.replace_weights_model.zero_grad()
            self.replace_weights_y_obj.zero_grad()

            # Forward pass using just the adversary since this will compute both gradients
            out = (1 / self.size) * self.y_obj(data, target, self.model, self.loss_function)
            out.backward()  # compute gradients with proper scaling

            # Save the gradients and return to CPU
            gradsX = [p.grad.detach().cpu() for i, p in enumerate(self.model.parameters())]
            gradsY = [p.grad.detach().cpu() for i, p in enumerate(self.y_obj.parameters())]

            return gradsX, gradsY

    def get_stoch_grad_difference_robust(self,
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

        # Loop over the data
        for batch_idx, (data, target) in enumerate(self.data_loader):

            # Put the data on the device
            data, target = data.to(self.device), target.to(self.device)

            # Reset the gradients
            self.replace_weights_model.zero_grad()
            self.replace_weights_y_obj.zero_grad()

            # Forward pass using just the adversary since this will compute both gradients
            out = (1 / self.size) * self.y_obj(data, target, self.model, self.loss_function)
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
            out = (1 / self.size) * self.y_obj(data, target, self.model, self.loss_function)
            out.backward()  # compute gradients with proper scaling

            # Save the gradients and return to CPU
            gradsX = [gradsX[i] - scaleX * p.grad.detach().cpu() for i, p in enumerate(self.model.parameters())]
            gradsY = [gradsY[i] - scaleY * p.grad.detach().cpu() for i, p in enumerate(self.y_obj.parameters())]

            return gradsX, gradsY

    def test_fair_class(self, weights: typing.List[torch.Tensor],
                        testing_data: torch.utils.data.DataLoader) -> tuple:
        '''Test the data using the average weights'''

        self.replace_weights_model.zero_grad()
        self.replace_weights_model.step(weights, self.device)
        self.model.eval()

        # Create separate testing loss for testing data
        loss_function = torch.nn.NLLLoss(reduction='sum')

        # Allocate space for testing loss and accuracy
        test_loss = 0
        correct = 0

        # Do not compute gradient with respect to the testing data
        with torch.no_grad():

            # Loop over testing data
            for data, target in testing_data:

                # Use CUDA
                data, target = data.to(self.device).float(), target.to(self.device).long()

                # Evaluate the model on the testing data
                output = self.model(data)
                test_loss += loss_function(output, target).item()

                # Gather predictions on testing data
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        # Compute number of testing data points
        num_test_points = int(len(testing_data.dataset) / self.size)

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

    def test_robust(self,
             avgX: typing.List[torch.Tensor],
             avgY: typing.List[torch.Tensor],
             testing_data: torch.utils.data.DataLoader) -> tuple:
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

            # Loop over testing data
            for data, target in testing_data:
                # Use CUDA
                data, target = data.to(self.device), target.to(self.device)

                # Evaluate the model on the testing data - perturbed by the adversary
                output = self.model(data + avgY[0].to(self.device))
                test_loss += loss_function(output, target).item()

                # Gather predictions on testing data
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        # Compute number of testing data points
        num_test_points = int(len(testing_data.dataset) / self.size)

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

    def solve(self, outer_iterations: int, testing_data: torch.utils.data.DataLoader) -> float:
        '''Perform the algorithm updates for outer_iterations iterations'''

        # Call the report procedure
        self.compute_errors(testing_data, iteration=0, epoch=self.epochs, initial_time=time.time())

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
                self.compute_errors(testing_data, iteration=i, epoch=self.epochs, initial_time=t0)

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