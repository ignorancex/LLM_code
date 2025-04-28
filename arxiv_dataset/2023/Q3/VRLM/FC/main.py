#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Fair classification main file
"""

from __future__ import print_function
import argparse
import os
import time
import torch
import numpy
import typing
from mpi4py import MPI
from torchvision import datasets, transforms

# Custom classes
from models.dposg import DPOSG
from models.gtsrvr import GTSRVR
from models.dmhsgd import DMHSGD
from models.proposed_method import *

# MPI set-up
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank() # get the current processor

# Main script to run
if __name__=='__main__':

    # ------------------------------------------------
    # Gather arguments
    parser = argparse.ArgumentParser(description='Testing on minimax fair classification problem.')

    parser.add_argument('--iterations', type=int, default=10000, help='Total number of communication rounds.')
    parser.add_argument('--comm_pattern', type=str, default='ring', choices=['ring', 'random', 'complete'],
                        help='Communication pattern.')
    parser.add_argument('--problem', type=str, default='robust', choices=['robust', 'fair_class'],
                        help='Problem setting.')
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar'],
                        help='Dataset to choose from.')
    parser.add_argument('--model', type=str, default='fc', choices=['fc', 'allcnn', 'resnet'],
                        help='Model architecture to choose from.')
    parser.add_argument('--method', type=str, default='dposg', choices=['dposg', 'proposed_spider', 'gtsrvr', 'dmhsgd', 'proposed_storm'],
                        help='Method to use.')
    parser.add_argument('--lrX', type=float, default=1e-1, help='Local learning rate for X.')
    parser.add_argument('--lrY', type=float, default=1e-1, help='Local learning rate for Y.')
    parser.add_argument('--lrLam1', type=float, default=1e-1, help='Lambda 1 learning rate.')
    parser.add_argument('--lrLam2', type=float, default=1e-1, help='Lambda 2 learning rate.')
    parser.add_argument('--betaX', type=float, default=1e-2, help='DM-HSGD gradient lr (x).')
    parser.add_argument('--betaY', type=float, default=1e-2, help='DM-HSGD gradient lr (y).')
    parser.add_argument('--trial', type=int, default=1, help='Which starting variables to use.')
    parser.add_argument('--mini_batch', type=int, default=100, help='Mini-batch size.')
    parser.add_argument('--mega_batch', type=int, default=2000, help='Mega-batch size.')
    parser.add_argument('--frequency', type=int, default=100, help='When to compute the mega batch.')
    parser.add_argument('--eta', type=float, default=0.1, help='Penalty term.')
    parser.add_argument('--report', type=int, default=25, help='How often to report criteria.')

    args = parser.parse_args()

    # ------------------------------------------------
    # Load Data and Transform
    if args.dataset == 'cifar':

        # Transform
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # Subset data to local agent
        num_samples = 50000 // size
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data', train=True, download=False,
                             transform=transform),
            batch_size=args.mini_batch, sampler=torch.utils.data.SubsetRandomSampler(
                [i for i in range(int(rank * num_samples), int((rank + 1) * num_samples))]))

        # Get a full gradient
        full_batch_train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data', train=True, download=False,
                             transform=transform),
            batch_size=num_samples, sampler=torch.utils.data.SubsetRandomSampler(
                [i for i in range(int(rank * num_samples), int((rank + 1) * num_samples))]))

        # Get a mega batch gradient
        mega_batch_train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data', train=True, download=False,
                             transform=transform),
            batch_size=args.mega_batch, sampler=torch.utils.data.SubsetRandomSampler(
                [i for i in range(int(rank * num_samples), int((rank + 1) * num_samples))]))

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data', train=False, download=False,
                           transform=transform),
            batch_size=10000 // size, sampler=torch.utils.data.SubsetRandomSampler(
                [i for i in range(int(rank * (10000 // size)), int((rank + 1) * (10000 // size)))]))

    else:

        # Transform
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Subset data to local agent
        num_samples = 60000 // size
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=False,
                             transform=transform),
            batch_size=args.mini_batch, sampler=torch.utils.data.SubsetRandomSampler(
                [i for i in range(int(rank * num_samples), int((rank + 1) * num_samples))]))

        # Get a full gradient
        full_batch_train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=False,
                             transform=transform),
            batch_size=num_samples, sampler=torch.utils.data.SubsetRandomSampler(
                [i for i in range(int(rank * num_samples), int((rank + 1) * num_samples))]))

        # Get a mega batch gradient
        mega_batch_train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=False,
                             transform=transform),
            batch_size=args.mega_batch, sampler=torch.utils.data.SubsetRandomSampler(
                [i for i in range(int(rank * num_samples), int((rank + 1) * num_samples))]))

        # Get a full gradient
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=False, download=False,
                           transform=transform),
            batch_size=10000 // size, sampler=torch.utils.data.SubsetRandomSampler(
                [i for i in range(int(rank * (10000 // size)), int((rank + 1) * (10000 // size)))]))

    # ------------------------------------------------
    # Set up communication matrix
    mixing_matrix = torch.tensor(numpy.load(f'mixing_matrices/{args.comm_pattern}_{size}.dat', allow_pickle=True))

    # ------------------------------------------------
    # Print training info
    if rank == 0:
        print(f"{'=' * 25} STARTING TRAINING {'=' * 25}")
        print(f'[GRAPH INFO] {size} agents | rho = {round(torch.sort(torch.eig(mixing_matrix)[0][:, 0])[0][-2].item(), 4)}')
        print(f'[TRAINING INFO] mini-batch = {args.mini_batch} | learning rates = ({args.lrX} / {args.lrY})\n')

    # COMMUNICATION BARRIER
    comm.Barrier()

    # ------------------------------------------------
    # Train the model
    algo_params = {'problem': args.problem, 'model': args.model,
                    'lrX': args.lrX, 'lrY': args.lrY,
                   'mini_batch': args.mini_batch, 'frequency': args.frequency,
                   'mega_batch': args.mega_batch,
                   'report': args.report, 'dataset_name': args.dataset,
                   'trial': args.trial, 'eta': args.eta,
                   'lrLam1': args.lrLam1, 'lrLam2': args.lrLam2,
                   'betaX': args.betaX, 'betaY': args.betaY}
    if args.method == 'dposg':
        solver = DPOSG(algo_params, mixing_matrix, train_loader, mega_batch_train_loader, full_batch_train_loader,
                       comm, size, rank)
    elif args.method == 'proposed_spider':
        solver = ProposedSPIDER(algo_params, mixing_matrix, train_loader, mega_batch_train_loader, full_batch_train_loader,
                       comm, size, rank)
    elif args.method == 'proposed_storm':
        solver = ProposedSTORM(algo_params, mixing_matrix, train_loader, mega_batch_train_loader, full_batch_train_loader,
                       comm, size, rank)
    elif args.method == 'gtsrvr':
        solver = GTSRVR(algo_params, mixing_matrix, train_loader, mega_batch_train_loader, full_batch_train_loader,
                          comm, size, rank)
    elif args.method == 'dmhsgd':
        solver = DMHSGD(algo_params, mixing_matrix, train_loader, mega_batch_train_loader, full_batch_train_loader,
                          comm, size, rank)
    else:
        if rank == 0:
            print(f"[ERROR] method {args.method} is not a valid choice.")
        solver = ProposedSTORM(algo_params, mixing_matrix, train_loader, mega_batch_train_loader, full_batch_train_loader,
                       comm, size, rank)
    algo_time = solver.solve(args.iterations, test_loader)

