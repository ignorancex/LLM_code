import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy
import torch
import argparse
import warnings
import random
import time

from clients.SGRepClassifier import *
from servers.FederatedServerClassifier import *
from plots.plots import*
from read_files import *
import constants
from utils.utils import *
from datasets.dataset_medmnist import *

from collections import Counter

warnings.filterwarnings("ignore")

torch.cuda.empty_cache()

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("running on GPU")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("running on GPU MPS")
else:
    device = torch.device("cpu")
    print("running on CPU")

random.seed(constants.MANUAL_SEED)
np.random.seed(constants.MANUAL_SEED)
torch.manual_seed(constants.MANUAL_SEED)

def run(args):
    create_directories_for_saving()

    if constants.ALGORITHM == "baseline":
        run_baseline(args)
    elif constants.ALGORITHM == "fedavg":
        run_fedavg(args)
    elif constants.ALGORITHM == "feddyn":
        run_feddyn(args)
    elif constants.ALGORITHM == "fedrep":
        run_repfl(args)
    elif constants.ALGORITHM == "feddc":
        run_feddc(args)
    elif constants.ALGORITHM == "centralized":
        run_centralized(args)




def plot_losses(losses, method):
    for client in range(constants.NUMBER_CLIENTS):
        global_loss = [losses[fold][client]['global_test1'] for fold in range(constants.NUMBER_FOLDS)]
        global_loss = numpy.array(global_loss)
        average_global_loss = numpy.mean(global_loss, axis = 0)

        # current_client_loss = {'global': average_global_loss}
        plot_one_loss(average_global_loss, client, method=method, current_run_name=constants.RUN_NAME)

def plot_losses_one_fold(losses, method, fold=0):
    for client in range(constants.NUMBER_CLIENTS):
        global_loss = [losses[fold][client]['global_test1']]
        global_loss = numpy.array(global_loss)
        average_global_loss = numpy.mean(global_loss, axis = 0)

        # current_client_loss = {'global': average_global_loss}
        plot_one_loss(average_global_loss, client, method=method, current_run_name=constants.RUN_NAME)


def test_method(models_fed, args):
    train_dataset, val_dataset, test_dataset = get_medmnist_dataset(balanced=args.balanced)

    fold_indices = generate_fold_indices_dataset(train_dataset)
    for fold_i in fold_indices:
        print(len(fold_i))
    fold_configuration = generate_fold_configurations(constants.NUMBER_FOLDS)

    metrics = ['acc', 'sens', 'spec', 'f1']

    for fold in range(constants.NUMBER_FOLDS):
        current_fold_configuration = fold_configuration[fold]
        test_index = fold_indices[current_fold_configuration[-1]]

        test_set = Subset(train_dataset, test_index)
        test_loader = DataLoader(test_set, batch_size=constants.BATCH_SIZE, shuffle=False)

        for client in range(constants.NUMBER_CLIENTS):
            acc_current = models_fed[fold][client].test(test_loader)
            print(acc_current)
            print("\n")

        for current_metric in metrics:
            save_metric_to_file(models_fed, fold, current_run_name=constants.RUN_NAME, metric=current_metric)

    for current_metric in metrics:
        save_average_metric_to_file(models_fed, current_run_name=constants.RUN_NAME, metric=current_metric)


def test_method_one_fold(models_fed, args, fold=0):
    train_dataset, val_dataset, test_dataset = get_medmnist_dataset(balanced=args.balanced)

    fold_indices = generate_fold_indices_dataset(train_dataset)
    for fold_i in fold_indices:
        print(len(fold_i))
    fold_configuration = generate_fold_configurations(constants.NUMBER_FOLDS)

    metrics = ['acc', 'sens', 'spec', 'f1']

    current_fold_configuration = fold_configuration[fold]
    test_index = fold_indices[current_fold_configuration[-1]]

    test_set = Subset(train_dataset, test_index)
    test_loader = DataLoader(test_set, batch_size=constants.BATCH_SIZE, shuffle=False)

    for client in range(constants.NUMBER_CLIENTS):
        metrics = models_fed[fold][client].test(test_loader)
        print(metrics)
        print("\n")

    for current_metric in metrics:
        save_metric_to_file(models_fed, fold, current_run_name=constants.RUN_NAME, metric=current_metric)




# # ************   Centralized   ************

def run_centralized(args):
    train_dataset, val_dataset, test_dataset = get_medmnist_dataset(balanced=args.balanced)
    print(len(train_dataset))

    fold_indices = generate_fold_indices_dataset(train_dataset)
    for fold_i in fold_indices:
        print(len(fold_i))
    fold_configuration = generate_fold_configurations(constants.NUMBER_FOLDS)

    losses_fed = initialize_losses_arrays()
    models_fed = []
    for fold in range(constants.NUMBER_FOLDS):
        print(f'Centralized Fold {fold}')
        current_fold_configuration = fold_configuration[fold]

        models_fed.append([])
        train_index = []
        # all client indices are part of training set
        for client in range(constants.NUMBER_FOLDS - 1):
            train_index.extend(fold_indices[current_fold_configuration[client]])
        
        train_set = Subset(train_dataset, train_index)

        models_fed[fold].append(SGRepClassifier(fold, 0, "Baseline", no_classes=constants.NUMBER_OF_CLASSES))
        models_fed[fold][0].set_training_data(train_set)

        for round in range(constants.NUMBER_OF_ROUNDS):
            # only one client
            global_loss_train = models_fed[fold][0].run_one_round()
            losses_fed[fold][0]['global_test1'].extend(global_loss_train)

        models_fed[fold][0].save_model()
        save_losses_to_file(losses_fed, fold, 'global_test1')
    
    test_method(models_fed, args)
    plot_losses(losses_fed, method="centralized")




# # ************   Baseline   ************

def run_baseline(args):
    train_dataset, val_dataset, test_dataset = get_medmnist_dataset(balanced=args.balanced)

    fold_indices = generate_fold_indices_dataset(train_dataset)
    for fold_i in fold_indices:
        print(len(fold_i))
    fold_configuration = generate_fold_configurations(constants.NUMBER_FOLDS)

    losses_fed = initialize_losses_arrays()
    models_fed = []
    for fold in range(constants.NUMBER_FOLDS):
        print(f'Baseline Fold {fold}')
        current_fold_configuration = fold_configuration[fold]

        models_fed.append([])
        train_datasets = []
        for client in range(constants.NUMBER_CLIENTS):
            client_train_index = fold_indices[current_fold_configuration[client]]
            
            train_set = Subset(train_dataset, client_train_index)
            train_datasets.append(train_set)

            models_fed[fold].append(SGRepClassifier(fold, client, "Baseline", no_classes=constants.NUMBER_OF_CLASSES))
            models_fed[fold][client].set_training_data(train_set)

        for round in range(constants.NUMBER_OF_ROUNDS):
            for client in range(constants.NUMBER_CLIENTS):
                print(f'Baseline Fold {fold} Client {client}')
                global_loss_train = models_fed[fold][client].run_one_round()

                losses_fed[fold][client]['global_test1'].extend(global_loss_train)

        for client in range(constants.NUMBER_CLIENTS):
            models_fed[fold][client].save_model()

        save_losses_to_file(losses_fed, fold, 'global_test1')
    
    test_method(models_fed, args)
    plot_losses(losses_fed, method="baseline")




# # ************   FedAvg   ************

def run_fedavg(args):
    train_dataset, val_dataset, test_dataset = get_medmnist_dataset(balanced=args.balanced)
    print(len(train_dataset))
    # for input, target in train_dataset:
    #     print(input)
    #     print(target)
    #     print('\n')

    fold_indices = generate_fold_indices_dataset(train_dataset)
    for fold_i in fold_indices:
        print(len(fold_i))
    fold_configuration = generate_fold_configurations(constants.NUMBER_FOLDS)

    losses_fed = initialize_losses_arrays()
    models_fed = []
    for fold in range(constants.NUMBER_FOLDS):
    # for fold in range(1):
        print(f'Federated Fold {fold}')
        federated_server = FederatedServerClassifier(fold, no_classes=constants.NUMBER_OF_CLASSES)
        global_model_initial_stat_dict = federated_server.get_global_model()
        current_fold_configuration = fold_configuration[fold]

        models_fed.append([])
        train_datasets = []
        for client in range(constants.NUMBER_CLIENTS):
            client_train_index = fold_indices[current_fold_configuration[client]]
            
            train_set = Subset(train_dataset, client_train_index)
            train_datasets.append(train_set)

            models_fed[fold].append(SGRepClassifier(fold, client, fed_alg="FedAvg", no_classes=constants.NUMBER_OF_CLASSES))
            models_fed[fold][client].initialize_model_parameters(global_model_initial_stat_dict.state_dict())
            models_fed[fold][client].set_training_data(train_set)

        for round in range(constants.NUMBER_OF_ROUNDS):
            for client in range(constants.NUMBER_CLIENTS):
                print(f'Federated Fold {fold} Client {client}')
                global_loss_train = models_fed[fold][client].run_one_round()

                losses_fed[fold][client]['global_test1'].extend(global_loss_train)

            global_model = federated_server.federated_average_intermodality(models_fed[fold])
            if round < constants.NUMBER_OF_ROUNDS - 1:
                for client in range(constants.NUMBER_CLIENTS):
                    models_fed[fold][client].update_parameters(copy.deepcopy(global_model))

        for client in range(constants.NUMBER_CLIENTS):
            models_fed[fold][client].save_model()
        federated_server.save_model()

        save_losses_to_file(losses_fed, fold, 'global_test1')
    
    test_method(models_fed, args)
    plot_losses(losses_fed, method="fedavg")




# # ************   FedDyn   ************

def run_feddyn(args):
    train_dataset, val_dataset, test_dataset = get_medmnist_dataset(balanced=args.balanced)

    fold_indices = generate_fold_indices_dataset(train_dataset)
    for fold_i in fold_indices:
        print(len(fold_i))
    fold_configuration = generate_fold_configurations(constants.NUMBER_FOLDS)

    losses_fed = initialize_losses_arrays()
    models_fed = []
    for fold in range(constants.NUMBER_FOLDS):
        print(f'FedDyn Fold {fold}')
        federated_server = FederatedServerClassifier(fold, no_classes=constants.NUMBER_OF_CLASSES)
        global_model_initial_stat_dict = federated_server.get_global_model()
        current_fold_configuration = fold_configuration[fold]

        models_fed.append([])
        train_datasets = []
        for client in range(constants.NUMBER_CLIENTS):
            client_train_index = fold_indices[current_fold_configuration[client]]
            
            train_set = Subset(train_dataset, client_train_index)
            train_datasets.append(train_set)

            models_fed[fold].append(SGRepClassifier(fold, client, fed_alg="FedDyn", no_classes=constants.NUMBER_OF_CLASSES))
            models_fed[fold][client].initialize_model_parameters(global_model_initial_stat_dict.state_dict())
            models_fed[fold][client].set_training_data(train_set)

        for round in range(constants.NUMBER_OF_ROUNDS):
            global_model = federated_server.get_global_model()
            global_model_param = federated_server.get_global_model_params()

            for client in range(constants.NUMBER_CLIENTS):
                print(f'FedDyn Fold {fold} Client {client}')

                # Start all clients from the global model
                models_fed[fold][client].update_parameters(global_model)
                models_fed[fold][client].set_global_model_params(global_model_param)

                global_loss_train = models_fed[fold][client].run_one_round()

                losses_fed[fold][client]['global_test1'].extend(global_loss_train)

            federated_server.update_global_model_feddyn(models_fed[fold])

        for client in range(constants.NUMBER_CLIENTS):
            models_fed[fold][client].save_model()
        federated_server.save_model()

        save_losses_to_file(losses_fed, fold, 'global_test1')
    
    test_method(models_fed, args)
    plot_losses(losses_fed, method="feddyn")


def run_repfl(args):
    train_dataset, val_dataset, test_dataset = get_medmnist_dataset(balanced=args.balanced)

    fold_indices = generate_fold_indices_dataset(train_dataset)
    for fold_i in fold_indices:
        print(len(fold_i))
    fold_configuration = generate_fold_configurations(constants.NUMBER_FOLDS)

    losses_fed = initialize_losses_arrays()
    models_fed = []
    for fold in range(constants.NUMBER_FOLDS):
    # for fold in range(3, 4):
        print(f'FedRep Fold {fold}')
        federated_server = FederatedServerClassifier(fold, no_classes=constants.NUMBER_OF_CLASSES)
        global_model_initial_stat_dict = federated_server.get_global_model()
        current_fold_configuration = fold_configuration[fold]
        
        models_fed.append([])
        models_fed.append([])
        models_fed.append([])
        models_fed.append([])
        train_datasets = []
        for client in range(constants.NUMBER_CLIENTS):
            client_train_index = fold_indices[current_fold_configuration[client]]

            train_set = Subset(train_dataset, client_train_index)
            train_datasets.append(train_set)

            # Create anchor model
            models_fed[fold].append(SGRepClassifier(fold, client, fed_alg="FedRep", no_classes=constants.NUMBER_OF_CLASSES))
            # print(f"Initializing model client {client}")

            models_fed[fold][client].initialize_model_parameters(global_model_initial_stat_dict.state_dict())
            models_fed[fold][client].set_training_data(train_set, no_perturbed_samples = 0, offset_perturbed_samples = 0, stratified=args.strat)

        start_time = time.time()
        for round in range(constants.NUMBER_OF_ROUNDS):
            all_models_to_aggregate = []
            for client in range(constants.NUMBER_CLIENTS):
                print(f'FedRep Fold {fold} Client {client}')
                global_loss_train = models_fed[fold][client].run_one_round()

                losses_fed[fold][client]['global_test1'].extend(global_loss_train)

            if constants.AGGREGATION == "simple":
                global_model = federated_server.federated_average_intermodality(models_fed[fold])
            else:
                global_model = federated_server.federated_diversity(models_fed[fold])

            if round < constants.NUMBER_OF_ROUNDS - 1:
                for client in range(constants.NUMBER_CLIENTS):
                    models_fed[fold][client].update_parameters(global_model)
        
        end_time = time.time() - start_time
        print(f"\nTotal time cost for one fold: {end_time}s.")
        # return

        for client in range(constants.NUMBER_CLIENTS):
            models_fed[fold][client].save_model()
        federated_server.save_model()

        save_losses_to_file(losses_fed, fold, 'global_test1')

    test_method(models_fed, args)
    plot_losses(losses_fed, method="fedrep")


# # ************   FedDC   ************

def run_feddc(args):
    train_dataset, val_dataset, test_dataset = get_medmnist_dataset(balanced=args.balanced)

    fold_indices = generate_fold_indices_dataset(train_dataset)
    for fold_i in fold_indices:
        print(len(fold_i))
    fold_configuration = generate_fold_configurations(constants.NUMBER_FOLDS)

    losses_fed = initialize_losses_arrays()
    models_fed = []
    for fold in range(constants.NUMBER_FOLDS):
        print(f'FedDC Fold {fold}')
        federated_server = FederatedServerClassifier(fold, no_classes=constants.NUMBER_OF_CLASSES)
        global_model_initial_stat_dict = federated_server.get_global_model()
        current_fold_configuration = fold_configuration[fold]

        models_fed.append([])
        train_datasets = []
        for client in range(constants.NUMBER_CLIENTS):
            client_train_index = fold_indices[current_fold_configuration[client]]
            
            train_set = Subset(train_dataset, client_train_index)
            train_datasets.append(train_set)

            models_fed[fold].append(SGRepClassifier(fold, client, fed_alg="FedDC", no_classes=constants.NUMBER_OF_CLASSES))
            models_fed[fold][client].initialize_model_parameters(global_model_initial_stat_dict.state_dict())
            models_fed[fold][client].set_training_data(train_set)

        for epoch in range(constants.NUMBER_OF_ROUNDS):
            for client in range(constants.NUMBER_CLIENTS):
                print(f'Federated Fold {fold} Client {client}')
                global_loss_train = models_fed[fold][client].train()

                losses_fed[fold][client]['global_test1'].extend(global_loss_train)

            if epoch % constants.DC_ROUND == constants.DC_ROUND - 1:

                new_models = federated_server.federated_daisy_chain(models_fed[fold])
                for client in range(constants.NUMBER_CLIENTS):
                    models_fed[fold][client].update_parameters(new_models[client])
            
            if epoch % constants.AGG_ROUND == constants.AGG_ROUND - 1:
                # print(f"avg round epoch: {epoch}")
                global_model = federated_server.federated_average_intermodality(models_fed[fold])

                if epoch % constants.AGG_ROUND < int(NUMBER_OF_ROUNDS / 10) - 1:
                    for client in range(constants.NUMBER_CLIENTS):
                        models_fed[fold][client].update_parameters(copy.deepcopy(global_model))

        for client in range(constants.NUMBER_CLIENTS):
            models_fed[fold][client].save_model()
        federated_server.save_model()

        save_losses_to_file(losses_fed, fold, 'global_test1')
    
    test_method(models_fed, args)
    plot_losses(losses_fed, method="feddc")


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()

    # general
    parser.add_argument('-alg', "--algorithm", default="baseline")
    parser.add_argument('-seed', "--seed", type=int, default=1773)
    parser.add_argument('-le', "--local_epochs", type=int, default=10)
    parser.add_argument('-r', "--rounds", type=int, default=10)
    parser.add_argument('-batch', "--batch_size", type=int, default=21)
    parser.add_argument('-folds', "--folds", type=int, default=4)
    parser.add_argument('-clients', "--clients", type=int, default=3)

    # dataset
    parser.add_argument('-dataset', "--dataset", default="pneumoniamnist")
    parser.add_argument('-classes', "--no_classes", type=int, default=2)
    parser.add_argument('-dataset_size', "--desired_dataset_subset_size", type=int, default=800)
    parser.add_argument('-balanced', "--balanced", type=bool, default=False)
    parser.add_argument('-strat', "--strat", type=bool, default=False)

    # directories
    parser.add_argument('-global_run_name', "--global_run_name", default="global_run_name")
    parser.add_argument('-run_name', "--run_name", default="run_name")
    parser.add_argument('-paths', "--paths", default="local")

    # FedRep
    parser.add_argument('-perturbed', "--no_perturbed_samples", type=int, default=10)
    parser.add_argument('-replicas', "--no_replicas", type=int, default=3)
    parser.add_argument('-depth', "--replica_depth", type=int, default=2)
    parser.add_argument('-agg', "--aggregation", type=str, default="simple")
    parser.add_argument('-div', "--div_metric", type=str, default="l2")

    # FedDyn
    parser.add_argument('-alpha', "--alpha", type=float, default=0.01)

    args = parser.parse_args()

    constants.ALGORITHM = args.algorithm
    constants.MANUAL_SEED = args.seed
    constants.NUM_LOCAL_EPOCHS = args.local_epochs
    constants.NUMBER_OF_ROUNDS = args.rounds
    constants.BATCH_SIZE = args.batch_size
    constants.NUMBER_FOLDS = args.folds
    constants.NUMBER_CLIENTS = args.clients
    constants.MEDMNIST_DATASET = args.dataset
    constants.NUMBER_OF_CLASSES = args.no_classes
    constants.DESIRED_DATASET_SUBSET_SIZE = args.desired_dataset_subset_size
    constants.GLOBAL_RUN_NAME = args.global_run_name
    constants.RUN_NAME = args.run_name
    if args.paths == "local":
        constants.DATASET_FOLDER_PATH = constants.DATASET_FOLDER_PATH_LOCAL
        constants.SAVING_FOLDER_PATH = constants.SAVING_FOLDER_PATH_LOCAL
    else:
        constants.DATASET_FOLDER_PATH = constants.DATASET_FOLDER_PATH_REMOTE
        constants.SAVING_FOLDER_PATH = constants.SAVING_FOLDER_PATH_REMOTE
    constants.NUMBER_PERTURBED_SAMPLES = args.no_perturbed_samples
    constants.NUMBER_REPLICAS = args.no_replicas
    if constants.ALGORITHM != "fedrep":
        constants.MAX_REPLICA_DEPTH = 0
    else:
        constants.MAX_REPLICA_DEPTH = args.replica_depth
    constants.AGGREGATION = args.aggregation
    constants.DIV_METRIC = args.div_metric
    constants.ALPHA_COEF = args.alpha

    print("=" * 50)
    print("Algorithm: {}".format(args.algorithm))
    print("Random seed: {}".format(args.seed))
    print("Local epochs: {}".format(args.local_epochs))
    print("Rounds: {}".format(args.rounds))
    print("Batch: {}".format(args.batch_size))
    print("Folds: {}".format(args.folds))
    print("Clients: {}".format(args.clients))
    print("Dataset: {}".format(args.dataset))
    print("Classes: {}".format(args.no_classes))
    print("Dataset subset size: {}".format(args.desired_dataset_subset_size))
    print("Balanced: {}".format(args.balanced))
    print("Stratified: {}".format(args.strat))
    print("Global run name: {}".format(args.global_run_name))
    print("Run name: {}".format(args.run_name))
    print("Paths: {}".format(args.paths))
    print("Saving folder path: {}".format(constants.SAVING_FOLDER_PATH))
    print("Perturbed: {}".format(args.no_perturbed_samples))
    print("Replicas: {}".format(args.no_replicas))
    print("Depth: {}".format(constants.MAX_REPLICA_DEPTH))
    print("Aggregation: {}".format(constants.AGGREGATION))
    print("Div Metric: {}".format(constants.DIV_METRIC))
    print("Alpha: {}".format(args.alpha))

    print("=" * 50)

    create_directories_for_saving()
    write_details_to_file(args, constants.RUN_NAME)

    run(args)

    time_cost = round(time.time()-total_start, 2)
    print(f"\nTotal time cost: {time_cost}s.")
    write_time_to_file(time_cost, constants.RUN_NAME)