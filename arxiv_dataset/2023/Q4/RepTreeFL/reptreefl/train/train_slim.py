import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import torch
import argparse
import warnings
import random
import time
from utils.preprocess import*
from clients.SGRepGenerator import *
from servers.FederatedServer import *
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
    source_data, target_data1, target_data2 = read_and_preprocess_files()
    fold_indices = generate_fold_indices(source_data)
    fold_configuration = generate_fold_configurations(NUMBER_FOLDS)
    client_resolution = [160, 160, 268, 268]

    for fold in range(constants.NUMBER_FOLDS):
        current_fold_configuration = fold_configuration[fold]
        test_index = fold_indices[current_fold_configuration[-1]]

        X_test_source, X_test_target1, X_test_target2 = source_data[test_index], target_data1[test_index], target_data2[test_index]
        
        for client in range(constants.NUMBER_CLIENTS):
            if client_resolution[client] == 160:
                models_fed[fold][client].test(X_test_source, X_test_target1)
            elif client_resolution[client] == 268:
                models_fed[fold][client].test(X_test_source, X_test_target2)

        save_mae_to_file(models_fed, fold, constants.RUN_NAME)




# # ************   Centralized   ************

def run_centralized(args):
    source_data, target_data1, target_data2 = read_and_preprocess_files()
    fold_indices = generate_fold_indices(source_data)
    fold_configuration = generate_fold_configurations(constants.NUMBER_FOLDS)
    client_resolution = [160, 160, 268, 268]

    losses = initialize_losses_arrays()
    models = []
    for fold in range(constants.NUMBER_FOLDS):
        print(f'Centralized Fold {fold}')
        current_fold_configuration = fold_configuration[fold]
        test_index = fold_indices[current_fold_configuration[-1]]

        X_test_source, X_test_target1, X_test_target2 = source_data[test_index], target_data1[test_index], target_data2[test_index]
        
        models.append([])
        train_index = []
        # all client indices are part of training set
        for client in range(constants.NUMBER_FOLDS - 1):
            train_index.extend(fold_indices[current_fold_configuration[client]])
        
        X_train_source, X_train_target1, X_train_target2 = source_data[train_index], target_data1[train_index], target_data2[train_index]

        models[fold].append(SGRepGenerator(268, fold, 0, "Baseline"))

        for round in range(NUMBER_OF_ROUNDS):
            global_loss_train = models[fold][0].run_one_round(X_train_source, X_train_target2)
            losses[fold][0]['global_test1'].extend(global_loss_train)

        models[fold][0].save_model()
        save_losses_to_file(losses, fold, 'global_test1')

        models[fold][0].test(X_test_source, X_test_target2)
        save_mae_to_file(models, fold)

    save_average_mae_to_file(models)




# # ************   Baseline   ************

def run_baseline(args):
    source_data, target_data1, target_data2 = read_and_preprocess_files()
    fold_indices = generate_fold_indices(source_data)
    fold_configuration = generate_fold_configurations(constants.NUMBER_FOLDS)
    client_resolution = [160, 160, 268, 268]

    losses = initialize_losses_arrays()
    models = []
    for fold in range(constants.NUMBER_FOLDS):
        print(f'Baseline Fold {fold}')
        current_fold_configuration = fold_configuration[fold]

        models.append([])
        for client in range(constants.NUMBER_CLIENTS):
            client_train_index = fold_indices[current_fold_configuration[client]]
            X_train_source, X_train_target1, X_train_target2 = source_data[client_train_index], target_data1[client_train_index], target_data2[client_train_index]

            models[fold].append(SGRepGenerator(client_resolution[client], fold, client, fed_alg="Baseline"))
            if client_resolution[client] == 160:
                models[fold][client].set_training_data(X_train_source, X_train_target1)
            elif client_resolution[client] == 268:
                models[fold][client].set_training_data(X_train_source, X_train_target2)
       
        for current_round in range(constants.NUMBER_OF_ROUNDS):
            for client in range(constants.NUMBER_CLIENTS):
                print(f'Baseline Fold {fold} Client {client}')
                global_loss_train = models[fold][client].run_one_round()

                losses[fold][client]['global_test1'].extend(global_loss_train)

        for client in range(constants.NUMBER_CLIENTS):
            models[fold][client].save_model()

        save_losses_to_file(losses, fold, 'global_test1')
    
    test_method(models, args)
    plot_losses(models, method="baseline")




# # ************   FedAvg   ************

def run_fedavg(args):
    source_data, target_data1, target_data2 = read_and_preprocess_files()
    fold_indices = generate_fold_indices(source_data)
    fold_configuration = generate_fold_configurations(constants.NUMBER_FOLDS)
    client_resolution = [160, 160, 268, 268]

    losses_fed = initialize_losses_arrays()
    models_fed = []
    for fold in range(constants.NUMBER_FOLDS):
        print(f'Federated Fold {fold}')
        federated_server = FederatedServer(fold)
        current_fold_configuration = fold_configuration[fold]

        models_fed.append([])
        for client in range(NUMBER_CLIENTS):
            client_train_index = fold_indices[current_fold_configuration[client]]
            X_train_source, X_train_target1, X_train_target2 = source_data[client_train_index], target_data1[client_train_index], target_data2[client_train_index]

            models_fed[fold].append(SGRepGenerator(client_resolution[client], fold, client, fed_alg="FedAvg"))
            if client_resolution[client] == 160:
                models_fed[fold][client].set_training_data(X_train_source, X_train_target1)
            elif client_resolution[client] == 268:
                models_fed[fold][client].set_training_data(X_train_source, X_train_target2)
       
        for current_round in range(NUMBER_OF_ROUNDS):
            for client in range(NUMBER_CLIENTS):
                print(f'Federated Fold {fold} Client {client}')
                global_loss_train = models_fed[fold][client].run_one_round()

                losses_fed[fold][client]['global_test1'].extend(global_loss_train)

            global_model = federated_server.federated_average_intermodality(models_fed[fold])
            if round < constants.NUMBER_OF_ROUNDS - 1:
                for client in range(constants.NUMBER_CLIENTS):
                    models_fed[fold][client].update_parameters(copy.deepcopy(global_model))

        for client in range(constants.NUMBER_CLIENTS):
            models_fed[fold][client].save_model()

        save_losses_to_file(losses_fed, fold, 'global_test1')
    
    test_method(models_fed, args)
    plot_losses(losses_fed, method="fedavg")



# # ************   FedDyn   ************

def run_feddyn(args):
    source_data, target_data1, target_data2 = read_and_preprocess_files()
    fold_indices = generate_fold_indices(source_data)
    fold_configuration = generate_fold_configurations(constants.NUMBER_FOLDS)
    client_resolution = [160, 160, 268, 268]

    losses_fed = initialize_losses_arrays()
    models_fed = []
    for fold in range(constants.NUMBER_FOLDS):
        print(f'FedDyn Fold {fold}')
        federated_server = FederatedServer(fold)
        current_fold_configuration = fold_configuration[fold]

        models_fed.append([])
        for client in range(NUMBER_CLIENTS):
            client_train_index = fold_indices[current_fold_configuration[client]]
            X_train_source, X_train_target1, X_train_target2 = source_data[client_train_index], target_data1[client_train_index], target_data2[client_train_index]

            models_fed[fold].append(SGRepGenerator(client_resolution[client], fold, client, fed_alg="FedDyn"))
            if client_resolution[client] == 160:
                models_fed[fold][client].set_training_data(X_train_source, X_train_target1)
            elif client_resolution[client] == 268:
                models_fed[fold][client].set_training_data(X_train_source, X_train_target2)
       
        for current_round in range(NUMBER_OF_ROUNDS):
            for client in range(NUMBER_CLIENTS):
                print(f'Federated Fold {fold} Client {client}')
                global_loss_train = models_fed[fold][client].run_one_round()

                losses_fed[fold][client]['global_test1'].extend(global_loss_train)

            federated_server.update_global_model_feddyn(models_fed[fold])

        for client in range(constants.NUMBER_CLIENTS):
            models_fed[fold][client].save_model()

        save_losses_to_file(losses_fed, fold, 'global_test1')

    test_method(models_fed, args)
    plot_losses(losses_fed, method="feddyn")


# # ************   RepFL   ************

def run_repfl(args):
    source_data, target_data1, target_data2 = read_and_preprocess_files()
    fold_indices = generate_fold_indices(source_data)
    fold_configuration = generate_fold_configurations(constants.NUMBER_FOLDS)
    client_resolution = [160, 160, 268, 268]

    losses_fed = initialize_losses_arrays()
    models_fed = []
    for fold in range(constants.NUMBER_FOLDS):
        print(f'Federated Fold {fold}')
        federated_server = FederatedServer(fold)
        current_fold_configuration = fold_configuration[fold]
        
        models_fed.append([])
        for client in range(constants.NUMBER_CLIENTS):
            client_train_index = fold_indices[current_fold_configuration[client]]
            X_train_source, X_train_target1, X_train_target2 = source_data[client_train_index], target_data1[client_train_index], target_data2[client_train_index]

            # Create anchor model
            models_fed[fold].append(SGRepGenerator(client_resolution[client], fold, client, fed_alg="FedRep"))
            if client_resolution[client] == 160:
                models_fed[fold][client].set_training_data(X_train_source, X_train_target1, no_perturbed_samples = 0, offset_perturbed_samples = 0)
            elif client_resolution[client] == 268:
                models_fed[fold][client].set_training_data(X_train_source, X_train_target2, no_perturbed_samples = 0, offset_perturbed_samples = 0)


        for current_round in range(constants.NUMBER_OF_ROUNDS):
            all_models_to_aggregate = []
            for client in range(constants.NUMBER_CLIENTS):
                print(f'Federated Fold {fold} Client {client}')
                l1_loss_train, global_loss_train = models_fed[fold][client].run_one_round()

                losses_fed[fold][client]['l1_test1'].extend(l1_loss_train)
                losses_fed[fold][client]['global_test1'].extend(global_loss_train)

            if constants.AGGREGATION == "simple":
                global_model = federated_server.federated_average_intermodality(models_fed[fold])
            else:
                global_model = federated_server.federated_diversity(models_fed[fold])

            if current_round < constants.NUMBER_OF_ROUNDS - 1:
                for client in range(constants.NUMBER_CLIENTS):
                    models_fed[fold][client].update_parameters(global_model)

        for client in range(constants.NUMBER_CLIENTS):
            models_fed[fold][client].save_model()

        save_losses_to_file(losses_fed, fold, 'global_test1')

    test_method(models_fed, args)
    plot_losses(losses_fed, method="fedrep")


# # ************   FedDC   ************

def run_feddc(args):
    source_data, target_data1, target_data2 = read_and_preprocess_files()
    fold_indices = generate_fold_indices(source_data)
    fold_configuration = generate_fold_configurations(constants.NUMBER_FOLDS)
    client_resolution = [160, 160, 268, 268]

    losses_fed = initialize_losses_arrays()
    models_fed = []
    for fold in range(NUMBER_FOLDS):
        print(f'FedDC Fold {fold}')
        federated_server = FederatedServer(fold)
        current_fold_configuration = fold_configuration[fold]
        test_index = fold_indices[current_fold_configuration[-1]]

        X_test_source, X_test_target1, X_test_target2 = source_data[test_index], target_data1[test_index], target_data2[test_index]
        
        models_fed.append([])
        train_datasets = []
        for client in range(NUMBER_CLIENTS):
            client_train_index = fold_indices[current_fold_configuration[client]]
            X_train_source, X_train_target1, X_train_target2 = source_data[client_train_index], target_data1[client_train_index], target_data2[client_train_index]
            train_datasets.append([X_train_source, X_train_target1, X_train_target2])

            models_fed[fold].append(SGRepGenerator(client_resolution[client], fold, client, "FedDC"))

        for epoch in range(NUMBER_OF_ROUNDS):
            for client in range(NUMBER_CLIENTS):
                # print(f'FedDC Fold {fold} Client {client}')
                X_train_source, X_train_target1, X_train_target2 = train_datasets[client]
                if client_resolution[client] == 160:
                    l1_loss_train, global_loss_train = models_fed[fold][client].train(X_train_source, X_train_target1)
                elif client_resolution[client] == 268:
                    l1_loss_train, global_loss_train = models_fed[fold][client].train(X_train_source, X_train_target2)

                losses_fed[fold][client]['l1_test1'].extend(l1_loss_train)
                losses_fed[fold][client]['global_test1'].extend(global_loss_train)

            if epoch % DC_ROUND == DC_ROUND - 1:
                new_models = federated_server.federated_daisy_chain(models_fed[fold])
                for client in range(NUMBER_CLIENTS):
                    models_fed[fold][client].update_parameters_daisy_chain(new_models[client])

            if epoch % AGG_ROUND == AGG_ROUND - 1:
                # print(f"avg round epoch: {epoch}")
                print(f'FedDC Fold {fold} Client {client} Epoch {epoch}')
                global_model = federated_server.federated_average_intermodality_two_models(models_fed[fold])

                if epoch % AGG_ROUND < int(NUMBER_OF_ROUNDS / 10) - 1:
                    for client in range(NUMBER_CLIENTS):
                        models_fed[fold][client].update_parameters(copy.deepcopy(global_model))

        for client in range(NUMBER_CLIENTS):
            models_fed[fold][client].save_model()
        federated_server.save_model()

        save_losses_to_file(losses_fed, fold, 'global_test1')

    test_method(models_fed, args)
    plot_losses(losses_fed, method="fedrep")


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()

    # general
    parser.add_argument('-alg', "--algorithm", default="baseline")
    parser.add_argument('-seed', "--seed", type=int, default=1773)
    parser.add_argument('-le', "--local_epochs", type=int, default=10)
    parser.add_argument('-r', "--rounds", type=int, default=10)
    parser.add_argument('-batch', "--batch_size", type=int, default=5)
    parser.add_argument('-folds', "--folds", type=int, default=5)
    parser.add_argument('-clients', "--clients", type=int, default=4)

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