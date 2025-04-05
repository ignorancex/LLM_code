# note to ming: apply_prox = true in main loop
# note to ming: and add mu to the config code
# in orignal paper, mu set is {0.001, 0.01, 0.1, 1}
import importlib
import numpy as np
import os
import sys
import random
import tensorflow as tf
import metrics.writer as metrics_writer

from baseline_constants import MAIN_PARAMS, MODEL_PARAMS
from client import Client
from server import Server
from model import ServerModel
from utils.constants import DATASETS


STAT_METRICS_PATH = 'metrics/stat_metrics.csv'
SYS_METRICS_PATH = 'metrics/sys_metrics.csv'

def online(clients):
    """We assume all users are always online."""
    return clients

def save_model(server_model, dataset, model):
    """Saves the given server model on checkpoints/dataset/model.ckpt."""
    # Save server model
    ckpt_path = os.path.join('checkpoints', dataset)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    save_path = server_model.save(os.path.join(ckpt_path, '%s.ckpt' % model))
    print('Model saved in path: %s' % save_path)


def print_metrics(metrics, weights):
    ordered_weights = [weights[c] for c in sorted(weights)]
    metric_names = metrics_writer.get_metrics_names(metrics)
    for metric in metric_names:
        ordered_metric = [metrics[c][metric] for c in sorted(metrics)]
        print('%s: %g, 10th percentile: %g, 90th percentile %g' \
              % (metric,
                 np.average(ordered_metric, weights=ordered_weights),
                 np.percentile(ordered_metric, 10),
                 np.percentile(ordered_metric, 90)))
    fom = [metrics[c][metric_names[0]] for c in sorted(metrics)]
    final = np.average(fom, weights=ordered_weights)
    return final

class Fedprox_Trainer:
    
    def __init__(self, users, groups, train_data, test_data):
        self.users = users
        self.train_data = train_data
        self.test_data = test_data
        
    def model_config(self, config, dataset, my_model):   
        shared_model = my_model
        model_path = '%s/%s.py' % (dataset, shared_model)
        if not os.path.exists(model_path):
            print('Please specify a valid dataset and a valid model.')
        model_path = '%s.%s' % (dataset, shared_model)

        print('############################## %s ##############################' % model_path)
        mod = importlib.import_module(model_path)
        ClientModel = getattr(mod, 'ClientProxModel')  
        # Suppress tf warnings
        tf.logging.set_verbosity(tf.logging.WARN)

        # Create 2 models
        model_params = MODEL_PARAMS[model_path]
        model_params_list = list(model_params)
        model_params_list[0] = config["lr"]
        model_params_list[1] = config["mu"]
        model_params = tuple(model_params_list)
        tf.reset_default_graph()
        client_model = ClientModel(config["seed"], *model_params)

        # Create server
        server = Server(client_model)

        # Create clients
        _users = self.users
        groups = [[] for _ in _users]
        clients =  [Client(u, g, self.train_data[u], self.test_data[u], client_model) \
                    for u, g in zip(_users, groups)]
        print('%d Clients in Total' % len(clients)) 
        return clients, server, client_model
    
    def begins(self, config, args):
        clients, server, client_model = self.model_config(config, args.dataset, 'cnn_prox')  
    
        num_rounds = config["num-rounds"]
        eval_every = config["eval-every"]
        epochs_per_round = config['epochs']
        batch_size = config['batch-size']
        clients_per_round = config["clients-per-round"]
        
        # Test untrained model on all clients
        stat_metrics = server.test_model(clients)
        all_ids, all_groups, all_num_samples = server.get_clients_info(clients)
    #     metrics_writer.print_metrics(0, all_ids, stat_metrics, all_groups, all_num_samples, STAT_METRICS_PATH)
        print_metrics(stat_metrics, all_num_samples)

        # Simulate training
        micro_acc = 0.
        for i in range(num_rounds):
            print('--- Round %d of %d: Training %d Clients ---' % (i+1, num_rounds, clients_per_round))

            server.select_clients(online(clients), num_clients=clients_per_round)
            c_ids, c_groups, c_num_samples = server.get_clients_info(None)

            sys_metics = server.train_model(single_center=None, num_epochs=epochs_per_round, batch_size=batch_size, minibatch=None, apply_prox=True)
            server.update_model_wmode()

            # Test model on all clients
            if (i + 1) % eval_every == 0 or (i + 1) == num_rounds:
                stat_metrics = server.test_model(clients)
                micro_acc = print_metrics(stat_metrics, all_num_samples)

        # Save server model
    #     save_model(server_model, args.dataset, shared_model)

        # Close models
    #     server_model.close()
        client_model.close() 
        return micro_acc
    
    def ends(self):
        print("experiment of Fedprox finished.")
        return
