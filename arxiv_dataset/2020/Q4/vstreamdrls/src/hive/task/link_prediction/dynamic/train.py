import argparse
import sys
import torch
import torch.nn as nn
import numpy as np
import os
import time
from os import path

from hive.algorithm.gnn.vstreamdrls.model import VStreamDRLSModel
from hive.algorithm.gnn.evolvegcn.model import EvolveGCNModel
from hive.data.transformation.single.graph.dynamic.transforms import Compose
from hive.task.link_prediction.static.evaluate import Evaluation
from azureml.core import Run

device = 'cuda' if torch.cuda.is_available() else 'cpu'
run = Run.get_context()

def vstreadrls(args):
    threshold = 20
    w = args.window
    dataset = Compose(args.data_path, args.dataset_file, device, model="VSTREAMDRLS")
    minutes = dataset.minutes
    print(f"Number of Nodes {dataset.num_nodes}")

    exp_val_mae = []
    exp_val_rmse = []
    exp_test_mae = []
    exp_test_rmse = []
    for exp in range(args.num_runs):

        val_mae_run = []
        val_rmse_run = []

        test_mae_run = []
        test_rmse_run = []

        for i in range(100):
            # print(f"Experiment {exp} Timestep {i}/{len(minutes)}")
            start_graph = i - w if (i-w) > 0 else 0
            end_graph = i
            train_data, val_data, test_data = dataset.split_data(ns=args.ns, start_minute=start_graph, end_minute=end_graph)

            adj_lists = [data['adj'] for data in train_data]
            adj_norm = train_data[-1]['adj_norm']
            features = train_data[-1]['features']
            train_edges_indices = train_data[-1]['indices']
            train_edges_values = train_data[-1]['values']

            val_edges_indices = val_data['indices']
            val_edges_values = val_data['values']

            test_edges_indices = test_data['indices']
            test_edges_values = test_data['values']

            if train_edges_indices[0].shape[0] > threshold and val_edges_indices[0].shape[0] > threshold and test_edges_indices[0].shape[0]:
                model = VStreamDRLSModel(layer_1_in=dataset.num_nodes, layer_1_out=args.emb_dim * 2, layer_2_out= args.emb_dim, n_heads=args.attention_heads, dropout=args.dropout).to(device=device)
                optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

                train_start_ts = time.time()
                for epoch in range(100):
                    model.train()
                    optimizer.zero_grad()

                    output = model(adj_lists, adj_norm, features)

                    src_representations = torch.index_select(output, 0, train_edges_indices[0])
                    dst_representations = torch.index_select(output, 0, train_edges_indices[1])
                    reconstructed_values = torch.sigmoid(torch.sum(torch.mul(src_representations, dst_representations), 1))
                    # print(torch.mean(reconstructed_values))
                    criterion = nn.MSELoss()
                    loss = torch.sqrt(criterion(reconstructed_values, train_edges_values))
                    # print(f"EPOCH {epoch} LOSS {loss.item()}")
                    loss.backward()
                    optimizer.step()
                train_stop_ts = time.time()
                # run.log('training_time', (train_stop_ts-train_start_ts))

                model.eval()
                infer_start_ts = time.time()
                representations_tensor = model(adj_lists, adj_norm, features).detach()
                infer_stop_ts = time.time()
                # run.log('inference_time', (infer_stop_ts - infer_start_ts))

                evaluation = Evaluation(representations_tensor, train_edges_indices, train_edges_values, device)
                evaluation.train_evaluation_model()

                val_mae, val_rmse = evaluation.evaluate(val_edges_indices, val_edges_values)
                test_mae, test_rmse = evaluation.evaluate(test_edges_indices, test_edges_values)

                # run.log('val_run_mae', val_mae)
                # run.log('val_run_rmse', val_rmse)
                #
                # run.log('test_run_mae', test_mae)
                # run.log('test_run_rmse', test_rmse)

                val_mae_run.append(val_mae)
                val_rmse_run.append(val_rmse)
                test_mae_run.append(test_mae)
                test_rmse_run.append(test_rmse)

                # print(f"{val_rmse}:{test_rmse}")

                print(f"Timestep {i} VAL_MAE {val_mae} VAL_RMSE {val_rmse} TEST_MAE {test_mae} TEST_RMSE {test_rmse}")

            else:
                print(f"Skipped experiment {exp} Timestep {i}/{len(minutes)} because of limited edges")
        exp_val_mae.append(val_mae_run)
        exp_val_rmse.append(val_rmse_run)
        exp_test_mae.append(test_mae_run)
        exp_test_rmse.append(test_rmse_run)

    average_val_mae_per_run = np.mean(exp_val_mae)
    std_val_mae_per_run = np.std(exp_val_mae)

    average_val_rmse_per_run = np.mean(exp_val_rmse)
    std_val_rmse_per_run = np.std(exp_val_rmse)

    average_test_mae_per_run = np.mean(exp_test_mae)
    std_test_mae_per_run = np.std(exp_test_mae)

    average_test_rmse_per_run = np.mean(exp_test_mae)
    std_test_rmse_per_run = np.std(exp_test_mae)

    run.log('average_val_mae', average_val_mae_per_run)
    run.log('std_val_mae', std_val_mae_per_run)

    run.log('average_val_rmse', average_val_rmse_per_run)
    run.log('std_val_rmse', std_val_rmse_per_run)

    run.log('average_test_mae', average_test_mae_per_run)
    run.log('std_test_mae', std_test_mae_per_run)

    run.log('average_test_rmse', average_test_rmse_per_run)
    run.log('std_test_rmse', std_test_rmse_per_run)

def count_parameters(model):
    return np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])

def evolvegcn(args):
    w = args.window
    dataset = Compose(args.data_path, args.dataset_file, device, model="EVOLVEGCN")
    minutes = dataset.minutes
    print(f"Number of Nodes {dataset.num_nodes}")

    # exp = args.experiment_id
    exp_results = []

    for i in range(10):
    # for i in range(len(minutes) - 1):
        # print(f"Experiment {exp} Timestep {i}/{len(minutes)}")z
        start_graph = i - w if (i - w) > 0 else 0
        end_graph = i
        train_data, val_data, test_data = dataset.split_data(ns=args.ns, start_minute=start_graph, end_minute=end_graph)

        adj_norms = [data['adj_norm'] for data in train_data]
        features = [data['features'] for data in train_data]
        train_edges_indices = train_data[-1]['indices']
        train_edges_values = train_data[-1]['values']

        val_edges_indices = val_data['indices']
        val_edges_values = val_data['values']

        test_edges_indices = test_data['indices']
        test_edges_values = test_data['values']

        if train_edges_indices[0].shape[0] > 20 and val_edges_indices[0].shape[0] > 20 and test_edges_indices[0].shape[
            0]:
            model = EvolveGCNModel(layer_1_in=dataset.num_nodes, layer_1_out=args.emb_dim * 2, layer_2_out= args.emb_dim).to(
                device=device)
            parameters = count_parameters(model)
            print(f"Number of parameters {parameters}")
            if i == 0:
                print(f"Model to train {model}")
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

            train_start_ts = time.time()
            for epoch in range(100):
                model.train()
                optimizer.zero_grad()

                output = model(adj_norms, features)

                src_representations = torch.index_select(output, 0, train_edges_indices[0])
                dst_representations = torch.index_select(output, 0, train_edges_indices[1])
                reconstructed_values = torch.sigmoid(torch.sum(torch.mul(src_representations, dst_representations), 1))
                # print(torch.mean(reconstructed_values))

                criterion = nn.MSELoss()
                loss = torch.sqrt(criterion(reconstructed_values, train_edges_values))
                loss.backward()
                optimizer.step()
            train_stop_ts = time.time()

            model.eval()
            infer_start_ts = time.time()
            representations_tensor = model(adj_norms, features).detach()
            infer_stop_ts = time.time()
            del model

            evaluation = Evaluation(representations_tensor, train_edges_indices, train_edges_values, device)
            evaluation.train_evaluation_model()


            val_mae, val_rmse = evaluation.evaluate(val_edges_indices, val_edges_values)
            test_mae, test_rmse = evaluation.evaluate(test_edges_indices, test_edges_values)
            result = {'val_rmse': val_rmse,
                      'val_mae': val_mae,
                      'test_rmse': test_rmse,
                      'test_mae': test_mae,
                      'train_time': (train_stop_ts - train_start_ts), 'infer_time': (infer_stop_ts - infer_start_ts)}
            print(f"Timestep {i} VAL_MAE {val_mae} VAL_RMSE {val_rmse} TEST_MAE {test_mae} TEST_RMSE {test_rmse}")
            exp_results.append(result)

            del train_data, \
                val_data, \
                test_data, \
                adj_norms, \
                features, \
                train_edges_indices, \
                train_edges_values, \
                val_edges_indices, \
                val_edges_values, \
                test_edges_indices, \
                test_edges_values, \
                evaluation


            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        else:
            print(f"Skipped experiment Timestep {i}/{len(minutes)} because of limited edges")



def execute(args):
    if args.model == 1:
        vstreadrls(args)
    elif args.model == 2:
        evolvegcn(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='Dataset path')
    parser.add_argument('--dataset_file', type=str, help="Dataset file")
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for ADAM')
    parser.add_argument('--attention_heads', type=int, default=1, help="Number of attention heads")
    parser.add_argument('--model', type=int, default=1, help="Model employed [VStreamDRLS:1, EvolveGCN:2]")
    parser.add_argument('--emb_dim', type=int, default=16, help="Dimension of Node embeddings")
    parser.add_argument('--ns', type=int, default=1, help="Negative samples")
    parser.add_argument('--dropout', type=float, default=0., help="Dropout")
    parser.add_argument('--num_runs', type=int, default=5, help="Number of runs for this set of hyperparameters")
    parser.add_argument('--window', type=int, default=1, help="Window for dynamic/stream tasks")
    args = parser.parse_args()

    sys.exit(execute(args))