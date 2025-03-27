import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import pickle as pkl
import os, sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from hive.algorithm.gnn.gat.model import GAT
from hive.data.transformation.single.graph.static.transforms import Compose
from hive.task.link_prediction.static.evaluate import Evaluation
from azureml.core import Run
run = Run.get_context()

def gat(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = Compose(args.data_path, args.dataset_file, device, relabel_nodes= args.node_relabel)

    train_adj, \
    train_edges_indices, \
    train_edges_values, \
    val_edges_indices, \
    val_edges_values, \
    test_edges_indices, \
    test_edges_values = dataset.split_data(ns=args.ns)
    features = dataset.get_features(node_features=args.features)
    print(f"Number of Nodes {dataset.num_nodes}")


    model = GAT(in_features=dataset.num_nodes, n_hidden=2 * args.emb_dim, n_out = args.emb_dim, dropout=args.dropout, alpha=0.2, n_heads = args.attention_heads).to(device=device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    start_ts = time.time()
    for epoch in range(100):
        start_epoch_ts = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features, train_adj)
        reconstruction = torch.sigmoid(torch.mm(output, torch.t(output)))

        reconstructed_values = reconstruction[train_edges_indices]
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(reconstructed_values, train_edges_values))
        run.log('loss', loss.item())
        print(f"Epoch {epoch} LOSS {loss}")
        loss.backward()
        optimizer.step()
        end_epoch_ts = time.time()
        run.log('epoch_time', (end_epoch_ts - start_epoch_ts))
    end_ts = time.time()
    run.log('training_time', (end_ts - start_ts))

    model.eval()
    representations_tensor = model(features, train_adj).detach()

    # pkl.dump(representations_tensor.to(device='cpu').numpy(), open('representations.pkl', 'wb'))

    evaluation = Evaluation(representations_tensor, train_edges_indices, train_edges_values, device)
    evaluation.train_evaluation_model()
    val_mae, val_rmse = evaluation.evaluate(val_edges_indices, val_edges_values)
    test_mae, test_rmse = evaluation.evaluate(test_edges_indices, test_edges_values)

    run.log('val_mae', val_mae)
    run.log('val_rmse', val_rmse)
    run.log('test_mae', test_mae)
    run.log('test_rmse', test_rmse)



def execute(args):
    if args.model == 1:
        gat(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='Dataset path')
    parser.add_argument('--dataset_file', type=str, help="Dataset file")
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for SGD')
    parser.add_argument('--attention_heads', type=int, default=1, help="Number of attention heads")
    parser.add_argument('--model', type=int, default=1, help="Model employed [GAT:1]")
    parser.add_argument('--emb_dim', type=int, default=256, help="Dimension of Node embeddings")
    parser.add_argument('--ns', type=int, default=1, help="Negative samples")
    parser.add_argument('--dropout', type=float, default=0.2, help="Dropout")
    parser.add_argument('--num_iter', type=int, default=5, help="Number of iterations")
    parser.add_argument('--features', type=bool, default=False, help="Include Node Features")
    parser.add_argument('--node_relabel', type=bool, default=False, help="Relabel Node IDs")
    args = parser.parse_args()

    sys.exit(execute(args))





