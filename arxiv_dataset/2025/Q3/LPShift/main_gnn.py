import torch
import torch_geometric.transforms as T
import numpy as np
import argparse
import os

from gnn_model import *
from utils import *
from torch.utils.data import DataLoader
from eval import evaluate_mrr
from synth_dataset import SynthDataset

ROOT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
log_print = get_logger('testrun', 'log', ROOT_DIR)

def get_metric_score(pos_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred, neg_train_pred):
    
    k_list = [1, 3, 10, 20, 50, 100]
    result = {}

    result_mrr_train = evaluate_mrr(pos_train_pred, neg_train_pred)
    result_mrr_val = evaluate_mrr(pos_val_pred, neg_val_pred )
    result_mrr_test = evaluate_mrr(pos_test_pred, neg_test_pred )
    
    result['MRR'] = (result_mrr_train['MRR'], result_mrr_val['MRR'], result_mrr_test['MRR'])
    for K in k_list:
        result[f'Hits@{K}'] = (result_mrr_train[f'mrr_hit{K}'], result_mrr_val[f'mrr_hit{K}'], result_mrr_test[f'mrr_hit{K}'])

    return result

def train(model, score_func, train_pos, data, optimizer, batch_size):
    model.train()
    score_func.train()

    total_loss = total_examples = 0
     
    x = data.x

    for perm in DataLoader(range(train_pos.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()
        num_nodes = x.size(0)
        adj = data.adj_t 

        h = model(x, adj)

        edge = train_pos[perm].t()

        pos_out = score_func(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        edge = torch.randint(0, num_nodes, edge.size(), dtype=torch.long,
                                device=h.device)
            
        neg_out = score_func(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(score_func.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples

@torch.no_grad()
def test_edge(score_func, input_data, h, batch_size,  negative_data=None):

    pos_preds = []
    neg_preds = []

    if negative_data is not None:
        
        for perm in DataLoader(range(input_data.size(0)),  batch_size):
            pos_edges = input_data[perm].t()
            neg_edges = torch.permute(negative_data[perm], (2, 0, 1))

            pos_scores = score_func(h[pos_edges[0]], h[pos_edges[1]]).cpu()
            neg_scores = score_func(h[neg_edges[0]], h[neg_edges[1]]).cpu()

            pos_preds += [pos_scores]
            neg_preds += [neg_scores]
        
        neg_preds = torch.cat(neg_preds, dim=0)
    else:
        neg_preds = None
        for perm  in DataLoader(range(input_data.size(0)), batch_size):
            edge = input_data[perm].t()
            pos_preds += [score_func(h[edge[0]], h[edge[1]]).cpu()]
            
    pos_preds = torch.cat(pos_preds, dim=0)

    return pos_preds, neg_preds

@torch.no_grad()
def test(model, score_func, data, evaluation_edges, batch_tup):
    model.eval()
    score_func.eval()

    train_bs, val_bs, test_bs = batch_tup
    pos_train_edge, pos_valid_edge, neg_valid_edge, pos_test_edge,  neg_test_edge, neg_train_edge = evaluation_edges

    x = data.x
    h = model(x, data.adj_t.to(x.device))
    x1 = h
    x2 = torch.tensor(1)
 
    pos_train_edge = pos_train_edge.to(x.device)
    pos_valid_edge = pos_valid_edge.to(x.device) 
    neg_valid_edge = neg_valid_edge.to(x.device)
    pos_test_edge = pos_test_edge.to(x.device) 
    neg_test_edge = neg_test_edge.to(x.device)
    neg_train_edge = neg_train_edge.to(x.device)

    pos_valid_pred, neg_valid_pred = test_edge(score_func, pos_valid_edge, h, val_bs, negative_data=neg_valid_edge)
    pos_test_pred, neg_test_pred = test_edge(score_func, pos_test_edge, h, test_bs, negative_data=neg_test_edge)
    if neg_train_edge.dim() == 2: neg_train_edge = neg_train_edge.unsqueeze(1)
    pos_train_pred, neg_train_pred = test_edge(score_func, pos_train_edge, h, train_bs, negative_data=neg_train_edge)

    pos_train_pred = torch.flatten(pos_train_pred)
    neg_train_pred = neg_train_pred.squeeze(-1)
   
    pos_valid_pred = torch.flatten(pos_valid_pred)
    pos_test_pred = torch.flatten(pos_test_pred)

    neg_valid_pred = neg_valid_pred.squeeze(-1)
    neg_test_pred = neg_test_pred.squeeze(-1)
    
    print("neg_train_pred size before predictions: ", neg_train_pred.size(), flush=True)
   
    print('train_pos train_neg valid_pos valid_neg test_pos test_neg', pos_train_pred.size(), neg_train_pred.size(), pos_valid_pred.size(), neg_valid_pred.size(), pos_test_pred.size(), neg_test_pred.size())
    
    result = get_metric_score(pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred, neg_train_pred)
    
    score_emb = [pos_valid_pred.cpu(),neg_valid_pred.cpu(), pos_test_pred.cpu(), neg_test_pred.cpu(), x1.cpu(), x2.cpu()]

    return result, score_emb

def main():
    parser = argparse.ArgumentParser(description='GNN training and testing pipeline for Flex')
    parser.add_argument('--data_name', type=str, default='ogbl-collab')
    parser.add_argument('--neg_mode', type=str, default='equal')
    parser.add_argument('--gnn_model', type=str, default='GCN')
    parser.add_argument('--score_model', type=str, default='mlp_score')

    ##gnn settings
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--num_layers_predictor', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--edge_drop', type=float, default=0.0)

    ### train settings
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--eval_steps', type=int, default=20)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--kill_cnt', dest='kill_cnt', default=100, type=int, help='early stopping')
    parser.add_argument('--output_dir', type=str, default='models')
    parser.add_argument('--input_dir', type=str, default=os.path.join(ROOT_DIR, "dataset"))
    parser.add_argument('--l2', type=float, default=0.0, help='L2 Regularization for Optimizer')
    parser.add_argument('--seed', type=int, default=1)
    
    ### log settings
    parser.add_argument('--save', action='store_true', default=False)
    parser.add_argument('--save_test', action='store_true', default=False)
    parser.add_argument('--use_saved_model', action='store_true', default=False)
    parser.add_argument('--log_steps', type=int, default=1)

    ### system settings
    parser.add_argument('--batch_size', type=int, default=65536)
    parser.add_argument('--val_batch_size', type=int, default=65536)
    parser.add_argument('--test_batch_size', type=int, default=65536)
    parser.add_argument('--device', type=int, default=0)

    args = parser.parse_args()
    print(args)

    init_seed(args.seed)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    loggers = {
        'Hits@1': Logger(args.runs),
        'Hits@3': Logger(args.runs),
        'Hits@10': Logger(args.runs),
        'Hits@20': Logger(args.runs),
        'Hits@50': Logger(args.runs),
        'Hits@100': Logger(args.runs),
        'MRR': Logger(args.runs)
    }
    eval_metric = 'MRR'

    for run in range(args.runs):

        print('#################################          ', run, '          #################################')
        if args.runs == 1:
            seed = args.seed
        else:
            seed = run + 1
        print('seed: ', seed)
        init_seed(seed)

        if 'cn' not in args.data_name.lower() and 'pa' not in args.data_name.lower() and 'sp' not in args.data_name.lower():
            raise AssertionError("LPShift and SynthDataset must contain 'CN', 'PA', 'SP' ")
        
        dataset_name = args.data_name + '_seed1'
        print("################################")
        print(f'Loading Dataset: {dataset_name}')
        print("################################")
        data = SynthDataset(dataset_name=dataset_name).get()
        if 'ppa' in dataset_name: data.x = data.x.float()
            
        split_edge = SynthDataset(dataset_name=dataset_name).get_edge_split()
        edge_index = data.edge_index

        while split_edge['train']['edge'].size(0) <= args.batch_size:
            args.batch_size = args.batch_size // 2
            if args.batch_size <= 0:
                raise Exception("Batch Size Reached 0 in Pos. Train Edges")
        
        while split_edge['valid']['edge'].size(0) <= args.val_batch_size:
            args.val_batch_size = args.val_batch_size // 2
            if args.val_batch_size <= 0:
                raise Exception("Batch Size Reached 0 in Pos. Val. Edges")
            
        while split_edge['test']['edge'].size(0) <= args.test_batch_size:
            args.test_batch_size = args.test_batch_size // 2
            if args.test_batch_size <= 0:
                raise Exception("Batch Size reached 0 in Pos. Testing Edges")

        batch_tup = (args.batch_size, args.val_batch_size, args.test_batch_size)

        input_channel = data.x.size(1)

        if hasattr(data, 'edge_weight'):
            if data.edge_weight != None:
                data.edge_weight = data.edge_weight.view(-1).to(torch.float)
                train_edge_weight = split_edge['train']['weight'].to(device)
                train_edge_weight = train_edge_weight.to(torch.float)
            else:
                train_edge_weight = None
        else:
            train_edge_weight = None

        data = T.ToSparseTensor()(data) 
        data.adj_t = data.adj_t.coalesce().bool().float() # Clamp edge_weights
        data.adj_t = data.adj_t.to_symmetric() # Enforce Symmetry
        data = data.to(device)
        model = eval(args.gnn_model)(input_channel, args.hidden_channels,
                        args.hidden_channels, args.num_layers, args.dropout, args.edge_drop).to(device)

        score_func = eval(args.score_model)(args.hidden_channels, args.hidden_channels,
                        1, args.num_layers_predictor, args.dropout).to(device)
        
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(score_func.parameters()),lr=args.lr, weight_decay=args.l2)
        
        pos_train_edge = split_edge['train']['edge']
        pos_valid_edge = split_edge['valid']['edge']
        pos_test_edge = split_edge['test']['edge']
            
        with open(f'dataset/{dataset_name}Dataset/heart_valid_samples.npy', "rb") as f:
            neg_valid_edge = np.load(f)
            neg_valid_edge = torch.from_numpy(neg_valid_edge)
        with open(f'dataset/{dataset_name}Dataset/heart_test_samples.npy', "rb") as f:
            neg_test_edge = np.load(f)
            neg_test_edge = torch.from_numpy(neg_test_edge)
    
        neg_train_edge = get_ogb_train_negs(split_edge, edge_index, data.num_nodes, 1, dataset_name)
        split_edge['train']['edge_neg'] = neg_train_edge

        evaluation_edges = [pos_train_edge, pos_valid_edge, neg_valid_edge, pos_test_edge,  neg_test_edge, neg_train_edge]
        print('train train_neg val val_neg test test_neg: ', pos_train_edge.size(), neg_train_edge.size(), pos_valid_edge.size(), neg_valid_edge.size(), pos_test_edge.size(), neg_test_edge.size(),  flush=True)
        
        save_path = args.output_dir +  f'/GCN_{args.data_name}_{seed}'

        model.reset_parameters()
        score_func.reset_parameters()

        if args.use_saved_model:
            model.load_state_dict(torch.load(save_path+'_model.pt'))
            score_func.load_state_dict(torch.load(save_path+'_predictor.pt'))

        best_valid, kill_cnt, best_test = 0, 0, 0

        for epoch in range(1, 1 + args.epochs):
            loss = train(model, score_func, pos_train_edge, data, optimizer, args.batch_size)
           
            if epoch % args.eval_steps == 0:
                results_rank, score_emb= test(model, score_func, data, evaluation_edges, batch_tup)

                for key, result in results_rank.items():
                    loggers[key].add_result(run, result)

                if epoch % args.log_steps == 0:
                    for key, result in results_rank.items():
                        train_hits, valid_hits, test_hits = result
                        
                        log_print.info(
                            f'Run: {run + 1:02d}, '
                            f'Epoch: {epoch:02d}, '
                            f'Loss: {loss:.4f}, '
                            f'Train: {100 * train_hits:.2f}%, '
                            f'Valid: {100 * valid_hits:.2f}%, '
                            f'Test: {100 * test_hits:.2f}%')

                r = torch.tensor(loggers[eval_metric].results[run])
                best_valid_current = round(r[:, 1].max().item(),4)
                best_test = round(r[r[:, 1].argmax(), 2].item(), 4)

                print(eval_metric)
                log_print.info(f'best valid: {100*best_valid_current:.2f}%, '
                                f'best test: {100*best_test:.2f}%')
                
                print('---')
                
                if best_valid_current > best_valid:
                    best_valid = best_valid_current
                    kill_cnt = 0
                    if args.save: 
                        torch.save(model.state_dict(), save_path+'_model.pt')
                        torch.save(score_func.state_dict(), save_path+'_predictor.pt')
                    if args.save_test:
                        torch.save(score_emb, save_path+'_scemb.pt')
                else:
                    kill_cnt += 1
                    if kill_cnt > args.kill_cnt: 
                        print("Early Stopping!!")
                        break
        
        for key in loggers.keys():
            if len(loggers[key].results[run]) > 0:
                print(key)
                loggers[key].print_statistics(run)
    
    result_all_run = {}
    for key in loggers.keys():
        if len(loggers[key].results[run]) > 0:
            print(key)
            _,  _, mean_list, var_list = loggers[key].print_statistics()
            result_all_run[key] = [mean_list, var_list]
    
    print(f"RESULTS FINISHED FOR GCN on {args.data_name}", flush=True)

if __name__ == "__main__":
    main()