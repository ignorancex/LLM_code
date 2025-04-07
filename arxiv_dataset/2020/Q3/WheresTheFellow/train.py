import pickle
import sys

import dgl
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
from torch_geometric.data import Data
import torch.nn.functional as F
import signal

from graph_generator import CalibrationDataset, HumanGraph
from nets.gat import GAT
from nets.rgcnDGL import RGCN


if torch.cuda.is_available() is True:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def activation_functions(activation_tuple_src):
    ret = []
    for x in activation_tuple_src:
        if x == 'relu':
            ret.append(F.relu)
        elif x == 'elu':
            ret.append(F.elu)
        elif x == 'tanh':
            ret.append(torch.tanh)
        elif x == 'leaky_relu':
            ret.append(F.leaky_relu)
        else:
            print('Unknown activation function {}.'.format(x))
            sys.exit(-1)
    return tuple(ret)

def describe_model(model):
    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())


def collate(sample):
    graphs, feats, labels = map(list, zip(*sample))
    graph = dgl.batch(graphs)
    feats = torch.from_numpy(np.concatenate(feats))
    labels = torch.from_numpy(np.concatenate(labels))
    return graph, feats, labels


def evaluate(feats, model, subgraph, labels, loss_fcn, fw, net_class):
    with torch.no_grad():
        model.eval()
        if fw == 'dgl' :
            model.g = subgraph
            for layer in model.layers:
                layer.g = subgraph
            if net_class in [RGCN]:
                output = model(feats.float(), subgraph.edata['rel_type'].to(device).squeeze())
            else:
                output = model(feats.float())
        oput = output[getMaskForBatch(subgraph)]
        loss_data = loss_fcn(oput, labels.float())
        predict = output[getMaskForBatch(subgraph)].data.cpu().numpy()
        score = mean_squared_error(labels.data.cpu().numpy(), predict)
        return score, loss_data.item()


def getMaskForBatch(subgraph):
    first_node_index_in_the_next_graph = 0
    indexes = []
    for g in dgl.unbatch(subgraph):
        indexes.append(first_node_index_in_the_next_graph)
        first_node_index_in_the_next_graph += g.number_of_nodes()
    return indexes


def num_of_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])

stop_training = False
ctrl_c_counter = 0
def signal_handler(sig, frame):
    global ctrl_c_counter
    ctrl_c_counter += 1
    if ctrl_c_counter >= 6:
        sys.exit(-1)
    elif ctrl_c_counter >= 3:
        global stop_training
        stop_training = True
    print('\nIf you press Ctr+c 3  times we will stop    _SAVING_   the training information ({} times)'.format(ctrl_c_counter))
    print(  'If you press Ctr+c 6+ times we will stop  _NOT SAVING_ the training information ({} times)'.format(ctrl_c_counter))

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# MAIN
def main(training_file, dev_file, test_file, epochs=None, patience=None, heads=None, num_layers=None,
         num_hidden=None, residual=None, in_drop=None, attn_drop=None, lr=None, weight_decay=None,
         alpha=None, batch_size=None, graph_type=None, net=None, activations=('elu', 'tanh'), fw='dgl'):

    if net.lower() == 'GAT'.lower():
        net_class = GAT
    elif net.lower() == 'RGCN'.lower():
        net_class = RGCN

    print('DEVICE', device)

    # define loss function
    loss_fcn = torch.nn.MSELoss()

    print('=========================')
    print('HEADS',        heads)
    print('LAYERS',       num_layers)
    print('HIDDEN',       num_hidden)
    print('RESIDUAL',     residual)
    print('inDROP',       in_drop)
    print('atDROP',       attn_drop)
    print('LR',           lr)
    print('DECAY',        weight_decay)
    print('ALPHA',        alpha)
    print('BATCH',        batch_size)
    print('GRAPH_ALT',    graph_type)
    print('ARCHITECTURE', net)
    print('=========================')

    # create the dataset
    print('Loading training set...')
    train_dataset = CalibrationDataset(training_file, mode='train', alt=graph_type)
    print('Loading dev set...')
    valid_dataset = CalibrationDataset(dev_file, mode='valid', alt=graph_type)
    print('Loading test set...')
    test_dataset  = CalibrationDataset(test_file, mode='test', alt=graph_type)
    print('Done loading files')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, collate_fn=collate)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate)

    num_rels = len(HumanGraph.get_rels())
    cur_step = 0
    best_loss = -1
    n_classes = train_dataset.labels.shape[1]
    print('Number of classes:  {}'.format(n_classes))
    num_feats = train_dataset.features.shape[1]
    print('Number of features: {}'.format(num_feats))
    g = train_dataset.graph
    # define the model

    print('LAST', fw, net)



    if fw == 'dgl':
        if net_class in [GAT]:
            model = net_class(g, num_layers, num_feats, num_hidden, n_classes, heads, activation_functions(activations),
                              in_drop, attn_drop, alpha, residual)
        elif net_class in [RGCN]:
            model = RGCN(g, gnn_layers=num_layers, in_dim=num_feats, hidden_dimensions=num_hidden, num_rels=num_rels,
                         activations=activation_functions(activations), feat_drop=in_drop)
            print(f'CREATING RGCN(GRAPH, gnn_layers:{num_layers}, num_feats:{num_feats}, num_hidden:{num_hidden}, num_rels:{num_rels}, non-linearity:{activation_functions(activations)}, drop:{in_drop})')
        else:
            print('Unhandled', net)
            sys.exit(1)
    #Describe the model
    #describe_model(model)

    # define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # for name, param in model.named_parameters():
        # if param.requires_grad:
        # print(name, param.data.shape)
    model = model.to(device)

    for epoch in range(epochs):
        if stop_training:
            print("Stopping training. Please wait.")
            break
        model.train()
        loss_list = []
        for batch, data in enumerate(train_dataloader):
            subgraph, feats, labels = data
            subgraph.set_n_initializer(dgl.init.zero_initializer)
            subgraph.set_e_initializer(dgl.init.zero_initializer)
            feats = feats.to(device)
            labels = labels.to(device)
            if fw == 'dgl':
                model.g = subgraph
                for layer in model.layers:
                    layer.g = subgraph
                if net_class in [RGCN]:
                    logits = model(feats.float(), subgraph.edata['rel_type'].squeeze().to(device))
                else:
                    logits = model(feats.float())
            loss = loss_fcn(logits[getMaskForBatch(subgraph)], labels.float())
            optimizer.zero_grad()
            a = list(model.parameters())[0].clone()
            loss.backward()
            optimizer.step()
            b = list(model.parameters())[0].clone()
            not_learning = torch.equal(a.data, b.data)
            if not_learning:
                print('Not learning')
                sys.exit(1)
            loss_list.append(loss.item())
        loss_data = np.array(loss_list).mean()
        print('Loss: {}'.format(loss_data))
        if epoch % 5 == 0:
            if epoch % 5 == 0:
                print("Epoch {:05d} | Loss: {:.4f} | Patience: {} | ".format(epoch, loss_data, cur_step), end='')
            score_list = []
            val_loss_list = []
            for batch, valid_data in enumerate(valid_dataloader):
                subgraph, feats, labels = valid_data
                subgraph.set_n_initializer(dgl.init.zero_initializer)
                subgraph.set_e_initializer(dgl.init.zero_initializer)
                feats = feats.to(device)
                labels = labels.to(device)
                score, val_loss = evaluate(feats.float(), model, subgraph, labels.float(), loss_fcn, fw, net_class)
                score_list.append(score)
                val_loss_list.append(val_loss)
            mean_score = np.array(score_list).mean()
            mean_val_loss = np.array(val_loss_list).mean()
            if epoch % 5 == 0:
                print("Score: {:.4f} MEAN: {:.4f} BEST: {:.4f}".format(mean_score, mean_val_loss, best_loss))
            # early stop
            if best_loss > mean_val_loss or best_loss < 0:
                print('Saving...')
                best_loss = mean_val_loss
                # Save the model
                torch.save(model.state_dict(), 'calibration_' + fw + '_' + net + '.tch')
                params = {'loss': best_loss,
                          'net': net,
                          'fw': fw,
                          'num_layers': num_layers,
                          'num_feats': num_feats,
                          'num_hidden': num_hidden,
                          'graph_type' : graph_type,
                          'n_classes': n_classes,
                          'heads': heads,
                          'F': F.relu,
                          'in_drop': in_drop,
                          'attn_drop': attn_drop,
                          'alpha': alpha,
                          'residual': residual,
                          'non-linearity': activations,
                          'num_rels': num_rels
                          }
                pickle.dump(params, open('calibration_' + fw + '_' + net + '_tmp.prms', 'wb'))
                cur_step = 0
            else:
                cur_step += 1
                if cur_step >= patience:
                    break
    model.eval()
    test_score_list = []
    for batch, test_data in enumerate(test_dataloader):
        subgraph, feats, labels = test_data
        subgraph.set_n_initializer(dgl.init.zero_initializer)
        subgraph.set_e_initializer(dgl.init.zero_initializer)
        feats = feats.to(device)
        labels = labels.to(device)
        test_score_list.append(evaluate(feats, model, subgraph, labels.float(), loss_fcn, fw, net_class)[1])
    print("MSE for the test set {}".format(np.array(test_score_list).mean()))
    params = {'loss': best_loss,
              'test_loss': np.array(test_score_list).mean(),
                'net': net,
                'fw': fw,
                'num_gnn_layers': num_layers,
                'num_feats': num_feats,
                'num_hidden': num_hidden,
                'graph_type' : graph_type,
                'n_classes': n_classes,
                'heads': heads,
                'F': F.relu,
                'in_drop': in_drop,
                'attn_drop': attn_drop,
                'alpha': alpha,
                'residual': residual,
                'non-linearity': activations,
                'num_rels': num_rels,
                'parameters': num_of_params(model)
                }
    pickle.dump(params, open('calibration_' + fw + '_' + net + '.prms', 'wb'))
    print('Number of parameters in the model {}.'.format(num_of_params(model)))

    return best_loss, np.array(test_score_list).mean()


if __name__ == '__main__':
    best_loss, test_loss = main('datasets/training_DS1.json',
                    'datasets/dev.json',
                    'datasets/test.json',
                     epochs=1000,
                     patience=15,
                     heads=[18, 17, 16],
                     num_layers=3,
                     num_hidden=[11, 8, 5],
                     residual=False,
                     in_drop=0.,
                     attn_drop=0.,
                     lr=0.0001,
                     weight_decay=0.00000001,
                     alpha=0.12,
                     batch_size=10,
                     graph_type='1',
                     net='gat',
                     activations=['relu', 'relu', 'tanh'],
                     fw='dgl')


