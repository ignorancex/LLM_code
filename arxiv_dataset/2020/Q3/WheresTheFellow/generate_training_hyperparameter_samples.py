"""
Generates or prints the task list.
"""
import sys

import pickle
from random import choice, randrange, uniform, seed
from math import floor

import matplotlib.pyplot as plt


seed()

def print_best_options(list_of_tasks):
    best_loss = -1
    best = None
    best_index = None
    pending, failed, done = 0, 0, 0
    for index in range(len(list_of_tasks)):
        # Get info
        selected = list_of_tasks[index]
        loss, fw, net = selected['test_loss'], selected['fw'], selected['gnn_network']
        # Handle: pending/failed/done
        if loss < 0.0:
            if selected['train_loss'] == 0.0:
                failed += 1
            else:
                pending += 1
        else:
            done += 1
            print('DONE', selected)
        # Handle overall best
        if (loss < best_loss and loss > 0.) or best_loss <= 0:
            best_loss = loss
            best = list_of_tasks[index]
            best_index = index
        # Handle best by type
        if (fw, net) in best_by_option:
            if loss > 0:
                if loss < best_by_option[(fw,net)]['test_loss']:
                    best_by_option[(fw,net)] = selected
        elif loss>0:
            best_by_option[(fw,net)] = selected

    print('BEST BY OPTION')
    for k in best_by_option:
        print(k)
        print(best_by_option[k])
        print('')
    print('Pending:', pending)
    print('Failed:', failed)
    print('Done:', done)
    print('Best result:')
    for k in best.keys():
        if '_scores' in k:
            continue
        if k == 'elapsed':
            v = best[k]
            hours = int(floor(v/3600))
            v = v - hours*3600
            minutes = int(floor(v/60))
            print(f'elapsed: {hours}h {minutes}m')
        else:
            print(k, best[k])
    print('Best index:', best_index)
    
    try:
        plt.plot(range(1,len(list_of_tasks[best_index]['train_scores'])),list_of_tasks[best_index]['train_scores'][1:],'-',label='train')
        plt.plot([5.*x for x in range(1,len(list_of_tasks[best_index]['dev_scores']))],list_of_tasks[best_index]['dev_scores'][1:],'-',label='dev')
        plt.legend()
        plt.show()
    except KeyError:
        pass


def get_random_hyperparameters(identifier):
    fw_net_map = {0: ('dgl', 'rgcn'), 1: ('dgl', 'gat')}

    fw, net = fw_net_map[randrange(len(fw_net_map))]
    if net in ['rgcn']:
        graph_type = choice(['1'])
    else:
        graph_type = '1'


    gnn_layers = randrange(start=2, stop=6)
    # print('gnn_layers: {}'.format(gnn_layers))
    last_gnn_units = 5 # x, y, z, cos/sin
    # print('last_gnn_units: {}'.format(last_gnn_units))
    first_gnn_units = randrange(start=10, stop=70)

    # print('first_gnn_units: {}'.format(first_gnn_units))
    gnn_units = [first_gnn_units]

    if 'gat' in net:
        first_gnn_heads = randrange(start=3, stop=20)
        # print('first_gnn_heads: {}'.format(first_gnn_heads))
        last_gnn_heads = randrange(start=3, stop=20)
        # print('last_gnn_heads: {}'.format(first_gnn_heads))
    else:
        first_gnn_heads = 1
        last_gnn_heads = 1
    gnn_heads = [first_gnn_heads]

    diff_units = float(last_gnn_units-first_gnn_units)/float(gnn_layers-1)
    diff_heads = float(last_gnn_heads-first_gnn_heads)/float(gnn_layers-1)
    for l in range(1, gnn_layers-1):
        hidden_units = int(first_gnn_units + diff_units*l)
        hidden_heads = int(first_gnn_heads + diff_heads*l)
        gnn_units.append(hidden_units)
        gnn_heads.append(hidden_heads)
    gnn_units.append(last_gnn_units)
    gnn_heads.append(last_gnn_heads)

    hyperparameters = {
        'identifier': identifier,
        'fw': fw,
        'gnn_network': net,
        'graph_type': graph_type,
        'epochs': 1000, # 1000
        'patience': 6,
        'outputs': 5,
        'batch_size': randrange(start=5, stop=15), # start=100, stop=1500
        'num_gnn_units': gnn_units,
        'num_gnn_heads': gnn_heads,
        'lr': choice([0.0001, 0.00025, 0.0005]),
        'num_gnn_layers': gnn_layers,
        'weight_decay': choice([0., 0.00000001, 0.00000000001]),
        'non-linearity': [choice(['relu', 'elu'])]*(gnn_layers-1)+['tanh'],
        'num_bases': choice([-1, -1, -1, -1, -1, 4, 8, 10, 16, 20]),
        'in_drop': choice([0.00001, 0.000001, 0.0, 0.0, 0.0, 0.0, 0.0]),
        'alpha': uniform(0.1, 0.3),
        'attn_drop': choice([0.00001, 0.000001, 0.00001, 0.0, 0.0, 0.0, 0.0, 0,0, 0.0, 0.0, 0.0, 0.0, 0.0, 0,0, 0.0, 0.0]),
        'train_loss': -1.,
        'dev_loss': -1.,
        'test_loss': -1.,
        'test_time': -1.
        }
    return hyperparameters

best_by_option = {}

if len(sys.argv) > 1:
    pathh = sys.argv[1]
else:
    pathh = 'LIST_OF_TASKS.pckl'
try:
    print_best_options(pickle.load(open(pathh, 'rb')))
except FileNotFoundError:
    generated_tasks = [get_random_hyperparameters(identifier) for identifier in range(10000)]
    pickle.dump(generated_tasks, open(pathh, 'wb'))







