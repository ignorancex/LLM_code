'''
 every node have different observations
        train observation length [ob_min, ob_max]
'''

from synthetic_sim import ChargedParticlesSim, SpringSim, VCell
import time
import numpy as np
import argparse
import os
import pickle
from distutils.util import strtobool
def makedirs(dirname):
	if not os.path.exists(dirname):
		os.makedirs(dirname)

parser = argparse.ArgumentParser()
parser.add_argument('--simulation', type=str, default='VCell',
                    help='What simulation to generate.nturgbd charged springs') #
parser.add_argument('--num-train', type=int, default=2,#20000,
                    help='Number of training simulations to generate. Does not applied to VCell')
parser.add_argument('--num-test', type=int, default=2,#5000,
                    help='Number of test simulations to generate. Does not applied to VCell')
parser.add_argument('--ode', type=int, default=6000,
                    help='Length of trajectory.')
parser.add_argument('--num-test-box', type=int, default=1,
                    help='Length of test set trajectory.')
parser.add_argument('--num-test-extra', type=int, default=40,
                    help='Length of test set trajectory.')
parser.add_argument('--sample-freq', type=int, default=100)
parser.add_argument('--sample-freq-vcell', type=int, default=1)
parser.add_argument('--ob_max', type=int, default=52,
                    help='Length of test set trajectory.')
parser.add_argument('--ob_min', type=int, default=40,
                    help='Length of test set trajectory.')
parser.add_argument('--n-balls', type=int, default=10,
                    help='Number of balls in the simulation.')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed.')
parser.add_argument('--interaction_strength', type=float, default=.1, help='interaction_strength for the spring connected paticle system')
parser.add_argument('--interaction_strength_charged', type=float, default=.1, help='interaction_strength for the spring connected paticle system')
parser.add_argument('--interaction_strength_spring', type=float, default=.1, help='interaction_strength for the spring connected paticle system')
parser.add_argument('--NRW_sigma', type=float, default=0.01, help='variance for NRW')
parser.add_argument('--NRW_graph_id', type=str, default='00', help='index of graph structure for NRW')
parser.add_argument('--NRW_fix_graph', type=strtobool, default=1, help='whether fix graph for each task')
#parser.add_argument('--vcell_n_vars', type=int, default=36, help='number of variables kept in vcell simulation data')
parser.add_argument('--vcell_selected_vars_path', type=str, default='/store/MS-GODE/data/Ran_selected_vars_01.txt', help='number of variables kept in vcell simulation data')
parser.add_argument('--vcell_path', type=str, default='/store/MS-GODE/data', help='directory of vcell simulation data')
parser.add_argument('--vcell_model', type=str, default='Ran', help='directory of vcell simulation data',choices=['Ran','egfr'])
parser.add_argument('--vcell_config', type=str, default='00', help='directory of vcell simulation data')
parser.add_argument('--vcell_tr_va_te_ratios', type=str, default='0.6_0.2_0.2', help='directory of vcell simulation data')
parser.add_argument('--vcell_n_hops', type=int, default=1, help='number of hops used in adj')
parser.add_argument('--box_size', type=float, default=1., help='box size for the spring connected paticles')
parser.add_argument('--actions_ntu', default=[0,10])
parser.add_argument('--motion_fully_connect', type=strtobool, default=False)
parser.add_argument('--overwrite_existing', type=strtobool, default=False)
parser.add_argument('--sampling', type=strtobool, default=False, help='whether to sample data')

args = parser.parse_args()
if args.simulation in ['nturgbd']:
    data_store_path = f'/store/MS-GODE/data/simulated/motion'
elif args.simulation in ['VCell']:
    data_store_path = f'/store/MS-GODE/data/simulated/VCell'
elif args.simulation in ['springs','charged','chargedspring']:
    data_store_path = f'/store/MS-GODE/data/simulated/train{args.num_train}_test{args.num_test}'
else:
    raise ValueError('undefined data type')
makedirs(data_store_path)

if args.simulation == 'springs':
    sim = SpringSim(noise_var=0.0, n_balls=args.n_balls, interaction_strength=args.interaction_strength, box_size=args.box_size)
    suffix = f"_springs{args.n_balls}_{args.box_size}_{args.interaction_strength}"
elif args.simulation == 'charged':
    sim = ChargedParticlesSim(noise_var=0.0, n_balls=args.n_balls, interaction_strength=args.interaction_strength, box_size=args.box_size)
    suffix = f"_charged{args.n_balls}_{args.box_size}_{args.interaction_strength}"
elif args.simulation == 'VCell':
    sim = VCell()
    suffix = f"VCell{args.vcell_model}_config{args.vcell_config}"

else:
    raise ValueError('Simulation {} not implemented'.format(args.simulation))

#suffix += str(args.n_balls)
np.random.seed(args.seed)

print(suffix)


def generate_dataset(args,num_sims,isTrain = True):
    loc_all,vel_all,edges,timestamps,full_trajs = list(),list(),list(),list(),list()
    for i in range(num_sims):
        t = time.time()
        #graph generation
        static_graph,diag_mask = sim.generate_static_graph()
        edges.append(static_graph)  # [5,5] interaction strength is not encoded in edges, spring type is also binary

        loc, vel, T_samples, full_traj = sim.sample_trajectory_static_graph_irregular_difflength_each(args, edges=static_graph,diag_mask = diag_mask,
                                                                                               isTrain=isTrain)
        #print(123)
        if i % 1000 == 0:
            print("Iter: {}, Simulation time: {}".format(i, time.time() - t))
        loc_all.append(loc)  # [49,2,5]
        vel_all.append(vel)  # [49,2,5]
        timestamps.append(T_samples)  # [99]
        full_trajs.append(full_traj)

    loc_all = np.asarray(loc_all)  # [5000,5 list(timestamps,2)]
    vel_all = np.asarray(vel_all)
    edges = np.stack(edges)
    timestamps = np.asarray(timestamps)

    return loc_all, vel_all, edges, timestamps, full_trajs



def generate_dataset_VCell(args,mode = 'train'):
    traj_generate_func = sim.sample_trajectory_static_graph_irregular_difflength_each if args.sampling else sim.sample_trajectory
    data_path = f'{args.vcell_path}/VCell_{args.vcell_model}_config{args.vcell_config}'
    vcell_files = os.listdir(data_path)
    for c in vcell_files:
        if 'Network' in c:
            path_adj = f'{data_path}/{c}'
        if '.dat' in c or 'config' in c:
            vcell_files.remove(c)
    #vcell_files.sort()
    dataset_size = len(vcell_files)
    dataset_splits = [round(dataset_size*float(args.vcell_tr_va_te_ratios.split('_')[0])),round(dataset_size*float(args.vcell_tr_va_te_ratios.split('_')[0])+dataset_size*float(args.vcell_tr_va_te_ratios.split('_')[1]))]
    #path_adj = f'{data_path}/{vcell_files[-1]}' # after sort, the last one is the adj file
    var_name_path = args.vcell_selected_vars_path #f'{data_path}/{vcell_files[-1]}' # get the names from an arbitrary traj
    sim.get_var_names(var_name_path)
    loc_all,edges,timestamps = list(),list(),list()
    static_graph = sim.generate_static_graph(path_adj, args.vcell_n_hops)
    edges.append(static_graph)  # [5,5]

    if mode == 'train':
        start_id, end_id = 0, dataset_splits[1]
    elif mode == 'val':
        start_id, end_id = dataset_splits[1], dataset_splits[2]
    elif mode == 'test':
        start_id, end_id = dataset_splits[2], dataset_size
    loc, T_samples = traj_generate_func(args, f'{data_path}/{vcell_files[start_id]}')
    loc_all.append(loc)  # [49,2,5]
    timestamps.append(T_samples)  # [99]

    for i,c in enumerate(vcell_files[start_id+1:end_id]):
        t = time.time()
        #graph generation
        #static_graph,diag_mask = sim.generate_static_graph()
        edges.append(static_graph)  # [5,5]
        loc, T_samples = traj_generate_func(args,f'{data_path}/{c}')
        if i % 1000 == 0:
            print("Iter: {}, Simulation time: {}".format(i, time.time() - t))
        loc_all.append(loc)  # [49,2,5]
        timestamps.append(T_samples)  # [99]

    loc_all = np.asarray(loc_all)  # [5000,5 list(timestamps,2)]
    edges = np.stack(edges)
    timestamps = np.asarray(timestamps)

    return loc_all, edges, timestamps








if os.path.isfile(f'{data_store_path}/loc_test' + suffix + f'sampling_{args.sampling}' + '.npy') and not args.overwrite_existing:
    # avoid overitting existing data
    print('the following data exists \n', f'{data_store_path}/loc_test' + suffix + f'sampling_{args.sampling}' + '.npy')
else:
    data_gen_func_dict = {'springs':generate_dataset,'charged':generate_dataset,'chargedspring':generate_dataset_chargedspring,"nturgbd":generate_dataset_ntu, 'NRW': generate_dataset_NRW, 'VCell': generate_dataset_VCell}
    train_file_names = {'springs':['loc_train','vel_train','edges_train','times_train','full_traj_tr'],'charged':['loc_train','vel_train','edges_train','times_train','full_traj_tr'],'chargedspring':['loc_train','vel_train','edges_charged_train','edges_spring_train','times_train'],'NRW':['loc_train','edges_train','times_train','full_traj_tr'],'VCell':['loc_train','edges_train','times_train']}
    test_file_names = {'springs':['loc_test','vel_test','edges_test','times_test','full_traj_te'],'charged':['loc_test','vel_test','edges_test','times_test','full_traj_te'],'chargedspring':['loc_test','vel_test','edges_charged_test','edges_spring_test','times_test'],'NRW':['loc_test','edges_test','times_test','full_traj_te'],'VCell':['loc_test','edges_test','times_test']}

    data_gen_func = data_gen_func_dict[args.simulation]
    if args.simulation in ["springs","charged","chargedspring"]:
        print(f"{args.simulation}, Generating {args.num_test} test simulations")
        results_test = data_gen_func(args, args.num_test, isTrain=False)
        print(f"{args.simulation},Generating {args.num_train} training simulations")
        results_train = data_gen_func(args, args.num_train, isTrain=True)

        for name,result in zip(test_file_names[args.simulation],results_test):
            np.save(f'{data_store_path}/{name}' + suffix + '.npy', result)
        for name,result in zip(train_file_names[args.simulation],results_train):
            np.save(f'{data_store_path}/{name}' + suffix + '.npy', result)

    elif args.simulation == "VCell":
        print(f"{args.simulation}, Generating test simulations")
        results_test = data_gen_func(args,mode='test')
        print(f"{args.simulation},Generating training simulations")
        results_train = data_gen_func(args,mode='train')

        for name, result in zip(test_file_names[args.simulation], results_test):
            np.save(f'{data_store_path}/{name}' + suffix + f'_Vars{args.vcell_selected_vars_path.split("/")[-1].split("_")[-1].replace(".txt","")}_' + f'sampling_{args.sampling}_' + f'{args.vcell_n_hops}hop' + '.npy', result)
        for name, result in zip(train_file_names[args.simulation], results_train):
            np.save(f'{data_store_path}/{name}' + suffix + f'_Vars{args.vcell_selected_vars_path.split("/")[-1].split("_")[-1].replace(".txt","")}_' + f'sampling_{args.sampling}_' + f'{args.vcell_n_hops}hop' + '.npy', result)
