import os
import argparse
import torch

from dataset import shapes_in, shapes_out


def args_parser():
    parser = argparse.ArgumentParser()

    # dataset and partition

    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'cifar10', 'cifar10c', 'spambase', 'agnews'],
                        help='dataset name')

    parser.add_argument('--severity', type=int, default=3,
                        help='corruption severity when dataset is cifar10c')

    parser.add_argument('--num_honest', type=int, default=100,
                        help='number of honest (benign) clients')

    parser.add_argument('--partition', type=str, default='step_2_inf',
                        help='how to partition dataset to clients, in format ${method}_${parameters}')

    parser.add_argument('--partition_seed', type=int, default=0,
                        help='pre-defined data partition for each client')

    parser.add_argument('--num_server_data_per_class', type=int, default=10,
                        help='Number of data per class stored in the server')

    parser.add_argument('--biased_server_sample', action='store_true', default=False,
                        help='whether server sample is biased')

    parser.add_argument('--server_data_noise', type=str, default='none',
                        help='type of noise added to server')

    # model training

    parser.add_argument('--model', type=str, default='2nn',
                        help='federated learning model')

    parser.add_argument('--loss', type=str, default='ce',
                        choices=['ce', 'bce'],
                        help='loss function')

    parser.add_argument('--metric', type=str, default='acc',
                        choices=['acc', 'bacc'],
                        help='metric function')

    parser.add_argument('--algorithm', type=str, default='fedavg',
                        help='the federated learning algorithm')

    # Server config

    parser.add_argument('--gm_rounds', type=int, default=100,
                        help='number of global communication rounds')

    parser.add_argument('--part_rate', type=float, default=1.0,
                        help='client participation rate in each communication rounds')

    # Client config

    parser.add_argument('--lm_opt', type=str, default='sgd',
                        help='local model optimizer')

    parser.add_argument('--lm_lr', type=float, default=0.1,
                        help='learning rate of the local model optimizer')

    parser.add_argument('--lm_epochs', type=int, default=1,
                        help='number of local training epochs, each epoch iterates the local dataset once')

    parser.add_argument('--batch_size', type=int, default=10000,
                        help='batch size')

    # LR Decay

    parser.add_argument('--decay_rate', type=float, default=1.0,
                        help='learning rate decay rate')

    parser.add_argument('--decay_start_round', type=int, default=0,
                        help='learning rate decay start at round ?')

    parser.add_argument('--decay_per_round', type=int, default=100,
                        help='learning rate decay per ? rounds')

    # fedprox

    parser.add_argument('--fedprox_mu', type=float, default=0.01,
                        help='learning rate decay rate')

    # ======== ======== Attacks ======== ========

    parser.add_argument('--attack', type=str, default='gaussian',
                        # choices=['gaussian', 'signflip', 'little', 'mimic'],
                        help='the algorithm for attacker')

    parser.add_argument('--num_byz', type=int, default=0,
                        help='number of malicious clients')

    # Hyperparameters

    # gaussian
    parser.add_argument('--gaussian_std', type=float, default=200,
                        help='scale of the gaussian attack')

    parser.add_argument('--gaussian_collude', action='store_true', default=False,
                        help='whether use colluded gaussain update')

    # signflip
    parser.add_argument('--signflip_factor', type=float, default=-10,
                        help='scale of the signflip attack')

    # min-max and min-sum
    parser.add_argument('--gamma_init', type=float, default=10,
                        help='scale of the signflip attack')

    parser.add_argument('--min_tau', type=float, default=0.00001,
                        help='scale of the signflip attack')

    # ======== ======== Defenses ======== ========

    parser.add_argument('--defense', type=str, default='average',
                        help='the aggregator used to defend attacks')

    parser.add_argument('--num_byz_resist', type=int, default=16,
                        help='number of byzantine clients that the server can resist')

    parser.add_argument('--server_gradient_weight', type=float, default=0.0,
                        help='mixture with server gradient')

    # Hyperparameters

    # Bucket
    parser.add_argument('--bucket_s', type=int, default=2,
                        help='bucket size for bucketing')

    # RAGE
    parser.add_argument('--rage_max_iter', type=int, default=20,
                        help='max iteration in stage 1')

    # Zeno
    parser.add_argument('--zeno_rho', type=float, default=0.0005,
                        help='zeno balancing update norm and loss change')

    # ByGARS
    parser.add_argument('--bygars_k', type=int, default=3,
                        help='number of iterations for bygars')

    parser.add_argument('--bygars_alpha', type=float, default=0.05,
                        help='learning rate for bygars')

    # BOBA
    parser.add_argument('--boba_max_iter', type=int, default=20,
                        help='max iteration in stage 1')

    parser.add_argument('--boba_pmin', type=float, default=-0.5,
                        help='reject a gradient if its minimum label ratio is smaller than pmin')

    # to control randomness
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed to use')

    # training
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='whether use cuda to train ')

    parser.add_argument('--num_workers', type=int, default=0,
                        help='num_workers of dataloader')

    # directories
    parser.add_argument('--data_dir', type=str, default='~/data',
                        help='where the data is stored')

    parser.add_argument('--partition_dir', type=str, default='~/data/boba/partition',
                        help='where the data partition is stored')

    parser.add_argument('--history_path', type=str, default='none')

    # for debug
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='whether to print a lot')

    parser.add_argument('--visualize', action='store_true', default=False,
                        help='whether to visualize ')

    args = parser.parse_args()

    # client
    args.num_clients = args.num_honest + args.num_byz

    args.cohort_size = max(1, round(args.num_honest * args.part_rate))
    args.num_clients_partial = args.cohort_size + args.num_byz

    # in and out-dimension of model
    args.shape_in = shapes_in[args.dataset]
    args.shape_out = shapes_out[args.dataset]
    args.num_labels = max(2, args.shape_out)

    args.data_dir = os.path.expanduser(args.data_dir)
    args.partition_dir = os.path.expanduser(args.partition_dir)

    # the path of partition config
    if args.partition_seed is None:
        args.partition_seed = args.seed

    corruption_filename = 'client_%d_partition_%s_corruption_severity_%d_seed_%d.pkl' % (
        args.num_honest, args.partition, args.severity, args.partition_seed)
    args.corruption_path = os.path.join(args.data_dir, 'flrobust', args.dataset, corruption_filename)



    partition_filename = 'client_%d_partition_%s_server_%d_seed_%d.pkl' % (
    args.num_honest, args.partition, args.num_server_data_per_class, args.partition_seed)
    args.partition_path = os.path.join(args.partition_dir, args.dataset, partition_filename)

    args.device = torch.device('cuda') if torch.cuda.is_available() and args.cuda else torch.device('cpu')

    return args
