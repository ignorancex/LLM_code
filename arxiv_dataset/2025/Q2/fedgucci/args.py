import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # Training-time neuron alignment
    parser.add_argument('--tna_hyperparam', type=float, default=0.3,
                        help="hyperparameter for the particular training-time neuron aligment hyperparameter")
    parser.add_argument('--mask_ratio', type=float, default=0.4,
                        help="for mask_activation, the percent of fixed neurons")
    parser.add_argument('--negative_ratio', type=float, default=0.8,
                        help="for mask_direction, ratio of negative direction (become smaller)")
    parser.add_argument('--gaussian_std', type=float, default=0.01,
                        help="for mask_gaussian, the gaussian std")
    parser.add_argument('--intrinsic_dim', type=float, default=1000,
                        help="for mask_intrinsic, the intrinsic_dim")

    # Data
    parser.add_argument('--num_cluster', type=int, default=4,
                        help="number of clusters, 2 or 4")
    parser.add_argument('--noniid_type', type=str, default='dirichlet',
                        help="the method of generating clusters, rota or swap or iid or dirichlet")
    parser.add_argument('--iid', type=int, default=0,  # select iid/non-iid #todo 0
                        help='set 1 for iid')
    parser.add_argument('--batchsize', type=int, default=128, # 128
                        help="batchsize")
    parser.add_argument('--validate_batchsize', type=int, default=128, # 128
                        help="batchsize")
    parser.add_argument('--dirichlet_alpha', type=float, default=0.5, # 128
                    help="dirichlet_alpha")
    parser.add_argument('--dirichlet_alpha2', type=float, default=False, # 128
                    help="dirichlet_alpha2")
    parser.add_argument('--data_ratio', type=float, default=None)

    # System
    parser.add_argument('--device', type=str, default='0',
                        help="device: {cuda, cpu}")
    parser.add_argument('--node_num', type=int, default=20, # 200
                        help="Number of nodes") # clusterSize: how much nodes in a cluster
    parser.add_argument('--T', type=int, default=200,  # 100 
                        help="Number of communication rounds")
    parser.add_argument('--E', type=int, default=1, # 3
                        help="Number of local epochs: E")
    parser.add_argument('--dataset', type=str, default='cifar10', # todo 'mnist'
                        help="Type of algorithms:{mnist, cifar10,cifar100, fmnist}") 
    parser.add_argument('--select_ratio', type=float, default=1.0, # 0.1 for 3 nodes
                    help="the ratio of client selection in each round")
    parser.add_argument('--local_model', type=str, default='CNN',
                        help='Type of local model: {CNN, ResNet8, AlexNet}')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--random_seed', type=int, default=10,
                        help="random seed for the whole experiment")
    parser.add_argument('--exp_name', type=str, default=None,
                        help="experiment name")
    parser.add_argument('--corrupt_percent', type=int, default=1,
                        help="corrupted clients percentage for robustness exp, x/4")
    parser.add_argument('--group', type=str, default=None)
    parser.add_argument('--table', type=str, default=None)
    parser.add_argument('--inner_table', type=str, default=None)

    # Server function
    parser.add_argument('--group_model_num', type=int, default=1,
                        help="number of models within the group")
    parser.add_argument('--beta', type=float, default=0.1,
                        help="beta for connectivity loss")

    parser.add_argument('--server_method', type=str, default='fedavg',
                        help="FedAvg, Vanilla, SAM, none. none represents without aggregating model.")
    parser.add_argument('--server_epochs', type=int, default=20,
                        help="optimizer epochs on server, change it to 1, 2, 3, 5, 10")
    parser.add_argument('--server_optimizer', type=str, default='adam',
                        help="type of fusion optimizer, adam or sgd")
    parser.add_argument('--server_valid_ratio', type=float, default=0.02, # 0.1
                    help="the ratio of validate set in the central server")  
    parser.add_argument('--server_lr', type=float, default=1.0)
                        
    # Client function
    parser.add_argument('--client_method', type=str, default='local_train',
                        help="")
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help="optimizer: {sgd, adam}")
    parser.add_argument('--client_valid_ratio', type=float, default=0.3, # 0.3
                    help="the ratio of validate set in the clients")  
    parser.add_argument('--lr', type=float, default=0.1,  # 0.08
                        help='learning rate')
    parser.add_argument('--local_wd_rate', type=float, default=5e-4,  # 0.08
                        help='clients local wd rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--mu', type=float, default=0.001,
                        help="proximal term mu")

    # sam
    parser.add_argument('--sam', action='store_true')
    parser.add_argument('--sam_rho', type=float, default=0.05)

    # balanced loss
    parser.add_argument("--is_balanced_loss", action='store_true')

    # lora
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=8)

    args = parser.parse_args()

    return args
