import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="heroGraph arguments")
    # dataset configuration
    parser.add_argument('--dataset', type=str, default='task_1', help="available datasets: [task_1, task_2, task_3]")
    parser.add_argument('--k_negative_num', type=int, default=5, help="the negative sample number")
    parser.add_argument('--batch_size', type=int, default=2048, help="the sampled edges number for each subgraph")
    parser.add_argument('--neighbour_sample_number', type=str, default='10,10', help='the number of neighbourhood is sampled')

    # model configuration
    parser.add_argument('--n_layer', type=int, default=2, help="the layer num of heroGraph")
    parser.add_argument('--input_dim', type=int, default=128, help="the size of hidden dimension")
    parser.add_argument('--hidden_dim', type=int, default=128, help="the size of hidden dimension")
    parser.add_argument('--output_dim', type=int, default=128, help="the size of hidden dimension")
    parser.add_argument('--activation', type=str, default='both')
    parser.add_argument('--dropout', type=float, default=0.1, help="the probability of dropout")

    # training configuration
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4, help="the learning rate")
    parser.add_argument('--alpha', type=float, default=0.5, help='loss hyper-parameter')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='L2 regularization')
    parser.add_argument('--log_filepath', type=str, default='run', help='dir to save training information')
    parser.add_argument('--enable_logger', type=bool, default=True, help='enable to start the logger')
    parser.add_argument('--random_seed', type=int, default=1024, help='random seed for training and testing')
    parser.add_argument('--step_size', type=int, default=10, help='learning decay step size')
    parser.add_argument('--gamma', type=float, default=0.1, help='learning rate decay ratio')

    parser.add_argument('--device', type=str, default='gpu', help='device to run the program')
    parser.add_argument('--comment', type=str, default='amazon_data', help='comment for each experiment')

    parser.add_argument('--data_domains', default=['movie', 'music', 'book'])
    parser.add_argument('--graph_domains', default=['movie', 'music', 'book'])
    parser.add_argument('--train_and_eval_domains', default=['movie', 'music', 'book'])
    parser.add_argument('--num_works', type=int, default=10)
    parser.add_argument('--projection_on', type=bool, default=True, help='enable to start the logger')
    parser.add_argument('--model', type=str, default='hybrid_rgcn', help='model type')
    parser.add_argument('--add_self', type=bool, default=True, help='model type')
    parser.add_argument('--self_loop_filter', type=list, default=['book', 'music'], help='model type')

    args = parser.parse_args()
    return args

args = parse_args()
