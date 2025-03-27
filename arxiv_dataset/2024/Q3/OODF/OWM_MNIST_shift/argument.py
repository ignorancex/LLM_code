import argparse

def parser():
    parser = argparse.ArgumentParser(description = 'Out-of-distribution forgetting in OWM')
    parser.add_argument('--dataset', type = str, default = 'mnist', choices = ['mnist'], help = 'dataset')
    parser.add_argument('--data_root', default = '/data0/user/lxguo/Data', help='the directory to save the dataset')
    
    # parameters for training
    parser.add_argument('--model_name', '-m_n', type = str, default = 'owm_fc', choices = ['owm_fc'])
    parser.add_argument('--batch_size', '-b', type = int, default = 128, help = 'batch size')
    parser.add_argument('--num_epochs', '-n_e', type = int, default = 30, help = 'the maximum numbers of the model see a sample')
    parser.add_argument('--learning_rate', '-lr', type = float, default = 0.2, help = 'learning rate')
    parser.add_argument('--lambda_loss', type = float, default = 1e-3, help = 'the regularization parameter')
    parser.add_argument('--owmfc_alpha_list', type = float, nargs = '+', default = [0.9, 0.6], help = 'the alpha list of OWM algorithm (input format 0 1 2 ...)')
    parser.add_argument('--owmfc_lambda_list', type = float, nargs = '+', default = [0.001, 1.0], help = 'the lambda list of OWM algorithm (input format 0 1 2 ...)')
    parser.add_argument('--CL_method', type = str, default = 'owm', choices = ['owm', 'none', 'joint'], help = 'continual learning algorithm')
    parser.add_argument('--num_hidden', type = int, default = 800, help = 'the number of neurons in MLP hidden layer')
    parser.add_argument('--num_classes', type = int, default = 10, help = 'the number of classification')
    parser.add_argument('--num_tasks', type = int, default = 10, help = 'the number of tasks')
    parser.add_argument('--momentum', type = float, default = 0.9, help = 'the momentum parameter')
    parser.add_argument('--order_list', type = int, nargs = '+', default = list(range(10)), help = 'the learning order of sequence (input format 0 1 2 ...)')
    parser.add_argument('--optimizer_type', '-opti', type = str, default = 'SGD', choices = ['SGD', 'Adam'], help = 'the type of optimizer')
    parser.add_argument('--clipgrad', type = float, default = 10.0, help = 'Clips gradient norm of an iterable of parameters.')
    parser.add_argument('--list_eval', type = bool, default = False, help = 'Whether calculate classification results of each class')
    
    # parameters for evaluating OODF
    parser.add_argument('--shift', action = 'store_true', default = False)

    # parameters for addressing OODF
    parser.add_argument('--argmt_perc', type = float, default = 4.0, help = 'How many samples of additional 11th class')
    parser.add_argument('--learn_noise_before', '-l_n_b', type = bool, default = False, help = 'Whether learn noise dataset before learning')
    
    # options for details
    parser.add_argument('--print_frequency', '-p_f', type = int, default = 5, help = 'the print frequency of training stage')

    return parser.parse_args()

def print_args(args, logger=None):
    for k, v in vars(args).items():
        if logger is not None:
            logger.info('{:<16} : {}'.format(k, v))
        else:
            print('{:<16} : {}'.format(k, v))

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'