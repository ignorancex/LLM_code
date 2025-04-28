import argparse

def get_config():
    parser = argparse.ArgumentParser()
    ## distributed
    parser.add_argument('--topo', default='ring', type=str, metavar='N',
                    help='Netwok connection topology')
    parser.add_argument('-na', '--nodes', default=2, type=int,
                    help='number of agents to connect')
    parser.add_argument('--ppB', default=2, type = int,
                        help='Number of branches in B-ary Tree graph')
    
    ## communication among GPUs
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:33069', type=str,
                    help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
    parser.add_argument('--gpu_rank', default=0, type=int)
    
    ## training hyperparameter
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('-lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('-ms', '--milestones', help='scheduler, default=[50, 80, 90]', nargs='+',
                    dest='milestones', type=int, default=[])
    parser.add_argument('-g', '--gamma', help='gamma, default=0.1',
                        dest='gamma', type=float, default=0.1)
    parser.add_argument('-ht', '--data_hete', action='store_true', default=False,
                        help='enable data heterogeneous by manually sort the data related to the label')
    
    ## training setting
    parser.add_argument('--model', default='resnet18',type=str,
                        help="what model is used")
    parser.add_argument('--datasets', default='cifar10',type=str,
                        help="what kind of dataset is used")
    
    parser.add_argument('--iterations', default=2000, type=int,
                        help="Number of iterations used for stochastic gradient descent")
    parser.add_argument('--shuffle', type=bool, default=True,
                        dest='shuffle', help='if `shuffle = True`, we make each nodes use different parts of the whole dataset at each epoch.')
    parser.add_argument('--seed', default=0, type=int,
                        help="random seed for shuffle sampler")
    
    ## warm-up procedure
    parser.add_argument('--warm_up', default=0, type=int,
                        help='warm-up epochs to get a better initial model')
    parser.add_argument('--warm_up_lr', default=0.0004, type=float,
                        help='learning rate in warm-up procedure')
    

    parser.add_argument('--record', default=300, type = int,
                        help='the check points that record the results (default means in every 200 iterations calculate the loss and accuracy)')


    ## record only
    parser.add_argument('--keep', default=1, type = int)
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                    help='number of total epochs to run')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')


    ## epochs + momentum sgd


    args = parser.parse_args()
    return args