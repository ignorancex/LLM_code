import argparse

""" Hyperparameter configuration """

parser = argparse.ArgumentParser(description='CFR-ISW')
## training setting
parser.add_argument('--batch_size', '-b', type=int, default=100,
                    help='input mini-batch size for training')
parser.add_argument('--train_seed','-ts', type=int, default=1,
                    help='random seed for parameters and training')                    
parser.add_argument('--epochs','-e', type=int, default=5000,
                    help='number of epochs to train (default: 20)')
## cuda & gpu
parser.add_argument('--no-gpu', action='store_true', default=False,
                    help='disables GPU training')
parser.add_argument('--gpu','-gpu', type=int, default=0,
                    help='device id of gpu')
## optimization setting
parser.add_argument('--optim','-o', type=str, default='Adam',
                    help='optimization algorithm')
parser.add_argument('--lr','-lr', type=float, default=0.0001,
                    help='learning rate')
parser.add_argument('--lrp','-lrp', type=float, default=0.0001,
                    help='learning rate for propensity score')

#0.0001
parser.add_argument('--weight_decay','-wd', type=float, default=0.001,
                    help='The strength of weight decay for Adam')          
## objective function
parser.add_argument('--weight_method','-w', type=str, default='PS',#'None','IPW','truncate','normalize','PS'
                    help='IPW Weighting schemes')
parser.add_argument('--imbalance_func','-imbf', type=str, default='wasserstein',
                    help='Function for balancing feature reprensetation')
parser.add_argument('--reg_alpha','-ra', type=float, default=1.0,
                    help='Regularization parameter for balancing feature representation')         
parser.add_argument('--reg_beta','-rb', type=float, default=1.0,
                    help='Regularization parameter for cross entropy loss')         
parser.add_argument('--reg_decay','-rd', type=float, default=1.0,
                    help='decay the strength of regularization as number of epochs increases')                      
parser.add_argument('--reg_soft','-rs', type=float, default=0.001,
                    help='Regularization parameter for soft rank')
parser.add_argument('--Pareto_est','-pe', type=str, default='PWM',
                    help='Method to estimate parameters of the Pareto distribution')
parser.add_argument('--heavi_method','-hm', type=str, default='sigmoid',
                    help='Method to calcurate the soft Heaviside function')
parser.add_argument('--heavi_k','-hk', type=float, default=1,
                    help='k of tanh(k*x)')

## NN architecture parameters
parser.add_argument('--enc_num_layers','-enl', type=int, default=3,
                    help='the number of layers of feature representation (encoder)')
parser.add_argument('--enc_h_dim','-ehd', type=int, default=200,
                    help='the feature dimension of feature representation (encoder)')
parser.add_argument('--enc_out_dim','-eod', type=int, default=200,
                    help='the output dimension of feature representation (encoder)')
parser.add_argument('--enc_dropout','-edp', type=float, default=0,
                    help='The ratio of drop out for feature representation (encoder)')
parser.add_argument('--bool_outhead','-oh', action='store_true', default=True,
                    help='split NNs into Phi and outcome heads: True for CFR (Shalit+; ICML2017); False for BNN (Johansson+; ICML2016)')
parser.add_argument('--oh_num_layers','-onl', type=int, default=3,
                    help='the number of layers of outcome head (decoder)')
parser.add_argument('--oh_h_dim','-ohd', type=int, default=100,
                    help='the feature dimension of outcome head (decoder)')
parser.add_argument('--oh_out_dim','-ood', type=int, default=1,
                    help='the output dimension of outcome head (decoder)')
parser.add_argument('--oh_dropout','-odp', type=float, default=0,
                    help='The ratio of drop out for outcome head (decoder)')
parser.add_argument('--propensity_net_type','-pnt', type=str, default='MLP',
                    help='the number of layers of outcome head (decoder)')
parser.add_argument('--prop_num_layers','-pnl', type=int, default=3,
                    help='the number of layers of outcome head (decoder)')
parser.add_argument('--prop_h_dim','-phd', type=int, default=100,
                    help='the feature dimension of outcome head (decoder)')
parser.add_argument('--prop_dropout','-pdp', type=float, default=0,
                    help='The ratio of drop out for outcome head (decoder)')
#dropout 0.145

## choose dataset 
parser.add_argument('--data','-d', type=str, default='news',
                    help='dataset type')
parser.add_argument('--synth_param','-sp', type=str, default='2,500,50',
                    help='Parameters of synth data (diff, n, 2p)')
parser.add_argument('--data_seed','-ds', type=int, default=1,
                    help='random seed for dataset index')
parser.add_argument('--data_index','-di', type=str, default='0',
                    help='data indices to use separated by comma (default is to use all data)')

## preprocess dataset
parser.add_argument('--noise','-noise', type=float, default=0.,
                    help='The magnitude of noise (for synthetic dataset)')                    
parser.add_argument('--comp','-comp', type=int, default=4,
                    help='The size of dimension for PCA for IHDP and News datasets')
parser.add_argument('--tfidf','-tfidf', action='store_true', default=False,
                    help='Use tfidf or not for News dataset')
parser.add_argument('--shuffle','-shf', action='store_false', default=True,
                    help='Use data shuffle')
## save results
parser.add_argument('--verbose','-verbose', type=int, default=1,
                    help='validation interval')
parser.add_argument('--report','-r', action='store_true', default=False,
                    help='For Saving the current Model')
parser.add_argument('--save_result','-sr', action='store_false', default=True,
                    help='For not saving the training result')
parser.add_argument('--output_name','-output', type=str, default='temp.txt',
                    help='the name of output file')
parser.add_argument('--output_dir','-output_d', type=str, default='results/',
                    help='the name of output directory')
args = parser.parse_args()
