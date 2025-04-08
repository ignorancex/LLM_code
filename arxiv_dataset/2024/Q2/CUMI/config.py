import argparse
import os

root = os.getcwd()+'/datasets/'
def get_args():
    parser = argparse.ArgumentParser(description='hyper-parameter in deterministic CUMI model')
    parser.add_argument('--lr', type=float, default=1e-1, help='learning rate (default=0.01)') 
    parser.add_argument('--batchsize', type=int, default=128, help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--data_path', type=str, default=root+'/XRMB.mat', help=""" 'data/XRMB.mat' """)
    parser.add_argument('--beta', type=list, default=[-0.001, 0.001], help="""
    loss = loss_ce +  + sum(reconstruction_error) + beta[0] * H_zc + beta[1] * TC_loss 
                                                                            """)
    return parser.parse_args()

