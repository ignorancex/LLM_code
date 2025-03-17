import argparse

import joblib
import numpy as np
import torch
from model.smooth_cross_entropy import smooth_crossentropy
from data.cifar import Cifar10, Cifar100
from utility.initialize import initialize
from tqdm import tqdm
import os

from model.resnet import ResNet34
print('pid:', os.getpid())

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=500, type=int, help="Batch size used in the training and validation loop.")
parser.add_argument("--momentum", default=0.9, type=float, help="SGD Momentum.")
parser.add_argument("--threads", default=2, type=int, help="Number of CPU threads for dataloaders.")
parser.add_argument("--model", default='res-34', type=str, help="network architecture")
parser.add_argument("--dataset", default='cifar10', type=str, help="dataset")

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--device', default="cuda:0", type=str, )

parser.add_argument('--num_points', type=int, default=11, help='num_points')
parser.add_argument('--min_epoch', type=int, default=0, help='min_epoch')
parser.add_argument('--max_epoch', type=int, default=199, help='max_epoch')

parser.add_argument('--base_dir', default="result_rebuttal", type=str, )
parser.add_argument('--ckpt1', default="none", type=str, )
parser.add_argument('--ckpt2', default="random", type=str, )

args = parser.parse_args()

args.ckpt_dir1 = f"{args.base_dir}/res-34_optadam-0.001-eps1e-08_init{args.ckpt1}-mv-100.0-1.0-5000_seed0_ckpt"
args.ckpt_dir2 = f"{args.base_dir}/res-34_optadam-0.001-eps1e-08_init{args.ckpt2}-mv-100.0-1.0-5000_seed0_ckpt"
args.save_path = f"{args.base_dir}/{args.ckpt1}-{args.ckpt2}-lin-connect_{args.min_epoch}-{args.max_epoch}.pkl"
initialize(args.seed)

device = torch.device(args.device if torch.cuda.is_available() else "cpu")
dataset = Cifar10(args.batch_size, args.threads)

def interpolate_weights(model1, model2, alpha):
    """Linearly interpolate between two models' weights."""
    interpolated_state_dict = {}
    for key in model1.state_dict().keys():
        interpolated_state_dict[key] = (1 - alpha) * model1.state_dict()[key] + alpha * model2.state_dict()[key]
    return interpolated_state_dict

def evaluate_loss(model, dataloader, criterion, device):
    """Evaluate the loss of a model on a given dataset."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = (b.to(device) for b in batch)
            #inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.mean().item() * targets.size(0)
    return total_loss / len(dataloader.dataset)

def linear_mode_connectivity(model1, model2, dataloader, criterion, device, num_points=10):
    """Perform linear mode connectivity by interpolating between two models."""
    alpha_values = np.linspace(0, 1, num_points)
    losses = []

    for alpha in tqdm(alpha_values, desc="Interpolating"):
        interpolated_model = ResNet34().to(device) # model1.__class__()  # Create a new instance of the model
        interpolated_model.load_state_dict(interpolate_weights(model1, model2, alpha))
        #interpolated_model.to(device)

        loss = evaluate_loss(interpolated_model, dataloader, criterion, device)
        print(f'alpha={alpha}, loss={loss}')
        losses.append(loss)

    return alpha_values, losses

def run_linear_connectivity(args):

    lin_connect=[]
    model1 = ResNet34().to(device)
    model2 = ResNet34().to(device)
    model1.eval()
    model2.eval()
    dataloader = dataset.test
    criterion = smooth_crossentropy
    for epoch in range(args.min_epoch, args.max_epoch+1):
        path1 = args.ckpt_dir1+'/' +str(epoch)+'.pt'
        path2 = args.ckpt_dir2+'/' +str(epoch)+'.pt'

        print('\n Doing ', epoch)
        print(path1,path2)


        model1.load_state_dict(torch.load(path1,map_location=device))
        model2.load_state_dict(torch.load(path2,map_location=device))

        alpha_values, losses = linear_mode_connectivity(model1, model2, dataloader, criterion, device, num_points=args.num_points)
        lin_connect.append([alpha_values, losses])

        joblib.dump(lin_connect, args.save_path)

    print('final done')


if __name__ == "__main__":
    run_linear_connectivity(args)

# nohup  python -u linear_connectivity.py > nohups/linear_connectivity.log  2>&1 &