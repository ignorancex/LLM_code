import torch
from model.model import IBCI
from utils import set_seed
from wrapper import train_IBCI
from config import get_args
from dataload import import_data
from sklearn.model_selection import StratifiedKFold, train_test_split
from scipy.io import loadmat
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = get_args()
temp = loadmat(args.data_path)
input_temp = torch.tensor(temp['X'][0][0]).float()
label_temp = torch.tensor(np.squeeze(temp['Y'])-1).long()
data_set = import_data(data_path=args.data_path)
set_seed(123)
input_dims=[temp['X'][0][i].shape[1] for i in range(temp['X'][0].shape[0])]
class_num=len(np.unique(label_temp))
train_idxs, test_idxs = train_test_split(range(len(input_temp)), test_size=0.2, random_state=42)
net = IBCI(input_dims=input_dims, class_num=class_num).to(device)
train_subset = torch.utils.data.Subset(data_set, train_idxs)
test_subset = torch.utils.data.Subset(data_set, test_idxs)
trainloader = torch.utils.data.DataLoader(train_subset, batch_size=args.batchsize, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(test_subset, batch_size=args.batchsize, shuffle=False, num_workers=0)


if __name__ == '__main__':
    report = train_IBCI(args, trainloader, testloader, net)
    print("Results:")
    print(f"Accuracy: {report[0]}\nPrecision: {report[1]}\nRecall: {report[2]}\nF1 Score: {report[3]}")
