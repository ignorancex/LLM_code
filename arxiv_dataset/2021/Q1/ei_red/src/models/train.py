import os, sys
from time import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import get_files_and_labels, load_model, run_model
from data_gen import REDDataset


debug = False
use_pbar = True

bs = 32
n_cpu = 4
n_epochs = 20


assert len(sys.argv) == 3, 'Run with method as first arg, and GPU ID' + \
                           ' as the second. For example: \n' + \
                           'train.py laplace 0'

model_type = sys.argv[1]
assert model_type in {'laplace','co_occur','direct'}

os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[2]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_dir = f'../../models/{model_type}/'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_path = os.path.join(model_dir,'model.pt')
model_hist = os.path.join(model_dir,'hist.txt')


files_tr, labels_tr = get_files_and_labels('train')
files_va, labels_va = get_files_and_labels('val')

n_classes = max(labels_tr) + 1
pre_proc, model = load_model(model_type,n_classes)

dg_kwargs = {'shuffle': True, 'batch_size': bs, 'num_workers': n_cpu}
dg_tr = DataLoader(REDDataset(files_tr,labels_tr,pre_proc,debug),**dg_kwargs)
dg_va = DataLoader(REDDataset(files_va,labels_va,pre_proc,debug),**dg_kwargs)

model.to(device)


optimizer = optim.Adam(model.parameters())

best_val_loss = float('inf')
cum_out_str = str()

for i in range(n_epochs):
    t1 = time()
    print(f'\nEPOCH {i+1} of {n_epochs}')
    tr_loss, tr_acc,_,_ = run_model(model,device,dg_tr,optimizer,use_pbar)
    va_loss, va_acc,_,_ = run_model(model,device,dg_va,None,use_pbar)


    out_str = f'Epoch {i+1}: {tr_loss:0.4f} {tr_acc:0.4f} ' + \
               f'{va_loss:0.4f} {va_acc:0.4f}'
    print(out_str)
    cum_out_str += out_str + '\n'

    t2 = time()
    if not use_pbar: print(f'{t2-t1:0.2f} seconds')

    if va_loss < best_val_loss:
        best_val_loss = va_loss
        torch.save(model.state_dict(),model_path)


with open(model_hist,'w+') as f: f.write(cum_out_str)


