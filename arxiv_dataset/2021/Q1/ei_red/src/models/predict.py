import torch
import os, sys
from time import time
import numpy as np
from utils import get_files_and_labels, load_model, run_model, get_all_files
from data_gen import REDDataset
from torch.utils.data import DataLoader

debug = False
use_pbar = True

bs = 32
n_cpu = 4


#assert len(sys.argv) == 4, 'Run with dataset as first arg and model type as second and gpu as third, ex: python train.py data_med laplace 0'

assert len(sys.argv) == 3, 'Run with method as the first arg, and GPU ID as the second, ex: python predict.py direct 0'

model_type = sys.argv[1]
assert model_type in {'laplace','co_occur', 'direct'}

os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[2]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_dir = '../../models/' + model_type
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
model_path = os.path.join(model_dir,'model.pt')
model_hist = os.path.join(model_dir,'hist.txt')




files_tr, labels_tr = get_files_and_labels('train')
#files_va, labels_va = get_files_and_labels('val')
#files_te, labels_te = get_files_and_labels('test')

n_classes = np.max(labels_tr) + 1

files_tr = get_all_files('train')
files_va = get_all_files('val')
files_te = get_all_files('test')



pre_proc, model = load_model(model_type,n_classes,weights_path=model_path)

dg_kwargs = {'shuffle': False, 'batch_size': bs, 'num_workers': n_cpu}


dg_tr = DataLoader(REDDataset(files_tr,[0,]*len(files_tr),pre_proc,debug),**dg_kwargs)
dg_va = DataLoader(REDDataset(files_va,[0,]*len(files_va),pre_proc,debug),**dg_kwargs)
dg_te = DataLoader(REDDataset(files_te,[0,]*len(files_te),pre_proc,debug),**dg_kwargs)

model.to(device)


#for dg, name in [(dg_tr,'train'),(dg_va,'val'),(dg_te,'test')]:
for dg, name in [(dg_te,'test')]:

    _, _, preds, _ = run_model(model,device,dg,use_pbar=True)

    this_dir = os.path.join(model_dir,name)
    if not os.path.exists(this_dir): os.mkdir(this_dir)

    np.save(os.path.join(this_dir,'outputs.npy'),preds)
    with open(os.path.join(this_dir,'fnames.txt'),'w+') as f:
        f.write('\n'.join(dg.dataset.fnames))


