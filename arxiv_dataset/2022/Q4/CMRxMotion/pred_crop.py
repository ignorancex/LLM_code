import os
import sys
import glob
import scipy.io as sio
import numpy as np

import torch
import torch.nn.functional as F

from framework_pytorch.core import Trainer, BasicDataset
from framework_pytorch.modelnet.model_vnet_nointer import VNet
from framework_pytorch.modelnet.model import SegModel
from framework_pytorch.utils import loss_fuctions as LF, process_methods as P, utils as U, eval_function as EF
from framework_pytorch.utils.data_augmentation import RandomAugmentation

from PIL import Image
from skimage.measure import label   

def getLargestCC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

def getBoundingBox(lab, padding=2, minsize=[96,96]):
    xs = 0
    xe = lab.shape[0]
    ys = 0
    ye = lab.shape[1]
    for x in range(lab.shape[0]):
        if lab[x,:,:].max() > 0.5:
            xs = max(x-padding, 0)
            break
    for x in range(lab.shape[0]-1, -1, -1):
        if lab[x,:,:].max() > 0.5:
            xe = min(x+1+padding, lab.shape[0])
            break
    for y in range(lab.shape[1]):
        if lab[:,y,:].max() > 0.5:
            ys = max(y-padding, 0)
            break
    for y in range(lab.shape[1]-1, -1, -1):
        if lab[:,y,:].max() > 0.5:
            ye = min(y+1+padding, lab.shape[1])
            break

    while xe-xs < minsize[0]:
        xe = min(xe+1, lab.shape[0])
        xs = max(xs-1, 0)
    while ye-ys < minsize[1]:
        ye = min(ye+1, lab.shape[1])
        ys = max(ys-1, 0)
    return xs, xe, ys, ye



# training parameters
epochs = 500
learning_rate = 1e-4
train_batch_size = 1
eval_batch_size = 1
loss_function = {LF.CrossEntropy(): 1.0, LF.SoftDice(): 1.0} # loss functions {method: weight}
aug_rate = 0.8
segmodel_path = './results/cmr_unet'

data_path = './CMRxMotion/training/'
data_list = np.load(data_path + '/train_list.npy', allow_pickle=True).item()

# set suffix for image and label (the difference between image path and label path) for data loading
img_suffix = '.nii.gz'
lab_suffix = '-label.nii.gz'

pre = {img_suffix: [np.squeeze,
                    P.CenterCrop([256,256,20]),
                    P.min_max,
                    P.ExpandDim(0),
                    ]
    }

pre_nc = {img_suffix: [np.squeeze, P.min_max]}


# set device to gpu if gpu is available, otherwise use cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = VNet(input_channel=1, class_number=2)
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
model = SegModel(net, optimizer, device, img_suffix, lab_suffix, dropout_rate=0, loss_functions=loss_function)
# init train and start train
trainer = Trainer(model)
trainer.restore(segmodel_path+'/ckpt/model_final.pt')

data_list = glob.glob(data_path + '/data/**/*.nii.gz')
data_list = [x for x in data_list if '-label.nii.gz' not in x]

data_set = BasicDataset(data_list, [img_suffix], pre)
data_set_nc = BasicDataset(data_list, [img_suffix], pre_nc)

l1 = np.loadtxt(data_path + '/lab1.txt', dtype='str')
l2 = np.loadtxt(data_path + '/lab2.txt', dtype='str')
l3 = np.loadtxt(data_path + '/lab3.txt', dtype='str')

xls = []
yls = []

model.net.eval()
for i in range(len(data_set)):
    fpath = data_set._file_list[i]
    fname = os.path.basename(fpath).split('.nii.gz')[0]
    clab = None
    if fname in l1:
        clab = 1
    elif fname in l2:
        clab = 2
    elif fname in l3:
        clab = 3
    else:
        raise 'error!'
    data_dict = data_set.__getitem__(i)
    img = np.expand_dims(data_dict[img_suffix], 0)
    img = torch.from_numpy(img)
    img_g = img.to(device=device)

    with torch.no_grad():
        logits = model.net(img_g)

    pred = F.softmax(logits, dim=1)

    pred = np.argmax(pred.cpu().numpy(), 1)[0]

    pred_l = getLargestCC(pred).astype(np.uint8)

    print(f'{i}/{len(data_set)}: {fname} - {clab}')

    cls_file_path = './data_cls'

    img = img.numpy().squeeze()

    nc_img = data_set_nc.__getitem__(i)[img_suffix].squeeze()
    assert nc_img.shape[-1] == pred_l.shape[-1]
    nc_pred_l = np.uint8(P.CenterPadding(nc_img.shape)(pred_l))


    tpath = f'{cls_file_path}/lab_{clab}/{fname}'
    if not os.path.exists(tpath):
        os.makedirs(tpath)
    for j in range(img.shape[-1]):
        if np.sum(nc_pred_l[...,j]) == 0:
            continue
        Image.fromarray(nc_pred_l[...,j]*255).save(f'{tpath}/{j}_pdorg_lab.png')
        Image.fromarray(nc_img[...,j]).save(f'{tpath}/{j}_pdorg.tif')




