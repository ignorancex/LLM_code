import sys
import glob
import scipy.io as sio
import numpy as np

import torch
from torch.utils.data.dataloader import DataLoader

from framework_pytorch.core import Trainer, BasicDataset
from framework_pytorch.modelnet.model_vnet_nointer import VNet
from framework_pytorch.modelnet.model import SegModel
from framework_pytorch.utils import loss_fuctions as LF, process_methods as P, utils as U
from framework_pytorch.utils.data_augmentation import RandomAugmentation

# training parameters
epochs = 500
learning_rate = 1e-4
train_batch_size = 1
eval_batch_size = 1
loss_function = {LF.CrossEntropy(): 1.0, LF.SoftDice(): 1.0} # loss functions {method: weight}
aug_rate = 0.8
output_path = './results/cmr_unet'

# training setttings
test_only = False

log_train_image = False
log_validation_image = True


data_path = './CMRxMotion/training/'

data_list = np.load(data_path + '/train_list.npy', allow_pickle=True).item()

train = data_list['d1_train'] + data_list['d2_train']
valid = data_list['d1_valid'] + data_list['d2_valid']

train = [data_path + '/data/' + x.split('-E')[0] + '/' + x + '.nii.gz' for x in train]
valid = [data_path + '/data/' + x.split('-E')[0] + '/' + x + '.nii.gz' for x in valid]



# set suffix for image and label (the difference between image path and label path) for data loading
img_suffix = '.nii.gz'
lab_suffix = '-label.nii.gz'

# set pre-process functions for image and label
pre = {img_suffix: [np.squeeze,
                    P.CenterCrop([256,256,10]),
                    P.min_max,
                    P.ExpandDim(0),
                    ],
        lab_suffix: [P.CenterCrop([256,256,10]),
                    #  lambda x: np.array(x>0.5, np.uint8),
                     P.OneHot([0,1,2,3])]
        }

aug = RandomAugmentation(img_suffix, lab_suffix, aug_rate=aug_rate)

# set device to gpu if gpu is available, otherwise use cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# build pytorch dataset, see core/basic_dataset
train_set = BasicDataset(train, [img_suffix, lab_suffix], pre, aug)
valid_set = BasicDataset(valid, [img_suffix, lab_suffix], pre)

# build pytorch data loader, shuffle train set while training
trainloader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
validloader = DataLoader(valid_set, batch_size=eval_batch_size)

# get a random image for graph draw
random_img = torch.tensor([train_set[0][img_suffix]]).to(device)

# build 3D unet for 5 classes segmentation, the input channel is 1
net = VNet(input_channel=1, class_number=4)
# init optimizer, adam is used here
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
# init the model, #TODO: more details see modelnet/model
model = SegModel(net, optimizer, device, img_suffix, lab_suffix, dropout_rate=0, loss_functions=loss_function)

# init train and start train
trainer = Trainer(model)
if not test_only:
    trainer.train(trainloader, validloader, 
        epochs=epochs, 
        output_path=output_path, 
        log_train_image=log_train_image, 
        log_validation_image=log_validation_image,
        log_graph_input=random_img)

# test on test set
trainer.restore(output_path+'/ckpt/model_final.pt')

# test set 1
test1 = data_list['d1_valid'] + data_list['d2_valid']
test1 = [data_path + '/data/' + x.split('-E')[0] + '/' + x + '.nii.gz' for x in test1]
test1_set = BasicDataset(test1, [img_suffix, lab_suffix], pre)
test1loader = DataLoader(test1_set, batch_size=eval_batch_size)

test_results = trainer.test(test1loader, output_path, log_image=True, logsuffix='1')
sio.savemat(f'{output_path}/test1_results.mat', test_results)
with open(f'{output_path}/test1_results.txt', 'a+') as f:
    f.write(U.dict_to_str(test_results) + '\n')

