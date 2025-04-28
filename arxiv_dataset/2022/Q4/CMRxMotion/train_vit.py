import numpy as np

import torch
from torch.utils.data.dataloader import DataLoader

from framework_pytorch.core import Trainer
from framework_pytorch.utils import loss_fuctions as LF, process_methods as P, utils as U
from framework_pytorch.utils.data_augmentation import RandomForegroundCrop

from customize.cmr_dataset import CMRDataset
from customize.timm_models import ViT
from customize.model_cls import ClsModel
from customize import _tools as T

# training parameters
epochs = 10
learning_rate = 0.0001
train_batch_size = 10
eval_batch_size = 10
slice_ratio = None
save_frequency = None
log_train_image = False
log_validation_image = True
loss_function = {LF.CrossEntropy(): 1.0} 

# model setting
pretrained = True
freeze = False
drop_rate = 0.5
input_channels = 1

# data setting
crop_size = [224,224]
foreground_ratio = 0.8
test_num_samples = 50
balance = True
mag = True
mag_name = 'org' if not mag else 'mag'

root_output_path = f'./results/vit_{mag_name}'

# data and results path
data_path = './data_cls'
root_output_path = './results/' + root_output_path


data_list = np.load('./data_cls/5fold.npy', allow_pickle=True).item()
for ifold in range(1, 6):
    output_path = root_output_path + f'/fold{ifold}/'
    valid_1 = data_list[f'fold_{ifold}_c1']
    valid_2 = data_list[f'fold_{ifold}_c2']
    valid_3 = data_list[f'fold_{ifold}_c3']

    train_1 = []
    train_2 = []
    train_3 = []
    for i in range(1, 6):
        if not i == ifold:
            train_1 += data_list[f'fold_{i}_c1']
            train_2 += data_list[f'fold_{i}_c2']
            train_3 += data_list[f'fold_{i}_c3']

    suffix = '*_pdorg.tif'
    train_1 = T.fill_imgs(train_1, slice_ratio, suffix)
    train_2 = T.fill_imgs(train_2, slice_ratio, suffix)
    train_3 = T.fill_imgs(train_3, slice_ratio, suffix)
    valid_1 = T.fill_imgs(valid_1, slice_ratio, suffix)
    valid_2 = T.fill_imgs(valid_2, slice_ratio, suffix)
    valid_3 = T.fill_imgs(valid_3, slice_ratio, suffix)
    if not balance:
        train_1 = train_1 + train_2 + train_3
        train_2 = None
        train_3 = None
        valid_1 = valid_1 + valid_2 + valid_3
        valid_2 = None
        valid_3 = None

    img_suffix = 'pdorg.tif'
    lab_suffix = 'org_lab'
    seg_suffix = 'pdorg_lab.png'

    pre = {lab_suffix: [lambda x: np.expand_dims(np.array([i==x-1 for i in range(3)], np.float32), 0)],
            seg_suffix: [lambda x: [0]]
            }
    if mag: 
        pre.update({img_suffix: [P.Magnitude(), P.Transpose([2, 0, 1]), P.ExpandDim(0)]})
    else:
        pre.update({img_suffix: [P.ExpandDim(-1), P.Transpose([2, 0, 1]), P.ExpandDim(0)]})

   

    aug = RandomForegroundCrop(img_suffix, seg_suffix, tar_size=crop_size, foreground_ratio=foreground_ratio)

    # set device to gpu if gpu is available, otherwise use cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def collate_fn(batch):
        batch_dict = None
        for b in batch:
            batch_dict = U.dict_concat(batch_dict, b)

        for key in batch_dict:
            batch_dict[key] = torch.tensor(batch_dict[key])
        
        return batch_dict
    # build pytorch dataset, see core/basic_dataset
    train_set = CMRDataset(train_1, train_2, train_3, [img_suffix, lab_suffix], pre, aug, seg_suffix=seg_suffix, shuffle=True)
    valid_set = CMRDataset(valid_1, valid_2, valid_3, [img_suffix, lab_suffix], pre, aug, seg_suffix=seg_suffix)

    # build pytorch data loader, shuffle train set while training
    trainloader = DataLoader(train_set, batch_size=train_batch_size, collate_fn=collate_fn)
    validloader = DataLoader(valid_set, batch_size=eval_batch_size, collate_fn=collate_fn)

    # get a random image for graph draw
    # random_img = torch.tensor(train_set[0][img_suffix]).to(device)
    random_img = None

    # build model
    net = ViT(num_classes=3, pretrained=pretrained, input_channels=input_channels, freeze=freeze)
    # init optimizer, adam is used here
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # init the model
    model = ClsModel(net, optimizer, device, img_suffix, lab_suffix, loss_functions=loss_function)

    # init train and start train
    trainer = Trainer(model)
    trainer.train(trainloader, validloader, 
        epochs=epochs, 
        output_path=output_path, 
        log_train_image=log_train_image, 
        log_validation_image=log_validation_image,
        log_graph_input=random_img,
        save_frequency=save_frequency)

