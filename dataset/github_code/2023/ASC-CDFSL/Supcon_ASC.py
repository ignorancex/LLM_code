import copy

import numpy as np
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim
import torchvision.transforms as transforms
import data.additional_transforms as add_transforms
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import time
import os
import glob
from itertools import combinations
from data.cssf_datamgr_custom_collate import ContrastiveBatchifier, SetDataManager
from torch.utils.data import DataLoader
import methods.backbone
from data import get_datafiles
from io_utils import model_dict, parse_args, get_resume_file, get_best_file, get_assigned_file
from loss import fewshot_task_loss, cosine_dist, euclidean_dist, contrastive_loss
from data import ISIC_few_shot, EuroSAT_few_shot, CropDisease_few_shot, Chest_few_shot
# from data.datamgr import TransformLoader
from data.datamgr import SimpleDataManager
import configs
import matplotlib.pyplot as plt
import pickle

def finetune(source_loader, novel_loader, n_query=15, aug=False, pretrained_dataset='miniImagenet', n_way=5, n_support=5):

    iter_num = len(novel_loader)

    acc_all = []

    for k, (x, x_aug, y) in enumerate(novel_loader):
        supcon_datamgr = ContrastiveBatchifier(n_way=n_way, n_support=n_support, image_size=image_size,
                                               augstrength=0)

        supcon_dataloader = supcon_datamgr.get_loader([sample[:n_support] for sample in x_aug])  # 从conft的代码copy过来的
        x = x.cuda()

        ###############################################################################################
        # load pretrained model on miniImageNet
        src_net = model_dict[params.model]().cuda()
        tgt_net = model_dict[params.model]().cuda()

        modelfile = "./output/checkpoints/baseline/399.tar"

        tmp = torch.load(modelfile, map_location='cuda:0')
        state = tmp['state']

        state_keys = list(state.keys())
        for _, key in enumerate(state_keys):
            if "feature." in key:
                newkey = key.replace("feature.", "")  # an architecture model has attribute 'feature', load architecture feature to backbone by casting name from 'feature.trunk.xx' to 'trunk.xx'
                state[newkey] = state.pop(key)
            else:
                state.pop(key)

        src_net.load_state_dict(state, strict=False)
        tgt_net.load_state_dict(state, strict=False)

        tgt_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, tgt_net.parameters()), lr=0.005)  # 学习率根据文中的表格改

        ###############################################################################################
        n_query = x.size(1) - n_support
        x_var = Variable(x)

        # y_a_i = Variable(torch.from_numpy(np.repeat(range(n_way), n_support))).cuda()  # .view(n_way, n_support)  # (25,)
        x_b_i = x_var[:, n_support:, :, :, :].contiguous().view(n_way * n_query, *x.size()[2:])
        x_a_i = x_var[:, :n_support, :, :, :].contiguous().view(n_way * n_support, *x.size()[2:])  # (25, 3, 224, 224)

        src_net.train()
        tgt_net.train()
        tau = 0.05  # 根据超参数表格进行更改

        total_epoch = 100

        for epoch in range(total_epoch):
            if epoch % len(source_loader) == 0:
                src_iter = iter(source_loader)
            x_src, y_src = next(src_iter)
            x_src, y_src = x_src.cuda(), y_src.cuda()
            tgt_opt.zero_grad()

            x_l_fewshot, _ = next(iter(supcon_dataloader))
            x_l_fewshot = x_l_fewshot.cuda()
            shots_per_way = n_support
            if n_support == 1:
                shots_per_way = x_l_fewshot.size(1)
                x_l_fewshot = x_l_fewshot.view(x_l_fewshot.size(0) * x_l_fewshot.size(1), *x_l_fewshot.size()[2:])


            z_a_i = tgt_net(x_l_fewshot)

            with torch.no_grad():
                centre = torch.mean(tgt_net(x_a_i), dim=0, keepdim=True)

            for name, para in tgt_net.named_modules():
                if 'BN' in name or '1' in name:
                    para.track_running_stats = False

            z_1 = tgt_net(x_src)

            for name, para in tgt_net.named_modules():
                if 'BN' in name or '1' in name:
                    para.track_running_stats = True

            with torch.no_grad():
                z_2 = src_net(x_src)
                att = F.softmax(-euclidean_dist(z_2.detach(), centre.detach()).squeeze(1), dim=0) * 64

            rep_loss = (euclidean_dist(z_1, z_2).diag() * att).mean()
            loss = contrastive_loss(z_a_i, shots_per_way, n_way, tau=tau) + rep_loss
            loss.backward()
            tgt_opt.step()

        src_net.eval()
        tgt_net.eval()

        with torch.no_grad():
            scores, z_all = fewshot_task_loss(tgt_net, x, n_way, n_support, n_query)

        y_query = np.repeat(range(n_way), n_query)
        topk_scores, topk_labels = scores.data.topk(1, 1, True, True)
        topk_ind = topk_labels.cpu().numpy()

        top1_correct = np.sum(topk_ind[:, 0] == y_query)
        correct_this, count_this = float(top1_correct), len(y_query)
        print(correct_this / count_this * 100)

        acc_all.append((correct_this / count_this * 100))


        temp = np.asarray(acc_all)
        temp_mean = np.mean(temp)
        print('epoch:', k, 'temp_mean:', temp_mean)

        ###############################################################################################

    acc_all = np.asarray(acc_all)
    acc_mean = np.mean(acc_all)
    acc_std = np.std(acc_all)
    print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, 1.96 * acc_std / np.sqrt(iter_num)))


if __name__ == '__main__':
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    np.random.seed(10)
    torch.manual_seed(10)
    torch.cuda.manual_seed(10)
    torch.cuda.manual_seed_all(10)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    params = parse_args('train')
    params.distractor = False

    ##################################################################
    image_size = 224
    iter_num = 600
    # params.n_shot = 1
    n_query = max(1, int(16 * params.test_n_way / params.train_n_way))  # if test_n_way is smaller than train_n_way, reduce n_query to keep batch size small
    few_shot_params = dict(n_way=params.test_n_way, n_support=params.n_shot)

    freeze_backbone = params.freeze_backbone
    ##################################################################
    pretrained_dataset = "miniImagenet"

    dataset_name = "EuroSAT"  # change target dataset here
    dataset_names = [dataset_name]
    novel_loaders = []

    dataloader_params = dict(
        image_size=image_size,
        num_aug=400,
        n_way=params.test_n_way,
        n_support=params.n_shot,
        n_episode=iter_num,
        n_query=15)

    if dataset_name == 'ISIC':
        datamgr = ISIC_few_shot.SetDataManager(image_size, n_eposide=iter_num, n_query=15, **few_shot_params)
        novel_loader = datamgr.get_data_loader(aug=False)
    elif dataset_name == 'EuroSAT':
        datamgr = EuroSAT_few_shot.SetDataManager(image_size, n_eposide=iter_num, n_query=15, **few_shot_params)
        novel_loader = datamgr.get_data_loader(aug=False)
    elif dataset_name == 'CropDisease':
        datamgr = CropDisease_few_shot.SetDataManager(image_size, n_eposide=iter_num, n_query=15, **few_shot_params)
        novel_loader = datamgr.get_data_loader(aug=False)
    elif dataset_name == 'ChestX':
        datamgr = Chest_few_shot.SetDataManager(image_size, n_eposide=iter_num, n_query=15, **few_shot_params)
        novel_loader = datamgr.get_data_loader(aug=False)
    elif dataset_name == 'cub':
        inference_file = "./filelists/cub/novel.json"
        novel_loader = SetDataManager(**dataloader_params).get_data_loader(inference_file)
    elif dataset_name == 'cars':
        inference_file = "./filelists/cars/novel.json"
        novel_loader = SetDataManager(**dataloader_params).get_data_loader(inference_file)
    elif dataset_name == 'places':
        inference_file = "./filelists/places/novel.json"
        novel_loader = SetDataManager(**dataloader_params).get_data_loader(inference_file)
    elif dataset_name == 'plantae':
        inference_file = "./filelists/plantae/novel.json"
        novel_loader = SetDataManager(**dataloader_params).get_data_loader(inference_file)

    novel_loaders.append(novel_loader)
    distractor_file = './filelists/miniImagenet/base.json'
    source_loader = SimpleDataManager(image_size, 64).get_data_loader(
        distractor_file, aug=False)


    #########################################################################
    for idx, novel_loader in enumerate(novel_loaders):
        print(dataset_names[idx])
        start_epoch = params.start_epoch
        stop_epoch = params.stop_epoch
        print(freeze_backbone)

        # replace finetine() with your own method
        finetune(source_loader, novel_loader, n_query=15, pretrained_dataset=pretrained_dataset,
                 aug=False, **few_shot_params)
