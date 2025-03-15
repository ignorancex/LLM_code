import argparse
import sys
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils
from ImageLoader_classification import ImageFolder

from tqdm import tqdm

from vqvae_codebooks import VQVAE, Classifier
from scheduler import CycleScheduler
import distributed as dist

import glob
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
from util.image_pool import ImagePool
from torchvision import models
import random
import yaml



def classification(pic,classifier):
    pred = classifier(pic, True)
    return pred

def discrimination(pic,classifier):
    pred = classifier(pic, False)
    return pred

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def train(epoch, loader, model, classifier, optimizer, scheduler, device, iteration, tparameter):

    pretrain = tparameter['PRETRAIN']
    pool_num = tparameter['POOL']
    embed_dim = tparameter['EMBED_DIM']
    pos_dim = tparameter['POS_DIM']

    if dist.is_primary():
        loader = tqdm(loader)

    criterion = nn.MSELoss()
    class_criterion = nn.CrossEntropyLoss()
    gan_criterion = torch.nn.L1Loss()
    #feature_criterion = torch.nn.L1Loss()
    feature_criterion = nn.MSELoss()
    recons_weight = 100
    latent_loss_weight = 0.25
    class_loss_weight = 100 #0.25
    features_loss_weight = 50   #10
    discriminator_loss_weight = 1  #10
    GAN_loss_weight = 1
    sample_size = 25

    mse_sum = 0
    mse_n = 0


    if pretrain:
        PATH = tparameter['NETWORK']

        state_dict = torch.load(PATH)
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict

        list = []
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():

            name = k
            if k in list:
                continue
            # remove `module.`
            new_state_dict[name] = v
        # load params

        model.load_state_dict(new_state_dict)

        PATH = tparameter['CLASSIFIER']

        state_dict = torch.load(PATH)
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict

        list = []  
        new_state_dict = OrderedDict()

        for k, v in state_dict.items():

            name = k
            if k in list:
                continue
            # remove `module.`
            new_state_dict[name] = v
        # load params

        classifier.load_state_dict(new_state_dict)

    optimizer_D = torch.optim.Adam(classifier.parameters(), lr=0.0001)
    optimizer_G = torch.optim.Adam(model.parameters(), lr=0.0003)

    clean_positive_pool = ImagePool(pool_num)
    clean_negtive_pool = ImagePool(pool_num)
    hard_positive_pool = ImagePool(pool_num)
    hard_negtive_pool = ImagePool(pool_num)
    for i, (img, label) in enumerate(loader):
        iteration = iteration + 1
        model.zero_grad()

        img = img.to(device)
        label = label.to(device)

        dec_positive, dec_negtive, diff_positive, diff_negtive, id_positive, id_negtive, avg_features_positive, avg_features_negtive = model(img)

 
        recon_loss = 0
        latent_loss = 0
        class_loss = 0
        features_loss = 0
        loss_classification = 0
        loss_discriminator = 0

        fake_positive_loss = []
        fake_negtive_loss = []
        real_input_loss = []
        fake_positive_loss_to_true = []
        fake_negtive_loss_to_true = []


        new_classification1 = torch.zeros((img.shape)).to(device)
        new_classification_label1 = torch.zeros((label.shape)).to(device).long()
        new_classification2 = torch.zeros((img.shape)).to(device)
        new_classification_label2 = torch.zeros((label.shape)).to(device).long()

        generator_positive_labels = torch.zeros((label.shape)).to(device).long()
        generator_negtive_labels = torch.zeros((label.shape)).to(device).long()

        index_list = [*range(0,img.shape[0]*2,1)]
        random.shuffle(index_list)

        for pi in range(len(label)):
            if label[pi] == 1:
                fake_pic = clean_positive_pool.query(dec_positive[pi]).unsqueeze(0).detach()
                fake_drop = clean_negtive_pool.query(dec_negtive[pi]).unsqueeze(0).detach()

                index = index_list.pop(0)
                if index < img.shape[0]:
                    new_classification1[index] = fake_pic
                    new_classification_label1[index] = torch.tensor(3).cuda()
                else:
                    index = index - img.shape[0]
                    new_classification2[index] = fake_pic
                    new_classification_label2[index] = torch.tensor(3).cuda()

                index = index_list.pop(0)
                if index < img.shape[0]:
                    new_classification1[index] = img[pi]
                    new_classification_label1[index] = label[pi]
                else:
                    index = index - img.shape[0]
                    new_classification2[index] = img[pi]
                    new_classification_label2[index] = label[pi]


                generator_positive_labels[pi] = label[pi]
                generator_negtive_labels[pi] = torch.tensor(0)

                recon_loss = recon_loss + criterion(dec_positive[pi], img[pi])
                latent_loss = latent_loss + diff_positive[pi]
                features_loss = features_loss - feature_criterion(avg_features_positive[pi][(embed_dim - pos_dim):embed_dim], avg_features_negtive[pi][(embed_dim - pos_dim):embed_dim])

            elif label[pi] == 0:
                if np.random.randint(2,size=1) == 1:
                    fake_pic = clean_negtive_pool.query(dec_positive[pi]).unsqueeze(0).detach()
                    fake_drop = clean_negtive_pool.query(dec_negtive[pi]).unsqueeze(0).detach()
                else:
                    fake_drop = clean_negtive_pool.query(dec_positive[pi]).unsqueeze(0).detach()
                    fake_pic = clean_negtive_pool.query(dec_negtive[pi]).unsqueeze(0).detach()

                index = index_list.pop(0)
                if index < img.shape[0]:
                    new_classification1[index] = fake_pic
                    new_classification_label1[index] = torch.tensor(2).cuda()
                else:
                    index = index - img.shape[0]
                    new_classification2[index] = fake_pic
                    new_classification_label2[index] = torch.tensor(2).cuda()

                index = index_list.pop(0)
                if index < img.shape[0]:
                    new_classification1[index] = img[pi]
                    new_classification_label1[index] = label[pi]
                else:
                    index = index - img.shape[0]
                    new_classification2[index] = img[pi]
                    new_classification_label2[index] = label[pi]

                generator_positive_labels[pi] = label[pi]
                generator_negtive_labels[pi] = label[pi]


                recon_loss = recon_loss + criterion(dec_negtive[pi], img[pi])
                latent_loss = latent_loss + diff_negtive[pi]
                features_loss = features_loss + feature_criterion(avg_features_positive[pi][(embed_dim - pos_dim):embed_dim], avg_features_negtive[pi][(embed_dim - pos_dim):embed_dim])

            


        classification_prediction1 = classification(new_classification1, classifier)
        classification_prediction2 = classification(new_classification2, classifier)

        generator_prediction1 = discrimination(dec_positive, classifier)
        generator_prediction2 = discrimination(dec_negtive, classifier)

        loss_classification = class_criterion(classification_prediction1, new_classification_label1)*len(classification_prediction1) + class_criterion(classification_prediction2, new_classification_label2)*len(classification_prediction2)

        for dip in range(len(generator_prediction1)):
            if generator_positive_labels[dip] == 0:
                loss_discriminator = loss_discriminator + 3*class_criterion(generator_prediction1[dip].unsqueeze(0),generator_positive_labels[dip].unsqueeze(0))
            else:
                loss_discriminator = loss_discriminator + class_criterion(generator_prediction1[dip].unsqueeze(0),generator_positive_labels[dip].unsqueeze(0))

        loss_discriminator = loss_discriminator + class_criterion(generator_prediction2,generator_negtive_labels)*len(generator_negtive_labels)

        '''
        recon_loss = recon_loss / img.shape[0]
        latent_loss = latent_loss / img.shape[0]
        class_loss = class_loss / img.shape[0]
        '''
        '''
        if i * img.shape[0] * epoch % 1000 == 0:
            GAN_loss = loss_fake_G + loss_fake_D
        else:
            GAN_loss = loss_fake_G
        '''


        if iteration % 10 == 0:
            for param in classifier.parameters():
                param.requires_grad = True

            optimizer_D.zero_grad()
            loss_classification.backward()
            optimizer_D.step()

        for param in classifier.parameters():
            param.requires_grad = False

        if scheduler is not None:
            scheduler.step()
        optimizer_G.zero_grad()
        loss = recons_weight * recon_loss + latent_loss_weight * latent_loss  + features_loss_weight * features_loss + discriminator_loss_weight * loss_discriminator 
        loss.backward()
        optimizer_G.step()

        part_mse_sum = recon_loss.item() * img.shape[0]
        part_mse_n = img.shape[0]
        comm = {"mse_sum": part_mse_sum, "mse_n": part_mse_n}
        comm = dist.all_gather(comm)

        for part in comm:
            mse_sum += part["mse_sum"]
            mse_n += part["mse_n"]

        if dist.is_primary():
            lr = optimizer.param_groups[0]["lr"]
            loader.set_description(
                (
                    f"epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; "
                    f"latent: {latent_loss.item():.5f}; avg mse: {mse_sum / mse_n:.5f};"
                    f"features loss： {features_loss.item():.5f};"
                    f"class loss： {loss_classification/(2*len(classification_prediction1)):.5f}; discrim loss: {loss_discriminator/(2*2*len(generator_prediction1)):.5f}; "
                    f"Pos max： {id_positive.max():.5f}; Neg max: {id_negtive.max():.5f};"
                    f"lr: {lr:.5f}"
                )
            )
    return iteration

def main():

    with open('configs/test.yaml', 'r') as f:
        temp = yaml.full_load(f.read())

    training_root = temp['DATASETS']['TRAIN']
    num_workers = int(temp['SOLVER']['NUM_WORKERS'])
    batch = int(temp['SOLVER']['BATCH'])
    epoch = int(temp['SOLVER']['EPOCH'])
    n_gpu = int(temp['SOLVER']['GPU'])
    lr = float(temp['SOLVER']['LR'])

    tparameter = temp['TPARAMETER']
    checkpoint_root = temp['OUTPUT']['CHECKPOINT']

    device = "cuda"

    if not os.path.exists(checkpoint_root):
        os.makedirs(checkpoint_root)

    distributed = dist.get_world_size() > 1

    dataset = ImageFolder(training_root,transform = transforms.ToTensor())

    sampler = dist.data_sampler(dataset, shuffle=True, distributed=distributed)
    loader = DataLoader(
        dataset, batch_size = batch // n_gpu, sampler=sampler, num_workers=num_workers
    )

    model = VQVAE().to(device)
    classifier = Classifier().to(device)

    if distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    if 0:
        scheduler = CycleScheduler(
            optimizer,
            lr,
            n_iter=len(loader) * epoch,
            momentum=None,
            warmup_proportion=0.05,
        )
    iteration = 0
    for i in range(epoch):
        iteration = train(i, loader, model, classifier, optimizer, scheduler, device, iteration, tparameter)
        epoch = i + 1

        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_root, f"vqvae_{str(i + 1).zfill(5)}.pt"))
            torch.save(classifier.state_dict(),os.path.join(checkpoint_root, f"classifier_{str(i + 1).zfill(5)}.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    with open('configs/test.yaml', 'r') as f:
        temp = yaml.full_load(f.read())

    n_gpu = int(temp['SOLVER']['GPU'])

    port = (
        2 ** 15
        + 2 ** 14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )
    dist_url =f"tcp://127.0.0.1:{port}"


    args = parser.parse_args()

    print(temp)

    dist.launch(main, n_gpu, 1, 0, dist_url)
