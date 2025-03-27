import argparse
import sys
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils

from tqdm import tqdm

from vqvae_codebooks import VQVAE,Classifier
from scheduler import CycleScheduler
import distributed as dist
from ImageLoader import Loader
import glob
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score, precision_score, log_loss
import torch.nn.functional as nnf
from skimage.transform import resize
import cv2
import yaml

def score(pred, target):


    m1 = pred.data.view(1, -1).cpu().numpy() # Flatten
    m2 = target.data.view(1, -1).cpu().numpy() # Flatten

    if m2.sum() == 0:
        if m1.sum() == 0:
            recall = 1.
            precision = 1.
            dice = 1.
            bce = log_loss(m2,m1)
        else:
            recall = 0.
            precision = 0.
            dice = 0.
            bce = log_loss(m2,m1)
    else:
        intersection = (m1 * m2).sum()
        tp = (m1 * m2).sum()
        tn = ((1-m1) * (1-m2)).sum()
        fp = ((1-m2)*m1).sum()
        fn = ((m2) * (1-m1)).sum()

        if m1.sum()== 0:
            recall = 0.
            precision = 0.
            another = 0.
        else:
            recall = tp / (tp + fp)
            precision = tp / (tp + fn)
            another = 2 * recall * precision / (precision + recall)

        dice = (2. * intersection) / (m1.sum() + m2.sum())

        bce = log_loss(m2,m1)

    return recall, precision, dice, bce

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def testing(epoch, loader_validation, loader_testing, model,classifier, optimizer, scheduler, device, test_output,tparameter, checkpoint_root, inputparameter):
    if dist.is_primary():
        loader_validation = tqdm(loader_validation)
        loader_testing = tqdm(loader_testing)
    criterion = nn.MSELoss()
    class_criterion = nn.CrossEntropyLoss()
    gan_criterion = torch.nn.L1Loss()
    #feature_criterion = torch.nn.L1Loss()
    feature_criterion = nn.MSELoss()
    recons_weight = 10
    latent_loss_weight = 0.25
    class_loss_weight = 2 #0.25
    features_loss_weight = 1
    GAN_loss_weight = 1
    sample_size = 25

    mse_sum = 0
    mse_n = 0

    best_valid_acc = 0
    best_valid_dice = 0

    for ki in range(1,10):
        see_trainingset = tparameter['SELFTEST']
        morphology = tparameter['MORPHOLOGY']
        n_embed = tparameter['N_EMBED']
        pos_embed = tparameter['POS_EMBED']
        ki = ki*10
        PATH = os.path.join(checkpoint_root,'vqvae_%s.pt' % str((ki)).zfill(5))

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

        PATH = os.path.join(checkpoint_root, 'classifier_%s.pt' % str((ki)).zfill(5))

        
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

        with torch.no_grad():
            valid_acc = 0
            valid_dice = 0
            num = 0
            pos = 0
            model.eval()
            if see_trainingset:
                trainingset = '/Data/Proteins2021/CaCL/training_best'
                all_labels = glob.glob(os.path.join(trainingset, '*'))
                for l in range(len(all_labels)):
                    all_images = glob.glob(os.path.join(all_labels[l], '*'))
                    for k in range(len(all_images[:100])):
                        if not '_mask.png' in all_images[k]:
                            num = num + 1
                            image_label = l
                            #mask = Image.open(all_images[k].replace('.png','_mask.png'))
                            image_name = os.path.basename(all_images[k])
                            sample = Image.open(all_images[k])
                            width, height = sample.size
                            crop_size = inputparameter['CROP_SIZE']
                            resize = inputparameter['SIZE']

                            #center_point

                            left = (width - crop_size) / 2
                            top = (height - crop_size) / 2
                            right = (width + crop_size) / 2
                            bottom = (height + crop_size) / 2

                            # Crop the center of the image
                            sample = sample.crop((left, top, right, bottom)).resize((resize,resize))

                            totensor = transforms.ToTensor()
                            sample = totensor(sample).to(device).unsqueeze(0)
                            dec_positive, dec_negtive, diff_positive, diff_negtive, id_positive, id_negtive, avg_features_positive, avg_features_negtive = model(sample)
                            mask = torch.zeros((sample.shape))
                            recon_loss = 0
                            latent_loss = 0
                            class_loss = 0
                            # features_loss = features_l1
                            positive_loss = []
                            negtive_loss = []
                            input_loss = []

                            recon_loss_positive = criterion(dec_positive, sample)
                            recon_loss_negative = criterion(dec_negtive, sample)
                            label = torch.tensor(l).cuda()

                            class_positive = classifier(dec_positive,False)
                            class_negtive = classifier(dec_negtive,False)
                            class_input = classifier(sample,False)

                            _, predict_positive = torch.max(class_positive, dim=1)
                            _, predict_negtive = torch.max(class_negtive, dim=1)
                            _, predict_input = torch.max(class_input, dim=1)

                            pred = predict_input.clone()

                            pred[pred == 2] = 0
                            pred[pred == 3] = 1

                            correct_tensor = pred.eq(label.data.view_as(pred))
                            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
                            valid_acc += accuracy.item()

                            

                            id_positive = torch.cat([id_positive, id_positive, id_positive], 0).permute([1, 2, 0])
                            id_negtive = torch.cat([id_negtive, id_negtive, id_negtive], 0).permute([1, 2, 0])

                            output_positive = id_positive
                            output_negtive = id_negtive

                            output_positive[output_positive >= (n_embed - pos_embed)] = 255
                            output_positive[output_positive < 255] = 0

                         
                            output_binary = output_positive.float()
                            output_binary[output_binary > 0] = 1.
                            output_binary = output_binary.unsqueeze(0).permute([0,3,1,2])


                            output_binary = nnf.interpolate(output_binary, size=(resize, resize), mode='nearest')

                            if morphology:
                                new_output = output_binary.permute([0, 2, 3, 1])
                                kernel = np.ones((5, 5), np.uint8)
                                kkk = new_output.cpu().numpy()
                                output_binary = torch.zeros((new_output.shape))

                                for i in range(len(kkk)):
                                    output_binary[i] = torch.tensor(cv2.dilate(kkk[i], kernel, iterations=1))

                                output_binary = output_binary.permute([0, 3, 1, 2]).cuda()

                            recall, precision, f1, bce = score(output_binary, mask)

                            sample = Image.fromarray((sample[0].permute([1, 2, 0]).cpu().numpy() * 255).astype(np.uint8))
                            sample_positive = Image.fromarray(
                                (dec_positive[0].permute([1, 2, 0]).cpu().numpy() * 255).astype(np.uint8))
                            sample_negative = Image.fromarray(
                                (dec_negtive[0].permute([1, 2, 0]).cpu().numpy() * 255).astype(np.uint8))
                            sample_binary = Image.fromarray(
                                (output_binary[0].permute([1, 2, 0]).cpu().numpy() * 255).astype(np.uint8))

                            output_folder = test_output + '/' + str(ki) + '/' + str(
                                l) + '/'

                            if not os.path.exists(output_folder):
                                os.makedirs(output_folder)
                            output_root = output_folder + '/' + image_name.replace('.jpg.jpg',
                                                                                   '') + '_' + str(
                                predict_positive.item()) + str(predict_negtive.item()) + str(predict_input.item())+ '_' + str(f1) + '.png'
                            a = get_concat_h(sample, sample_binary)
                            a = get_concat_h(a, sample_positive)
                            get_concat_h(a, sample_negative).save(output_root)

                #print(str(ki))
                #print(str(valid_acc/num))


            else:
                for i, (img, mask, label) in enumerate(loader_validation):
                    num += img.size(0)
                    img = img.to(device)
                    mask = mask.to(device)
                    label = label.to(device)

                    dec_positive, dec_negtive, diff_positive, diff_negtive, id_positive, id_negtive, avg_features_positive, avg_features_negtive = model(img)


                    class_positive = classifier(dec_positive, False)
                    class_negtive = classifier(dec_negtive, False)
                    class_input = classifier(img, False)

                    _, predict_positive = torch.max(class_positive, dim=1)
                    _, predict_negtive = torch.max(class_negtive, dim=1)
                    _, pred= torch.max(class_input, dim=1)

                    pred[pred == 2] = 0
                    pred[pred == 3] = 1


                    correct_tensor = pred.eq(label.data.view_as(pred))
                    accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
                    valid_acc += accuracy.item() * img.size(0)

                    id_positive = torch.cat([id_positive.unsqueeze(1), id_positive.unsqueeze(1), id_positive.unsqueeze(1)], 1)
                    id_negtive = torch.cat([id_negtive.unsqueeze(1), id_negtive.unsqueeze(1), id_negtive.unsqueeze(1)], 1)

                    output_positive = id_positive
                    output_negtive = id_negtive

                    
                    output_positive[output_positive >= 27] = 255
                    output_positive[output_positive < 255] = 0

                    
                    output_binary = output_positive.float()
                    output_binary[output_binary > 0] = 1.

                    
                    output_binary = nnf.interpolate(output_binary,size=(128,128),mode = 'nearest')

                    if morphology:
                        new_output = output_binary.permute([0,2,3,1])
                        kernel = np.ones((5, 5), np.uint8)
                        kkk = new_output.cpu().numpy()
                        output_binary = torch.zeros((new_output.shape))

                        for i in range(len(kkk)):
                            
                            output_binary[i] = torch.tensor(cv2.dilate(kkk[i], kernel, iterations=1))

                        output_binary = output_binary.permute([0, 3, 1, 2]).cuda()
                    
                    recall, precision, f1, bce = score(output_binary, mask)

                    valid_dice += f1 * img.size(0)
                        


                valid_acc = valid_acc / num
                valid_dice = valid_dice / num
                
                print('Validation_Epoch: ' + str((ki)) + ' acc: ' + str(valid_acc) + ' dice: ' + str(valid_dice))
                

                if 1:
                    best_valid_dice = valid_dice

                    test_acc = 0
                    test_dice = 0
                    best_test_dice = 0
                    best_test_acc = 0
                    num = 0
                    pos = 0
                    for j, (img, mask, label) in enumerate(loader_testing):
                        num += img.size(0)
                        img = img.to(device)
                        mask = mask.to(device)
                        label = label.to(device)

                        dec_positive, dec_negtive, diff_positive, diff_negtive, id_positive, id_negtive, avg_features_positive, avg_features_negtive = model(
                            img)

                        class_positive = classifier(dec_positive, False)
                        class_negtive = classifier(dec_negtive, False)
                        class_input = classifier(img, False)

                        _, predict_positive = torch.max(class_positive, dim=1)
                        _, predict_negtive = torch.max(class_negtive, dim=1)
                        _, pred = torch.max(class_input, dim=1)

                        pred[pred == 2] = 0
                        pred[pred == 3] = 1


                        correct_tensor = pred.eq(label.data.view_as(pred))
                        accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))
                        test_acc += accuracy.item() * img.size(0)

                        id_positive = torch.cat([id_positive.unsqueeze(1), id_positive.unsqueeze(1), id_positive.unsqueeze(1)], 1)
                        id_negtive = torch.cat([id_negtive.unsqueeze(1), id_negtive.unsqueeze(1), id_negtive.unsqueeze(1)],1)

                        output_positive = id_positive
                        output_negtive = id_negtive


                        output_positive[output_positive >= 27] = 255
                        output_positive[output_positive < 255] = 0

                        
                        output_binary = output_positive.float()
                        output_binary[output_binary > 0] = 1.

                        output_binary = nnf.interpolate(output_binary, size=(128, 128), mode='nearest')

                        if morphology:
                            new_output = output_binary.permute([0, 2, 3, 1])
                            kernel = np.ones((3, 3), np.uint8)
                            kkk = new_output.cpu().numpy()
                            output_binary = torch.zeros((new_output.shape))

                            for i in range(len(kkk)):
                                
                                output_binary[i] = torch.tensor(cv2.dilate(kkk[i], kernel, iterations=1))

                            output_binary = output_binary.permute([0, 3, 1, 2]).cuda()

                        
                        recall, precision, f1, bce = score(output_binary, mask)
                        
                        test_dice += f1 * img.size(0)
                       

                        for pi in range(len(img)):
                            sample = Image.fromarray((img[pi].permute([1, 2, 0]).cpu().numpy() * 255).astype(np.uint8))
                            sample_positive = Image.fromarray((dec_positive[pi].permute([1, 2, 0]).cpu().numpy() * 255).astype(np.uint8))
                            sample_negative = Image.fromarray((dec_negtive[pi].permute([1, 2, 0]).cpu().numpy() * 255).astype(np.uint8))
                            sample_mask = Image.fromarray((mask[pi].permute([1, 2, 0]).cpu().numpy() * 255).astype(np.uint8))
                            sample_binary = Image.fromarray((output_binary[pi].permute([1, 2, 0]).cpu().numpy() * 255).astype(np.uint8))

                            sample_label = label[pi]
                            sample_prediction = pred[pi]

                            output_folder = 'binary_test/' + str((ki)) + '/' + str(
                                sample_label.item()) + '/'

                            if not os.path.exists(output_folder):
                                os.makedirs(output_folder)
                            output_root = output_folder + '/' + 'Epoch_' + str((ki)) + 'Batch_' + str(j) + 'Num_' + str(pi) + 'prediction_' + str(predict_positive[pi].item())+str(predict_negtive[pi].item())+str(pred[pi].item()) + '.png'
                            a = get_concat_h(sample, sample_mask)
                            a = get_concat_h(a, sample_positive)
                            a = get_concat_h(a, sample_negative)
                            get_concat_h(a, sample_binary).save(output_root)

                    test_acc = test_acc / num
                    test_dice = test_dice / num

                    print(
                        'Test_Epoch: ' + str((ki)) + ' acc: ' + str(test_acc) + ' dice: ' + str(test_dice))

                    if test_acc > best_test_acc:
                        #print('Epoch ' + str((ki + 1) * 5) + ' has the best classification accuracy: ' + str(test_acc))
                        best_test_acc = test_acc

                    if test_dice > best_test_dice:
                        #print('Epoch ' + str((ki + 1) * 5) + ' has the best segmentation dice: ' + str(test_dice))
                        best_test_dice = test_dice



def main():

    with open('configs/test.yaml', 'r') as f:
        temp = yaml.full_load(f.read())


    testing_root = temp['DATASETS']['TEST']
    test_output = temp['OUTPUT']['ROOT']
    epoch = int(temp['SOLVER']['EPOCH'])
    n_gpu = int(temp['SOLVER']['GPU'])
    lr = float(temp['SOLVER']['LR'])
    inputparameter = temp['INPUT']

    tparameter = temp['TPARAMETER']
    checkpoint_root = temp['OUTPUT']['CHECKPOINT']

    if not os.path.exists(test_output):
        os.makedirs(test_output)

    device = "cuda"

    distributed = dist.get_world_size() > 1

    validationset = Loader(testing_root)
    testingset = Loader(testing_root)


    sampler1 = dist.data_sampler(validationset, shuffle=True, distributed=distributed)
    sampler2 = dist.data_sampler(testingset, shuffle=True, distributed=distributed)
    loader_validation = DataLoader(
        validationset, batch_size=1 // n_gpu, sampler=sampler1, num_workers=0
    )
    loader_testing = DataLoader(
        testingset, batch_size=1 // n_gpu, sampler=sampler2, num_workers=0
    )

    model = VQVAE().to(device)
    classifier = Classifier().to(device)
    if distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

    optimizer = optim.Adam(model.parameters(), lr = lr)
    scheduler = None

    if 0:
        scheduler = CycleScheduler(
            optimizer,
            lr,
            n_iter=len(loader) * epoch,
            momentum=None,
            warmup_proportion=0.05,
        )

    for i in range(epoch):
        testing(i, loader_validation, loader_testing, model, classifier, optimizer, scheduler, device, test_output,tparameter, checkpoint_root, inputparameter)

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
    dist_url = f"tcp://127.0.0.1:{port}"

    args = parser.parse_args()

    print(temp)

    dist.launch(main, n_gpu, 1, 0, dist_url)
