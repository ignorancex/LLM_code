from unet2 import *
from data import *
import time
import ipykernel
import os
import argparse
import warnings

import tensorflow as tf
GPU_NUM = -1
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_NUM)

data_gen_args = dict(rotation_range=0.2,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     horizontal_flip=True,
                     fill_mode='nearest')
data_gen_args_simple = dict()

def parse_args():
    parser = argparse.ArgumentParser(description="Run NeuMF.")
    parser.add_argument('--addr', nargs='?', default=r"data\\",help="data folder")
    parser.add_argument('--imageFolder', nargs='?', default='train_binary',help="image folder")
    parser.add_argument('--paddingFolder', nargs='?', default='label_padding', help="padding image folder")
    parser.add_argument('--model',nargs='?', default='unet' ,help="unet or unet++")
    parser.add_argument('--loss', nargs='?', default='binary_crossentropy', help="loss function:binary_crossentropy,weightedBCE or jointLoss")
    parser.add_argument('--dataAugmentation', nargs='?', default=False, help="use data augmentation or not")
    parser.add_argument('--iterations', nargs='?', default=5, help="number of iterations")
    parser.add_argument('--batchSize', nargs='?', default=8, help="batch size")
    parser.add_argument('--epochs', nargs='?', default=100, help="number of epochs")
    parser.add_argument('--lr', nargs='?', default=1e-4, help="learning rate")
    return parser.parse_args()

def splitArray(dic, iterations):
    string = ''
    for i in range(iterations):
        for j in range(len(dic[i])):
            string = string + str(dic[i][j]) + ' '
        string = string + '\n'
    return string

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    args = parse_args()
    iterations = args.iterations
    batch_size = args.batchSize
    step_per_epoch = 1
    epochs = args.epochs
    lr = 1e-4
    aug_train_dir = 'data/aug_train'
    aug_val_dir = 'data/aug_validation'
    curDate = time.strftime('%m%d')
    modelFolder_init = 'h5/' + curDate
    resFolder_init = 'res/' + curDate
    addr = args.addr
    or_fd = args.imageFolder
    pa_fd = args.paddingFolder
    generator = data_gen_args if args.dataAugmentation else data_gen_args_simple
    kernelSize = [64]

    if (not os.path.isdir(modelFolder_init)):
        os.makedirs(modelFolder_init)
    if (not os.path.isdir(resFolder_init)):
        os.makedirs(resFolder_init)
    for j in range(iterations):
        modelFolder = modelFolder_init + '/' + curDate + '_' + str(j)
        if (not os.path.isdir(modelFolder)):
            os.makedirs(modelFolder)

    for j in range(len(kernelSize)):
        for k in range(iterations):
            aug_train_dir, aug_val_dir, imgDir, maskDir = randomSplitDataset(k, iterations, 155,
                                                                             img_dir=addr + '/' + or_fd,
                                                                             mask_dir=addr + '/' + pa_fd,
                                                                             aug_train_dir=aug_train_dir,
                                                                             aug_val_dir=aug_val_dir)
            train_generator = trainGenerator(batch_size, aug_train_dir, imgDir, maskDir, generator)
            test_generator = trainGenerator(batch_size, aug_val_dir, imgDir, maskDir, data_gen_args_simple)
            Loss = args.loss
            kernelNum = kernelSize[j]
            if args.model == 'unet':
                model = unet_2(k=kernelNum, loss=Loss, lr=lr)
            elif args.model == 'unet++':
                model = unet2_2(k=kernelNum, loss=Loss, lr=lr)
            else:
                print("No such model!")
                break
            # save results of each iterations
            modelName = 'unet' + '_' + args.model + '_' + str(kernelNum)
            lossName = Loss if isinstance(Loss, str) else str(Loss.__name__)
            modelFolder = modelFolder_init + '/' + curDate + '_' + str(k)
            modelPath = modelFolder + '/' + modelName + "_" + lossName + "_" + str(step_per_epoch) + "_" + str(
                epochs) + '.hdf5'
            model_checkpoint = ModelCheckpoint(modelPath, monitor='loss', verbose=1, save_best_only=True)
            # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1)
            history = model.fit_generator(data_gen(train_generator), steps_per_epoch=step_per_epoch, epochs=epochs,
                                          verbose=1,
                                          validation_data=data_gen(test_generator), validation_steps=1,
                                          callbacks=[model_checkpoint])
            print(history.history)
