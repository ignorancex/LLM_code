"""
Experiments
"""

import splitting as sp
import train
import torchvision
import torchvision.transforms as transforms
import torch


trainLoader, testLoader = train.cifar_loaders(train_batch_size=128, test_batch_size=400, augment=False)
#trainLoader, testLoader = train.mnist_loaders(train_batch_size=100, test_batch_size=1000)


train.train(trainLoader, testLoader,
            train.NESingleConvNet(sp.NEmonForwardStep,
                                 in_dim=32,
                        in_channels=3,
                        out_channels=81,
                        alpha=0.5,
                        max_iter=300,
                        tol=1e-2,
                        m=0),
        max_lr=1e-3,
        lr_mode='step',
        step=25,
        change_mo=False,
        epochs=40,
#        epochs=1,
        print_freq=100,
        tune_alpha=False,
        regularizer=0)

#train.train(trainLoader, testLoader,
#      train.NESingleFcNet(sp.NEmonForwardStep,
#                        in_dim=28**2,
#                        out_dim=100,
#                        alpha=1.0,
#                        max_iter=300,
#                        tol=1e-2,
#                        m=0.0),
#        max_lr=1e-3,
#        lr_mode='step',
#        step=10,
#        change_mo=False,
#        epochs=40,
#        epochs=10,
#        print_freq=100,
#        tune_alpha=False,
#        regularizer = 0)

#train.train(trainLoader, testLoader,
#            train.NESingleConvNet(sp.NEmonForwardStep,
#                                 in_dim=28,
#                        in_channels=1,
#                        out_channels=64,
#                        alpha=0.5,
#                        max_iter=300,
#                        tol=1e-2,
#                        m=0),
#        max_lr=1e-3,
#        lr_mode='step',
#        step=25,
#        change_mo=False,
#        epochs=40,
#        epochs=1,
#        print_freq=100,
#        tune_alpha=False,
#        regularizer=0)