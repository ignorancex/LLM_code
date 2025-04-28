"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer_offline import Visualizer
from metrics.mse_psnr_ssim_vif import azimuthalAverage
from util.util import save_ordered_dict_as_csv
from tqdm import tqdm
import numpy as np
import json

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

if __name__ == '__main__':
    opt = TrainOptions().parse()  # get training options
    opt.isTrain = False
    opt.epoch = 50
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup2(opt, "/Volumes/sandisk/cycleGAN_results/models_cycleGAN/")  # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    model.eval()

    # elcap_complete (50 patients)
    opt.text_file = "./data/ELCAP.csv"  # load the csv file containing test data info
    opt.dataset_mode = "LIDC_IDRI"
    elcap_complete = create_dataset(opt)
    print(len(elcap_complete))
    ####################################
    # Test 3 (8 patient from LIDC/IDRI)
    opt.text_file = "./data/LIDC_test.csv"  # load the csv file containing test data info
    opt.dataset_mode = "LIDC_IDRI"
    dataset_test_3 = create_dataset(opt)
    print(len(dataset_test_3))
    ####################################

    opt.dataset_len = len(elcap_complete)
    opt.test = 'elcap_complete'
    opt.dataset_mode = "LIDC_IDRI"
    print(f"Test: {opt.test}")
    for j, data_test in tqdm(enumerate(elcap_complete)):
        model.set_input(data_test)  # unpack data from data loader
        model.test(j)  # run inference

    model.save_raps(opt.epoch)

    ####################################
    opt.dataset_len = len(dataset_test_3)
    opt.dataset_mode = "LIDC_IDRI"
    opt.test = 'test_3'
    print(f"Test: {opt.test}")
    for j, data_test in tqdm(enumerate(dataset_test_3)):
        model.set_input(data_test)  # unpack data from data loader
        model.test(j)  # run inference

    model.save_raps(opt.epoch)


