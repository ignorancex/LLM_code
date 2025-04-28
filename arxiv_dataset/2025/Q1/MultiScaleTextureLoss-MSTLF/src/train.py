"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import time
from options.train_options import TrainOptions
from data import create_dataset
from loss_functions.early_stopping import EarlyStopping
from models import create_model
from util.visualizer_offline import Visualizer
from util.util import save_ordered_dict_as_csv
from tqdm import tqdm
# from efemarai import ef
import torch

torch.manual_seed(42)
torch.autograd.set_detect_anomaly(True)
if __name__ == '__main__':
    opt = TrainOptions().parse()  # get training options
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options (train dataset)
    dataset_size = len(dataset)  # get the number of images in the training dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup1(opt)  # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    total_iters = 0  # the total number of training iterations

    # Test 1 (1 patient from Mayo dataset)
    opt.text_file = "./data/mayo_test_1p.csv"  # load the csv file containing test data info
    opt.serial_batches = True
    opt.batch_size = 1
    dataset_test = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options (test dataset)
    print(len(dataset_test))

    # Test 2 (8 patient from Mayo dataset extended)
    opt.text_file = "./data/mayo_test_ext.csv"  # load the csv file containing test data info
    dataset_test_2 = create_dataset(opt)
    print(len(dataset_test_2))

    # Test 3 (8 patient from LIDC/IDRI)
    opt.text_file = "./data/LIDC_test.csv"  # load the csv file containing test data info
    dataset_test_3 = create_dataset(opt)
    print(len(dataset_test_3))

    opt.batch_size = 2
    for epoch in range(opt.epoch_count,
                       opt.n_epochs + opt.n_epochs_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()  # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        for i, data in enumerate(dataset):  # inner loop within one epoch

                iter_start_time = time.time()  # timer for computation per iteration

                total_iters += opt.batch_size
                epoch_iter += opt.batch_size
                model.set_input(data)  # unpack data from dataset and apply preprocessing
                model.optimize_parameters()  # calculate loss functions, get gradients, update network weights


        t_data = iter_start_time - iter_data_time
        current_losses = model.get_current_losses()
        t_comp = (time.time() - iter_start_time) / opt.batch_size
        visualizer.print_current_losses(epoch, epoch_iter, current_losses, t_comp, t_data)
        tracked_losses = model.track_current_losses()
        visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, current_losses, tracked_losses,
                                       'Loss_functions')

        print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
        save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
        model.save_networks(save_suffix)

        iter_data_time = time.time()

        # TEST
        test_time = 0
        if epoch == 1 or epoch == 50 or epoch == 200:
            test_start = time.time()
            opt.isTrain = False
            model.eval()
            # are needed.
            opt.test = 'test_1'
            for j, data_test in enumerate(dataset_test):
                model.set_input(data_test)  # unpack data from data loader
                model.test(j)  # run inference

            model.avg_performance()
            visualizer.plot_metrics(model.get_avg_test_metrics(), model.get_epoch_performance(), epoch,
                                    f'Average Metrics on {opt.test}')
            model.empty_dictionary()

            opt.test = 'test_2'
            for j, data_test in enumerate(dataset_test_2):
                model.set_input(data_test)  # unpack data from data loader
                model.test(j)  # run inference

            model.avg_performance()
            visualizer.plot_metrics(model.get_avg_test_metrics(), model.get_epoch_performance(), epoch,
                                    f'Average Metrics on {opt.test}')
            model.empty_dictionary()

            opt.test = 'test_3'
            opt.dataset_mode = "LIDC_IDRI"
            for j, data_test in enumerate(dataset_test_3):
                model.set_input(data_test)  # unpack data from data loader
                model.test(j)  # run inference

            model.avg_performance()
            visualizer.plot_metrics(model.get_avg_test_metrics(), model.get_epoch_performance(), epoch,
                                    f'Average Metrics on {opt.test}')
            model.empty_dictionary()

            opt.isTrain = True
            model.train()
            test_end = time.time()
            test_time = test_end - test_start
            ##########

        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)
            # model.save_texture_indexes()
            if epoch == 1 or epoch == 20 or epoch == 50 or epoch == 100 or epoch == 200:
                model.save_attention_maps()
                model.save_attention_weights()

        model.update_learning_rate()  # update learning rates at the end beginning of every epoch.
        print('End of epoch %d / %d \t Time Taken: %d sec' % (
            epoch, opt.n_epochs + opt.n_epochs_decay, (time.time() - epoch_start_time) - test_time))
        visualizer.print_time(epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time)