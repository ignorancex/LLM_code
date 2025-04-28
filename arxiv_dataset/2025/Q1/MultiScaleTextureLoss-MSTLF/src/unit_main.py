import datetime
import time
from data import create_dataset
import torch
from options.train_unit_options import TrainOptions
from models.unit_model import UNITModel

cuda = True if torch.cuda.is_available() else False

if __name__ == '__main__':

    # ------------------------------
    # Train dataset
    opt = TrainOptions().parse()  # get training options
    opt.epoch = 0
    opt.text_file = "./data/mayo_training_9pat.csv"
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options (train dataset)
    dataset_size = len(dataset)  # get the number of images in the training dataset.
    print('The number of training images = %d' % dataset_size)
    # ------------------------------
    # Test 1 (1 patient from Mayo dataset)
    opt.text_file = "./data/mayo_test_1p.csv"  # load the csv file containing test data info
    opt.serial_batches = True
    opt.batch_size = 1
    dataset_test = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options (test dataset)
    print(len(dataset_test))
    # ------------------------------
    # Test 2 (8 patient from Mayo dataset extended)
    opt.text_file = "./data/mayo_test_ext.csv"  # load the csv file containing test data info
    dataset_test_2 = create_dataset(opt)
    print(len(dataset_test_2))
    # ------------------------------
    # elcap_complete (ELCAP dataset 50 patients)
    opt.text_file = "./data/ELCAP.csv"  # load the csv file containing test data info
    opt.dataset_mode = "LIDC_IDRI"
    elcap_complete = create_dataset(opt)
    print(len(elcap_complete))
    # ------------------------------
    # Test 3 (8 patient from LIDC/IDRI)
    opt.text_file = "./data/LIDC_test.csv"  # load the csv file containing test data info
    opt.dataset_mode = "LIDC_IDRI"
    dataset_test_3 = create_dataset(opt)
    print(len(dataset_test_3))
    # ------------------------------
    model = UNITModel(opt)

    # ----------
    #  Training
    # ----------
    opt.batch_size = 1
    opt.serial_batches = False
    prev_time = time.time()
    for epoch in range(opt.epoch, opt.n_epochs):
        opt.dataset_mode = "mayo"
        epoch_start_time = time.time()
        for i, batch in enumerate(dataset):

            # Set model input
            model.set_input(batch)
            model.optimize_parameters()

            opt.serial_batches = True
            opt.batch_size = 1
            test_start = time.time()
            opt.isTrain = False
            model.eval()
            # ---------------------------------
            opt.dataset_len = len(dataset_test)
            opt.test = 'test_1'
            print(f"Test: {opt.test}")
            for j, data_test in enumerate(dataset_test):
                model.set_input(data_test)  # unpack data from data loader
                model.test()  # run inference

            # --------------
            #  Log Progress
            # --------------
            # Determine approximate time left
            batches_done = epoch * len(dataset) + i
            batches_left = opt.n_epochs * len(dataset) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            model.print_current_loss(epoch, dataset_size, i, time_left)

        model.plot_current_losses(epoch, model.track_current_losses(), 'Loss_functions')
        if epoch == 50 or epoch == 100 or epoch == 200:
            model.save_networks(epoch)

        # ------
        #  Test
        # ------
        test_time = 0
        test_start = time.time()
        if epoch == 50 or epoch == 100 or epoch == 200:
            opt.serial_batches = True
            opt.batch_size = 1
            test_start = time.time()
            opt.isTrain = False
            model.eval()
            # ---------------------------------
            opt.dataset_len = len(dataset_test)
            opt.test = 'test_1'
            print(f"Test: {opt.test}")
            for j, data_test in enumerate(dataset_test):
                model.set_input(data_test)  # unpack data from data loader
                model.test()  # run inference

            model.compute_FrechetInceptionDistance()
            model.save_metrics(model.get_epoch_performance(), epoch)
            # ---------------------------------
            opt.dataset_len = len(dataset_test_2)
            opt.test = 'test_2'
            print(f"Test: {opt.test}")
            for j, data_test in enumerate(dataset_test_2):
                model.set_input(data_test)  # unpack data from data loader
                model.test()  # run inference

            model.compute_FrechetInceptionDistance()
            model.save_metrics(model.get_epoch_performance(), epoch)
            # ---------------------------------
            opt.dataset_len = len(elcap_complete)
            opt.test = 'elcap_complete'
            opt.dataset_mode = "LIDC_IDRI"
            print(f"Test: {opt.test}")
            for j, data_test in enumerate(elcap_complete):
                model.set_input(data_test)  # unpack data from data loader
                model.test()  # run inference

            model.compute_FrechetInceptionDistance()
            model.save_metrics(model.get_epoch_performance(), epoch)
            # ---------------------------------
            opt.dataset_len = len(dataset_test_3)
            opt.dataset_mode = "LIDC_IDRI"
            opt.test = 'test_3'
            print(f"Test: {opt.test}")
            for j, data_test in enumerate(dataset_test_3):
                model.set_input(data_test)  # unpack data from data loader
                model.test()  # run inference

            model.compute_FrechetInceptionDistance()
            model.save_metrics(model.get_epoch_performance(), epoch)

            opt.batch_size = 16
            opt.serial_batches = False
            model.train()

        test_end = time.time()
        test_time = test_end - test_start
        model.update_learning_rate()
        model.print_time(epoch, (time.time() - epoch_start_time) - test_time)
