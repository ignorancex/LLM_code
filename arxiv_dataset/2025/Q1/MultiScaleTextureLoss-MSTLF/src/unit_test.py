import datetime
import time
from data import create_dataset
import torch
from options.train_unit_options import TrainOptions
from models.unit_model import UNITModel
from loss_functions.early_stopping import EarlyStopping
from tqdm import tqdm 
import random

cuda = True if torch.cuda.is_available() else False

if __name__ == '__main__':

    # ------------------------------
    # Train dataset
    opt = TrainOptions().parse()  # get training options
    opt.isTrain = False
    opt.epoch = 50
    model = UNITModel(opt)
    model.setup2(opt, "/Volumes/Untitled/models_UNIT/")
    model.eval()
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
    opt.text_file = "./data/elcap_data.csv"  # load the csv file containing test data info
    opt.dataset_mode = "LIDC_IDRI"
    elcap_complete = create_dataset(opt)
    print(len(elcap_complete))
    # ------------------------------
    # Test 3 (8 patient from LIDC/IDRI)
    opt.text_file = "./data/LIDC_test.csv"  # load the csv file containing test data info
    opt.dataset_mode = "LIDC_IDRI"
    dataset_test_3 = create_dataset(opt)
    print(len(dataset_test_3))

    ####################################
    """opt.dataset_len = len(dataset_test_2)
    opt.test = 'test_2'
    opt.dataset_mode = "mayo"
    print(f"Test: {opt.test}")
    random.seed(42)
    list_index = [102]  # random.sample(range(2996), 20)
    for j, data_test in tqdm(enumerate(dataset_test_2)):
        model.set_input(data_test)  # unpack data from data loader
        model.test_visuals(j, list_index)  # run inference"""
    ####################################
    opt.dataset_len = len(dataset_test_3)
    opt.test = 'test_3'
    opt.dataset_mode = "LIDC_IDRI"
    random.seed(42)
    list_index = [1516]  # random.sample(range(1831), 20)
    print(f"Test: {opt.test}")
    for j, data_test in tqdm(enumerate(dataset_test_3)):
        model.set_input(data_test)  # unpack data from data loader
        model.test_visuals(j, list_index)  # run inference
    ####################################
    opt.dataset_len = len(elcap_complete)
    opt.test = 'elcap_complete'
    opt.dataset_mode = "LIDC_IDRI"
    random.seed(42)
    list_index = [11087]  # random.sample(range(12360), 20)
    print(f"Test: {opt.test}")
    for j, data_test in tqdm(enumerate(elcap_complete)):
        model.set_input(data_test)  # unpack data from data loader
        model.test_visuals(j, list_index)  # run inference

    """opt.dataset_len = len(dataset_test)
    opt.test = 'test_1'
    opt.dataset_mode = "mayo"
    print(f"Test: {opt.test}")
    for j, data_test in tqdm(enumerate(dataset_test)):
        model.set_input(data_test)  # unpack data from data loader
        model.test()  # run inference

    model.save_metrics_per_patient(opt.epoch)"""
    ####################################
    """opt.dataset_len = len(dataset_test_2)
    opt.dataset_mode = "mayo"
    opt.test = 'test_2'
    print(f"Test: {opt.test}")
    for j, data_test in tqdm(enumerate(dataset_test_2)):
        model.set_input(data_test)  # unpack data from data loader
        model.test_3()  # run inference
    
    model.fid_compute()
    model.save_metrics_per_patient(opt.epoch)"""
    ####################################
    # opt.dataset_len = len(elcap_complete)
    # opt.test = 'elcap_complete'
    # opt.dataset_mode = "LIDC_IDRI"
    # print(f"Test: {opt.test}")
    # for j, data_test in tqdm(enumerate(elcap_complete)):
    #    model.set_input(data_test)  # unpack data from data loader
    #    model.test()  # run inference

    # model.save_noise_metrics(opt.epoch)   
    # model.save_raps_per_patient(opt.epoch)
    # model.save_metrics_per_patient(opt.epoch)

    ####################################
    # opt.dataset_len = len(dataset_test_3)
    # opt.dataset_mode = "LIDC_IDRI"
    # opt.test = 'test_3'
    # print(f"Test: {opt.test}")
    # for j, data_test in tqdm(enumerate(dataset_test_3)):
    #    model.set_input(data_test)  # unpack data from data loader
    #    model.test()  # run inference

    # model.save_noise_metrics(opt.epoch)
    # model.save_raps_per_patient(opt.epoch)
    # model.save_metrics_per_patient(opt.epoch)
    
      
