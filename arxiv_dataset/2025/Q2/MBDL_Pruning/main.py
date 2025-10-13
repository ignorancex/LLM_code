# -----------------
# Importing from Python module
# -----------------
import argparse
import copy
import os
import json
import datetime
import fire
import h5py
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
# -----------------
# Importing from files
# -----------------
from torch_util.module import CNNBlock, EDSR, DeepUnfolding, DEQ, ResBlock, DnCNN, crop_images
from torch_util.module import single_ftran, single_fmult, mul_ftran, mul_fmult
from torch_util.metrics import Stack, Mean, compare_psnr, compare_ssim, compare_snr
from torch_util.common import write_pruning, to_tiff, check_and_mkdir
from dataset.pmri_fastmri_brain import RealMeasurement, uniformly_cartesian_mask, fmult, ftran
from dataset.modl import MoDLDataset, CombinedDataset
import Pruning_Finetuning as Pruning_Finetuning
from sota_module.method.baseline.e2evarnet import E2EVarNetWrapper, DatasetWrapper, VarNetWrapper
from Pruning_Finetuning import inputDataDict, absolute_helper, pruning_recon_module

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


def main():
    # -------------------------
    # Define all user inputs
    # -------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str)
    parser.add_argument('--task_config', type=str)
    args = parser.parse_args()
    gpu_index = args.gpu
    config_file = args.task_config
    
    with open(config_file) as File:
        config = json.load(File)
    
    purposeOfProgram = config['setting']['purpose']
        
    # -------------------------
    # Define computational source setup
    # -------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    from torch_util.module import cvpr2018_net as voxelmorph
    from torch_util.module import EDSR, DeepUnfolding, DEQ
    import Pruning_Finetuning as Pruning_Finetuning
    from dataset.modl import load_synthetic_MoDL_dataset
    mu_list = config['module']['recon']['mu_list']  # for RED
    gamma_list = config['module']['recon']['gamma_list']  # for pnp and RED
    alpha_list = config['module']['recon']['alpha_list']  # for pnp
    iteration_k = config['module']['recon']['iteration_k']

    # -------------------------
    # Define dataset
    # -------------------------
    generator = torch.Generator()
    generator.manual_seed(0)
    MoDLdataset = RealMeasurement(
        idx_list=range(1375),
        acceleration_rate=1,
        is_return_y_smps_hat=True,
        config = config)
    data_split_ratio = config['setting']['data_split_ratio']
    dataset_lengths = [int(len(MoDLdataset) * data_split_ratio[0]), int(len(MoDLdataset) * data_split_ratio[1]),
                       len(MoDLdataset) - int(len(MoDLdataset) * data_split_ratio[0]) - int(
                           len(MoDLdataset) * data_split_ratio[1])]

    sample_indices = []

    for count in range(int(len(MoDLdataset) * data_split_ratio[1])):
        if count % 10 == 0:
            sample_indices.append(count)

    _, _, example_dataset = torch.utils.data.random_split(MoDLdataset, dataset_lengths, generator=generator)

    sample_single_y_tran = []
    sample_mul_y_tran = []
    sample_single_y = []
    sample_mul_y = []
    sample_x = []
    sample_mask = []
    sample_sensitivity_map = []
    for sample in DataLoader(Subset(example_dataset, sample_indices), batch_size=1):
        sample_single_y_tran_instant, sample_mul_y_tran_instant, sample_single_y_instant, sample_mul_y_instant, sample_x_instant, sample_mask_instant, sample_sensitivity_map_instant = (
            i.cuda() for i in sample)
        sample_single_y_tran.append(sample_single_y_tran_instant)
        sample_mul_y_tran.append(sample_mul_y_tran_instant)
        sample_single_y.append(sample_single_y_instant)
        sample_mul_y.append(sample_mul_y_instant)
        sample_x.append(sample_x_instant)
        sample_mask.append(sample_mask_instant)
        sample_sensitivity_map.append(sample_sensitivity_map_instant)

    tiff_log_metrics = Stack()
    tiff_base_log_batch = {
        "Purpose": config['setting']['purpose'],
        "Mul_Coil": config['dataset']['multi_coil'],
    }
    excel_log_metrics = Stack()
    pruning_base_log_batch = {
        "Purpose": config['setting']['purpose'],
        "Mul_Coil": config['dataset']['multi_coil'],
        "Pruning_Iterative_Steps": config['pruning']['iterative_steps'],
        "Data_Fidelity": config['pruning']['data_fidelity']
    }
    tiff_path = config['setting']['root_path'] + config['setting']['tiff_path']
    check_and_mkdir(tiff_path)

    check_and_mkdir(tiff_path + 'ground_truth/')
    check_and_mkdir(tiff_path + 'under_sample/')

    for i in range(len(sample_indices)):
        ground_truth = sample_x[i].permute([0, 2, 3, 1]).contiguous().detach().cpu().numpy()
        ground_truth = np.sqrt(ground_truth[:, :, :, 0] ** 2 + ground_truth[:, :, :, 1] ** 2)
        (sample_mul_y_tran[i])[sample_x[i] == 0] = 0
        under_sample = sample_mul_y_tran[i].permute([0, 2, 3, 1]).contiguous().detach().cpu().numpy()
        under_sample = np.sqrt(under_sample[:, :, :, 0] ** 2 + under_sample[:, :, :, 1] ** 2)
        to_tiff(ground_truth, path=tiff_path + 'ground_truth/' + 'ground_truth' + "_" + str(i) + '.tiff',
                is_normalized=True)
        to_tiff(under_sample, path=tiff_path + 'under_sample/' + 'under_sample' + "_" + str(i) + '.tiff',
                is_normalized=True)

        tiff_base_log_batch = {**tiff_base_log_batch, 'Under_sample_ssim' + str(i): compare_ssim(
            absolute_helper(crop_images(sample_mul_y_tran[i])), absolute_helper(crop_images(sample_x[i]))).item()}
        tiff_base_log_batch = {**tiff_base_log_batch, 'Under_sample_psnr' + str(i): compare_psnr(
            absolute_helper(crop_images(sample_mul_y_tran[i])), absolute_helper(crop_images(sample_x[i]))).item()}

    pruning_module_name_list = config['pruning']['module']
    importance_list = config['pruning']['importance']
    sparsity_list = config['pruning']['ch_sparsity']
    
    
    print(f"\n-------------------------")
    print(f"Program purpose: {purposeOfProgram}")
    print(f"Fine-tune model: {config['pruning']['fine_tuning']}")
    print(f"Fine-tune loss type: {config['pruning']['fine_tune_loss_type']}")
    print(f"-------------------------\n")
    
    print(purposeOfProgram)
    for ii, pruning_module_name in enumerate(pruning_module_name_list):
        
        print(f"\n-------------------------\nPurning module: {pruning_module_name}\n-------------------------\n")

        generator = torch.Generator()
        generator.manual_seed(0)
        _, _, example_dataset = torch.utils.data.random_split(MoDLdataset, dataset_lengths, generator=generator)
        if pruning_module_name in ["E2EVARNET", "VARNET"]:
            example_dataset = DatasetWrapper(example_dataset)
        sample_single_y_tran = []
        sample_mul_y_tran = []
        sample_single_y = []
        sample_mul_y = []
        sample_x = []
        sample_mask = []
        sample_sensitivity_map = []
        for sample in DataLoader(Subset(example_dataset, sample_indices), batch_size=1):
            sample_single_y_tran_instant, sample_mul_y_tran_instant, sample_single_y_instant, sample_mul_y_instant, sample_x_instant, sample_mask_instant, sample_sensitivity_map_instant = \
                (i.cuda() for i in sample)
            sample_single_y_tran.append(sample_single_y_tran_instant)
            sample_mul_y_tran.append(sample_mul_y_tran_instant)
            sample_single_y.append(sample_single_y_instant)
            sample_mul_y.append(sample_mul_y_instant)
            sample_x.append(sample_x_instant)
            sample_mask.append(sample_mask_instant)
            sample_sensitivity_map.append(sample_sensitivity_map_instant)

        if config['dataset']['multi_coil'] == True:
            test_time_inputs = inputDataDict(sample_mul_y_tran[0], sample_mask[0], sample_sensitivity_map[0],
                                                sample_mul_y[0], module_name=pruning_module_name)
            test_inputs = []
            for i in range(len(sample_indices)):
                test_inputs.append(
                    inputDataDict(sample_mul_y_tran[i], sample_mask[i], sample_sensitivity_map[i], sample_mul_y[i],
                                    module_name=pruning_module_name))

        else:
            test_time_inputs = inputDataDict(sample_single_y_tran[0], sample_mask[0], sample_sensitivity_map[0],
                                                sample_single_y[0], module_name=pruning_module_name)
            test_inputs = []
            for i in range(len(sample_indices)):
                test_inputs.append(inputDataDict(sample_single_y_tran[i], sample_mask[i], sample_sensitivity_map[i],
                                                    sample_single_y[i], module_name=pruning_module_name))

        pruning_instant_log_batch = pruning_base_log_batch
        tiff_instant_log_batch = tiff_base_log_batch
        tiff_instant_log_batch = {**tiff_instant_log_batch, 'Module': pruning_module_name}
        pruning_instant_log_batch = {**pruning_instant_log_batch, 'Module': pruning_module_name}

        method_dict = {
            'Pruning_Finetuning': Pruning_Finetuning,
        }

        for jj, importance in enumerate(importance_list):
            pruning_iinstant_log_batch = pruning_instant_log_batch
            tiff_iinstant_log_batch = tiff_instant_log_batch
            
            # -------------------------
            # Define Model Architecture
            # -------------------------
            if pruning_module_name == "DEQ":
                pruning_module_architecture = DEQ(mu_list[0], gamma_list[0], alpha_list[0], config, pruning_module_name)

            elif pruning_module_name == "E2EVARNET":
                pruning_module_architecture = E2EVarNetWrapper(config)
                model_sum = sum(p.numel() for p in pruning_module_architecture.parameters() if p.requires_grad)

            elif pruning_module_name == "VARNET":
                pruning_module_architecture = VarNetWrapper(config)
                model_sum = sum(p.numel() for p in pruning_module_architecture.parameters() if p.requires_grad)

            print(f"\n-------------------------\nNetwork loading {config['pruning']['module_loading']} for saved model performance evaluation \n-------------------------\n")
            if config['pruning']['module_loading'] == False:
                model = pruning_module_architecture
            else:  # Load saved model
                if pruning_module_name == "DEQ":
                    pruning_module_architecture.load_state_dict(
                        torch.load(config['pruning']['module_path_deq'], map_location=torch.device(device)))

                elif pruning_module_name == "E2EVARNET":
                    pruning_module_architecture.load_state_dict(
                        torch.load(config['pruning']['module_path_e2evarnet'], map_location=torch.device(device)))

                elif pruning_module_name == "VARNET":
                    pruning_module_architecture.load_state_dict(torch.load(config['pruning']['module_path_varnet'], map_location=torch.device(device)))

                pruning_module_architecture.cuda()
                model = pruning_module_architecture

            # -------------------------
            # Measuring test time on unpruned netowrk
            # -------------------------
            saved_starter, saved_ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            if pruning_module_name == "DEQ":
                saved_repetitions = 30
            else:
                saved_repetitions = 100
            saved_timings = np.zeros((saved_repetitions, 1))
            for _ in range(10):
                _ = pruning_module_architecture(test_time_inputs)
            with torch.no_grad():
                for rep in range(saved_repetitions):
                    saved_starter.record()
                    _ = pruning_module_architecture(test_time_inputs)
                    saved_ender.record()
                    torch.cuda.synchronize()
                    saved_curr_time = saved_starter.elapsed_time(saved_ender)
                    saved_timings[rep] = saved_curr_time
                    
                    
            # -------------------------
            # Test saved model
            # -------------------------
            print(f"\n-------------------------\nTest Saved Model Prediction\n-------------------------\n")
            zero_ssim, zero_psnr, unpruned_ssim, unpruned_psnr = method_dict[
                config['setting']['method']].pruning_test(
                dataset=MoDLdataset,
                recon_module=pruning_module_architecture,
                regis_module=None,
                config=config, recon_module_type=pruning_module_name
            )
            # -------------------------
            # Save result of pretrained model
            # -------------------------
            saved_model_path = tiff_path + pruning_module_name + '/' + importance + '/saved_model/'
            check_and_mkdir(saved_model_path)

            for i in range(len(sample_indices)):
                if pruning_module_name == "E2EVARNET":
                    [saved_recon, _] = pruning_module_architecture(test_inputs[i])
                else:
                    saved_recon = pruning_module_architecture(test_inputs[i])

                saved_recon = saved_recon.detach()
                saved_recon[sample_x[i] == 0] = 0

                saved_recon_tiff = saved_recon.permute([0, 2, 3, 1]).contiguous().detach().cpu().numpy()
                saved_recon_tiff = np.sqrt(saved_recon_tiff[:, :, :, 0] ** 2 + saved_recon_tiff[:, :, :, 1] ** 2)
                to_tiff(saved_recon_tiff, path=saved_model_path + "saved_model_" + str(i) + '.tiff', is_normalized=True)

                tiff_iinstant_log_batch = {**tiff_iinstant_log_batch, 'saved_ssim' + str(i): compare_ssim(
                    absolute_helper(crop_images(saved_recon)), absolute_helper(crop_images(sample_x[i]))).item()}
                tiff_iinstant_log_batch = {**tiff_iinstant_log_batch, 'saved_psnr' + str(i): compare_psnr(
                    absolute_helper(crop_images(saved_recon)), absolute_helper(crop_images(sample_x[i]))).item()}

            tiff_iinstant_log_batch = {**tiff_iinstant_log_batch, 'Pruning_Criteria': importance}
            tiff_iinstant_log_batch = {**tiff_iinstant_log_batch, 'Saved_Model(Test Time)': np.sum(saved_timings) / saved_repetitions}

            pruning_iinstant_log_batch = {**pruning_iinstant_log_batch, 'Pruning_Criteria': importance}
            pruning_iinstant_log_batch = {**pruning_iinstant_log_batch, 'Zero_Filled(SSIM)': zero_ssim}
            pruning_iinstant_log_batch = {**pruning_iinstant_log_batch, 'Zero_Filled(PSNR)': zero_psnr}
            pruning_iinstant_log_batch = {**pruning_iinstant_log_batch, 'Saved_Model(Test Time)': np.sum(saved_timings) / saved_repetitions}
            pruning_iinstant_log_batch = {**pruning_iinstant_log_batch, 'Unpruned(SSIM)': unpruned_ssim}
            pruning_iinstant_log_batch = {**pruning_iinstant_log_batch, 'Unpruned(PSNR)': unpruned_psnr}

            for kk, sparsity in enumerate(sparsity_list):
                
                pruning_iiinstant_log_batch = pruning_iinstant_log_batch
                tiff_iiinstant_log_batch = tiff_iinstant_log_batch

                # -------------------------
                # Define Model Architecture
                # -------------------------
                if pruning_module_name == "DEQ":
                    pruning_module_architecture = DEQ(mu_list[0], gamma_list[0], alpha_list[0], config, pruning_module_name)
                elif pruning_module_name == "E2EVARNET":
                    pruning_module_architecture = E2EVarNetWrapper(config)
                elif pruning_module_name == "VARNET":
                    pruning_module_architecture = VarNetWrapper(config)
                else:
                    raise ValueError(f"Check the pruning_module_name: {pruning_module_name}")

                print(f"\n-------------------------\nNetwork loading {config['pruning']['module_loading']} for one-shot pruned model performance evaluation \n-------------------------\n")
                if config['pruning']['module_loading'] == False:
                    model = pruning_module_architecture
                else:  # Load saved model
                    if pruning_module_name == "DEQ":
                        pruning_module_architecture.load_state_dict(
                            torch.load(config['pruning']['module_path_deq'], map_location=torch.device(device)))
                        if config['pruning']['fine_tune_loss_type'] == "testdc":
                            pass
                        else:
                            pruning_module_architecture.setTol(tolValue=4e-3)
                    elif pruning_module_name == "E2EVARNET":
                        pruning_module_architecture.load_state_dict(
                            torch.load(config['pruning']['module_path_e2evarnet'], map_location=torch.device(device)))
                    elif pruning_module_name == "VARNET":
                        pruning_module_architecture.load_state_dict(
                            torch.load(config['pruning']['module_path_varnet'], map_location=torch.device(device)))
                    else:
                        raise ValueError(f"Check the pruning_module_name: {pruning_module_name}")

                    pruning_module_architecture.cuda()

                # -------------------------
                # Prune the Model
                # -------------------------
                print(f"\n-------------------------\nPrune {pruning_module_name}\n-------------------------\n")
                one_shot_pruned_model, unpruned_model, initial_number_of_parameters, final_number_of_parameters = pruning_recon_module(
                    pruning_module_architecture, importance, dataset=MoDLdataset, sparsity=sparsity,
                    module_name=pruning_module_name, fine_tuning=False, config = config)

                if config['pruning']['fine_tuning'] == False:
                    # -------------------------
                    # One-shot Pruning Test (before fine-tuning)
                    # -------------------------
                    print(f"\n-------------------------\nTest One-shot Prediction\n-------------------------\n")
                    zero_ssim, zero_psnr, one_shot_pruned_ssim, one_shot_pruned_psnr = method_dict[
                        config['setting']['method']].pruning_test(
                        dataset=MoDLdataset,
                        recon_module=one_shot_pruned_model,
                        regis_module=None,
                        config=config, recon_module_type=pruning_module_name
                    )
                else:
                    # -------------------------
                    # One-shot Pruning Test (before fine-tuning)
                    # -------------------------
                    print(f"\n-------------------------\nTest One-shot Prediction\n-------------------------\n")
                    zero_ssim, zero_psnr, one_shot_pruned_ssim, one_shot_pruned_psnr = method_dict[
                        config['setting']['method']].pruning_test(
                        dataset=MoDLdataset,
                        recon_module=one_shot_pruned_model,
                        regis_module=None,
                        config=config, recon_module_type=pruning_module_name
                    )
                    
                    # -------------------------
                    # Save result of one-shot model
                    # -------------------------
                    one_shot_path = tiff_path + pruning_module_name + '/' + importance + '/' + str(sparsity) + '/one_shot_model/'
                    check_and_mkdir(one_shot_path)
                    for i in range(len(sample_indices)):
                        if pruning_module_name == "E2EVARNET":
                            [saved_recon, _] = one_shot_pruned_model(test_inputs[i])
                            saved_recon = saved_recon.detach()
                        elif pruning_module_name == "DEQ":
                            saved_recon = one_shot_pruned_model(test_inputs[i])
                            saved_recon = saved_recon.detach()
                        else:
                            saved_recon = one_shot_pruned_model(test_inputs[i])
                            saved_recon = saved_recon.detach()
                        saved_recon[sample_x[i] == 0] = 0
                        saved_recon_tiff = saved_recon.permute([0, 2, 3, 1]).contiguous().detach().cpu().numpy()
                        saved_recon_tiff = np.sqrt(
                            saved_recon_tiff[:, :, :, 0] ** 2 + saved_recon_tiff[:, :, :, 1] ** 2)
                        to_tiff(saved_recon_tiff,
                                path=one_shot_path + "one_shot_model_" + str(i) + '.tiff', is_normalized=True)
                        tiff_iiinstant_log_batch = {**tiff_iiinstant_log_batch,
                                                    'one_shot_ssim' + str(i): compare_ssim(
                                                        absolute_helper(crop_images(saved_recon)),
                                                        absolute_helper(
                                                            crop_images(sample_x[i]))).item()}
                        tiff_iiinstant_log_batch = {**tiff_iiinstant_log_batch,
                                                    'one_shot_psnr' + str(i): compare_psnr(
                                                        absolute_helper(crop_images(saved_recon)),
                                                        absolute_helper(crop_images(sample_x[i]))).item()}

                    print(f"\n-------------------------\nStart Fine Tuning\n-------------------------\n")
                    
                    # -------------------------
                    # Fine-tuning
                    # -------------------------
                    fine_tuned_pruned_model, unpruned_model, initial_number_of_parameters, final_number_of_parameters = pruning_recon_module(
                        unpruned_model, importance, dataset=MoDLdataset, sparsity=sparsity,
                        module_name=pruning_module_name, fine_tuning=True, config = config)

                    print(f"\n-------------------------\nEnd Fine Tuning and Test the Result\n-------------------------\n")

                    # -------------------------
                    # Test fined model
                    # -------------------------
                    zero_ssim, zero_psnr, fine_tuned_pruned_ssim, fine_tuned_pruned_psnr = method_dict[
                        config['setting']['method']].pruning_test(
                        dataset=MoDLdataset,
                        recon_module=fine_tuned_pruned_model,
                        regis_module=None,
                        config=config, recon_module_type=pruning_module_name
                    )

                    # -------------------------
                    # Save result of fine-tuned model
                    # -------------------------
                    fine_tuned_path = tiff_path + pruning_module_name + '/' + importance + '/' + str(
                        sparsity) + '/fine_tuned_model/'
                    check_and_mkdir(fine_tuned_path)
                    for i in range(len(sample_indices)):
                        if pruning_module_name == "E2EVARNET":
                            [saved_recon, _] = fine_tuned_pruned_model(test_inputs[i])
                        else:
                            saved_recon = fine_tuned_pruned_model(test_inputs[i])
                        # saved_recon = fine_tuned_pruned_model(test_inputs[i])
                        saved_recon[sample_x[i] == 0] = 0
                        saved_recon_tiff = saved_recon.permute([0, 2, 3, 1]).contiguous().detach().cpu().numpy()
                        saved_recon_tiff = np.sqrt(
                            saved_recon_tiff[:, :, :, 0] ** 2 + saved_recon_tiff[:, :, :, 1] ** 2)
                        to_tiff(saved_recon_tiff,
                                path=fine_tuned_path + "fine_tuned_model_" + str(
                                    i) + '.tiff', is_normalized=True)

                        tiff_iiinstant_log_batch = {**tiff_iiinstant_log_batch,
                                                    'fine_tuned_ssim' + str(i): compare_ssim(
                                                        absolute_helper(crop_images(saved_recon)),
                                                        absolute_helper(
                                                            crop_images(sample_x[i]))).item()}
                        tiff_iiinstant_log_batch = {**tiff_iiinstant_log_batch,
                                                    'fine_tuned_psnr' + str(i): compare_psnr(
                                                        absolute_helper(crop_images(saved_recon)),
                                                        absolute_helper(crop_images(sample_x[i]))).item()}

                # -------------------------
                # Measuring test time on pruned netowrk
                # -------------------------
                fine_starter, fine_ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True)
                if pruning_module_name == "DEQ":
                    fine_repetitions = 30
                else:
                    fine_repetitions = 100
                fine_timings = np.zeros((fine_repetitions, 1))
                for _ in range(10): # GPU-WARM-UP
                    _ = fine_tuned_pruned_model(test_time_inputs)
                with torch.no_grad():
                    for rep in range(fine_repetitions):
                        fine_starter.record()
                        _ = fine_tuned_pruned_model(test_time_inputs)
                        fine_ender.record()
                        # WAIT FOR GPU SYNC
                        torch.cuda.synchronize()
                        fine_curr_time = fine_starter.elapsed_time(fine_ender)
                        fine_timings[rep] = fine_curr_time
                        
                # -------------------------
                # Saving all experimental results
                # -------------------------
                tiff_iiinstant_log_batch = {**tiff_iiinstant_log_batch, 'Fine_Tuned(Test Time)': np.sum(fine_timings) / fine_repetitions}
                tiff_iiinstant_log_batch = {**tiff_iiinstant_log_batch, 'Pruning_Sparsity': sparsity}
                tiff_iiinstant_log_batch = {**tiff_iiinstant_log_batch, 'Initial_nparams': initial_number_of_parameters}
                tiff_iiinstant_log_batch = {**tiff_iiinstant_log_batch, 'Final_nparams': final_number_of_parameters}
                torch.save(one_shot_pruned_model.state_dict(), one_shot_path + pruning_module_name + ".pt")
                torch.save(fine_tuned_pruned_model.state_dict(), fine_tuned_path + pruning_module_name + ".pt")
                pruning_iiinstant_log_batch = {**pruning_iiinstant_log_batch, 'Pruning_Sparsity': sparsity}
                pruning_iiinstant_log_batch = {**pruning_iiinstant_log_batch, 'Oneshot(SSIM)': one_shot_pruned_ssim}
                pruning_iiinstant_log_batch = {**pruning_iiinstant_log_batch, 'Oneshot(PSNR)': one_shot_pruned_psnr}
                pruning_iiinstant_log_batch = {**pruning_iiinstant_log_batch, 'Fine_Tuned(SSIM)': fine_tuned_pruned_ssim}
                pruning_iiinstant_log_batch = {**pruning_iiinstant_log_batch, 'Fine_Tuned(PSNR)': fine_tuned_pruned_psnr}
                pruning_iiinstant_log_batch = {**pruning_iiinstant_log_batch, 'Fine_Tuned(Test Time)': np.sum(fine_timings) / fine_repetitions}
                pruning_iiinstant_log_batch = {**pruning_iiinstant_log_batch, 'Initial_nparams': initial_number_of_parameters}
                pruning_iiinstant_log_batch = {**pruning_iiinstant_log_batch, 'Final_nparams': final_number_of_parameters}
                pruning_iiinstant_log_batch = {**pruning_iiinstant_log_batch, 'Loss_Consensus': config['method']['loss_recon_consensus']}
                pruning_iiinstant_log_batch = {**pruning_iiinstant_log_batch, 'Prior_Type': config['pruning']['prior_path']}
                excel_log_metrics = Stack()
                excel_log_metrics.update_state(pruning_iiinstant_log_batch)
                pruing_excel_file_name = tiff_path + config['setting']['purpose'] + config['pruning']['fine_tune_loss_type'] + "tiff.csv"
                write_pruning(log_dict=excel_log_metrics.result(), save_path=pruing_excel_file_name)
                tiff_log_metrics = Stack()
                tiff_log_metrics.update_state(tiff_iiinstant_log_batch)
                write_pruning(log_dict=tiff_log_metrics.result(), save_path=tiff_path + "tiff.csv")
                del one_shot_pruned_model
                del unpruned_model
                del fine_tuned_pruned_model

                torch.cuda.empty_cache()
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    fire.Fire(main)

