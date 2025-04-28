import argparse
import os
import numpy as np
import nibabel as nib
import json
import time

import yaml
from utils.sampling import HealpixSampling
from utils.dataset import SingleSubjectdMRI
from utils.subject import SubjectdMRI
from model.model import ModelMultiSubject as Model

import torch
from torch.utils.data.dataloader import DataLoader

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def save_data(name, image, affine, header, save_result_path, root_save_result_path):
    image = np.array(image).astype(np.float32)
    img = nib.Nifti1Image(image, affine, header)
    nib.save(img, f"{save_result_path}/{name}.nii.gz")
    try:
        os.remove(f"{root_save_result_path}/{name}.nii.gz")
    except:
        pass
    os.symlink(f"{save_result_path}/{name}.nii.gz", f"{root_save_result_path}/{name}.nii.gz")


def main(config, config_training):
    # Load response function from weight file
    weight = torch.load(f"{config['testing']['model_path']}/history/epoch_{config['testing']['epoch']}.pth", map_location=torch.device('cpu'))
    if 'reconstruction.conv_equi.polar_filter' in weight.keys():
        polar_filter_equi = weight['reconstruction.conv_equi.polar_filter']
    else:
        polar_filter_equi = None
    if 'reconstruction.conv_inv.polar_filter' in weight.keys():
        polar_filter_inva = weight['reconstruction.conv_inv.polar_filter']
    else:
        polar_filter_inva = None
    
    # Create the deconvolution model
    feature_in = config_training['training']['feature_in']
    patch_size = config_training['model']['patch_size'] * (not config_training['model']['concatenate']) + 1 * config_training['model']['concatenate']
    graphSampling = HealpixSampling(config_training['model']['n_side'], config_training['model']['depth'], patch_size, sh_degree=config_training['model']['sh_degree'], pooling_name=config_training['model']['conv_name'], pooling_mode='average', hemisphere=config_training['model']['use_hemisphere'], legacy=config_training['model']['use_legacy']) # I changed it to max pooling, switching back now (looking for performance drop since midl). Max pooling: bad
    model = Model(graphSampling, polar_filter_equi, polar_filter_inva, feature_in, config_training['model']['filter_start'], config_training['model']['kernel_sizeSph'], config_training['model']['kernel_sizeSpa'], config_training['model']['normalize'], config_training['model']['conv_name'], config_training['model']['isoSpa'], config_training['model']['train_rf'])
    model.load_state_dict(weight, strict=False)
    model = model.to(DEVICE)
    model.eval()

    # Load the dataset
    rf_isotropic_names = []
    if config_training['model']['tissues']['gm']:
        rf_isotropic_names.append('gm_response')
    if config_training['model']['tissues']['csf']:
        rf_isotropic_names.append('csf_response')
    bvals_input = config_training['data']['bvals_input']
    bvals_output = config_training['data']['bvals_output']
    subject = SubjectdMRI(config['data']['data_path'], response_function_name=config_training['data']['rf_name'], verbose=True,
                    features_name='features', mask_name='mask', bvecs_name='bvecs.bvecs', bvals_name='bvals.bvals', gradient_mask_input_name=config_training['data']['gradient_mask'],
                    rf_isotropic_names=rf_isotropic_names, normalize_per_shell=config_training['data']['normalize_per_shell'], normalize_in_mask=config_training['data']['normalize_in_mask'], sh_degree=config_training['model']['sh_degree'],
                    loading_method=config['data']['loading_method'])
    dataset = SingleSubjectdMRI(subject, trained_bvals_input=bvals_input, trained_bvals_output=bvals_output, patch_size=config_training['model']['patch_size'], concatenate=config_training['model']['concatenate'], verbose=True)
    dataloader_test = DataLoader(dataset=dataset, batch_size=config['testing']['batch_size'], shuffle=False, num_workers=config['data']['cpu_dataloader'])
    n_batch = len(dataloader_test)
    print(n_batch)

    # Output initialization
    if config['testing']['middle_voxel']:
        b_selected = 1
        b_start = patch_size//2
        b_end = b_start + 1
    else:
        b_selected = patch_size
        b_start = 0
        b_end = b_selected

    nb_coef = int((config_training['model']['sh_degree'] + 1) * (config_training['model']['sh_degree'] / 2 + 1))
    count = np.zeros((dataset.subject.image.mask.shape[0],
                    dataset.subject.image.mask.shape[1],
                    dataset.subject.image.mask.shape[2]))
    reconstruction_list = np.zeros((dataset.subject.image.mask.shape[0],
                                    dataset.subject.image.mask.shape[1],
                                    dataset.subject.image.mask.shape[2], dataset.n_bval_output))
    if config_training['model']['tissues']['wm']:
        fodf_shc_wm_list = np.zeros((dataset.subject.image.mask.shape[0],
                                     dataset.subject.image.mask.shape[1],
                                     dataset.subject.image.mask.shape[2], nb_coef))
    if config_training['model']['tissues']['gm']:
        fodf_shc_gm_list = np.zeros((dataset.subject.image.mask.shape[0],
                                     dataset.subject.image.mask.shape[1],
                                     dataset.subject.image.mask.shape[2], 1))
    if config_training['model']['tissues']['csf']:
        fodf_shc_csf_list = np.zeros((dataset.subject.image.mask.shape[0],
                                      dataset.subject.image.mask.shape[1],
                                      dataset.subject.image.mask.shape[2], 1))
        
    # Test on batch.
    with torch.no_grad():
        start_tot = time.time()
        start = time.time()
        for i, data in enumerate(dataloader_test):
            #print(str(i * 100 / n_batch) + " %", end='\r', flush=True)
            # Load the data in the DEVICE
            input_features = data['input_features'].to(DEVICE)
            output_features = data['output_features'].to(DEVICE)
            output_mask = data['output_mask'].to(DEVICE)
            output_b0 = data['output_b0'].to(DEVICE)
            input_signal_to_shc = data['input_signal_to_shc'].to(DEVICE)
            output_shc_to_signal = data['output_shc_to_signal'].to(DEVICE)
            coords = data['coords']
            #print(torch.mean(input_features), torch.mean(output_features), torch.sum(output_mask), coords)
            model_time = time.time()
            output_reconstructed, deconvolved_equi_shc, deconvolved_inva_shc = model(input_features, output_b0, input_signal_to_shc, output_shc_to_signal)
            #print(torch.mean(output_reconstructed), torch.mean(deconvolved_equi_shc), torch.sum(deconvolved_inva_shc))
            #print(torch.mean((output_reconstructed - output_features)[output_mask.expand(-1, output_features.shape[1], -1, -1, -1)>0]**2))
            output_reconstructed = output_reconstructed * dataset.subject.response_functions.norm
            model_time =  time.time() - model_time
            for_loop_time = time.time()
                    
            for j in range(len(input_features)):
                x, y, z = coords[j].cpu().numpy().astype(int)
                reconstruction_list[x - (b_selected // 2):x + (b_selected // 2) + (b_selected%2),
                                    y - (b_selected // 2):y + (b_selected // 2) + (b_selected%2),
                                    z - (b_selected // 2):z + (b_selected // 2) + (b_selected%2)] += output_reconstructed[j, :, b_start:b_end, b_start:b_end, b_start:b_end].permute(1, 2, 3, 0).cpu().detach().numpy()
                if config_training['model']['tissues']['wm']:
                    fodf_shc_wm_list[x - (b_selected // 2):x + (b_selected // 2) + (b_selected%2),
                                    y - (b_selected // 2):y + (b_selected // 2) + (b_selected%2),
                                    z - (b_selected // 2):z + (b_selected // 2) + (b_selected%2)] += deconvolved_equi_shc[j, 0, :, b_start:b_end, b_start:b_end, b_start:b_end].permute(1, 2, 3, 0).cpu().detach().numpy()
                index = 0
                if config_training['model']['tissues']['gm']:
                    fodf_shc_gm_list[x - (b_selected // 2):x + (b_selected // 2) + (b_selected%2),
                                    y - (b_selected // 2):y + (b_selected // 2) + (b_selected%2),
                                    z - (b_selected // 2):z + (b_selected // 2) + (b_selected%2)] += deconvolved_inva_shc[j, index, :, b_start:b_end, b_start:b_end, b_start:b_end].permute(1, 2, 3, 0).cpu().detach().numpy()
                    index += 1
                if config_training['model']['tissues']['csf']:
                    fodf_shc_csf_list[x - (b_selected // 2):x + (b_selected // 2) + (b_selected%2),
                                    y - (b_selected // 2):y + (b_selected // 2) + (b_selected%2),
                                    z - (b_selected // 2):z + (b_selected // 2) + (b_selected%2)] += deconvolved_inva_shc[j, index, :, b_start:b_end, b_start:b_end, b_start:b_end].permute(1, 2, 3, 0).cpu().detach().numpy()
                count[x - (b_selected // 2):x + (b_selected // 2) + (b_selected%2),
                    y - (b_selected // 2):y + (b_selected // 2) + (b_selected%2),
                    z - (b_selected // 2):z + (b_selected // 2) + (b_selected%2)] += 1
            for_loop_time = time.time() - for_loop_time
            end = time.time()
            batch_time = end - start
            tot_time = end - start_tot
            if i%10 == 0:
                print(f'{i * 100 / n_batch:.3f}%, Elapsed time: {tot_time:.3f}s, Batch time: {batch_time:.3f}s - Model Time: {model_time:.3f} - Loop time: {for_loop_time:.3f} - {x} - {y} - {z} - estimated remaining time: {(n_batch / (i + 1) - 1) * tot_time :.1f}s - estimated total time: {(n_batch / (i + 1)) * tot_time :.1f}s', end='\r')
            start = time.time()

    # Average patch
    try:
        reconstruction_list[count!=0] = reconstruction_list[count!=0] / count[count!=0, None]
        if config_training['model']['tissues']['wm']:
            fodf_shc_wm_list[count!=0] = fodf_shc_wm_list[count!=0] / count[count!=0, None]
        if config_training['model']['tissues']['gm']:
            fodf_shc_gm_list[count!=0] = fodf_shc_gm_list[count!=0] / count[count!=0, None]
        if config_training['model']['tissues']['csf']:
            fodf_shc_csf_list[count!=0] = fodf_shc_csf_list[count!=0] / count[count!=0, None]
    except:
        print('Count failed')
    
    # Save the results
    root_save_result_path = f"{config['data']['data_path']}/result/{config['testing']['expname']}"
    save_result_path = f"{root_save_result_path}/test{'_middle'*config['testing']['middle_voxel']}/epoch_{config['testing']['epoch']}"
    yaml.safe_dump(config, open(os.path.join(root_save_result_path, 'config.yml'), 'w'), default_flow_style=False)
    yaml.safe_dump(config, open(os.path.join(save_result_path, 'config.yml'), 'w'), default_flow_style=False)
    
    # Number of model pass per voxel
    if config['testing']['save_count']:
        save_data('count', count, dataset.subject.image.affine, dataset.subject.image.header, save_result_path, root_save_result_path)
    # Reconstruction
    if config['testing']['save_reconstruction']:
        save_data('reconstruction', reconstruction_list, dataset.subject.image.affine, dataset.subject.image.header, save_result_path, root_save_result_path)
    # fODFs
    if config['testing']['save_fodf']:
        if config_training['model']['tissues']['wm']:
            save_data('fodf', fodf_shc_wm_list, dataset.subject.image.affine, dataset.subject.image.header, save_result_path, root_save_result_path)
        if config_training['model']['tissues']['gm']:
            save_data('fodf_gm', fodf_shc_gm_list, dataset.subject.image.affine, dataset.subject.image.header, save_result_path, root_save_result_path)
        if config_training['model']['tissues']['csf']:
            save_data('fodf_csf', fodf_shc_csf_list, dataset.subject.image.affine, dataset.subject.image.header, save_result_path, root_save_result_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        required=True,
        help='Path of the config file (default: None)',
        type=str
    )
    args = parser.parse_args()
    config_path = args.config
    config = yaml.safe_load(open(config_path, 'r'))

    # Test properties
    batch_size = config['testing']['batch_size']
    middle_voxel = config['testing']['middle_voxel']
    
    # Data path
    data_path = config['data']['data_path']
    assert os.path.exists(data_path)

    # Load trained model
    model_path = config['testing']['model_path']
    assert os.path.exists(f'{model_path}'), f'{model_path} does not exist'
    config_training = yaml.safe_load(open(f'{model_path}/config.yml', 'r'))
    if config['testing']['epoch'] is None:
        config['testing']['epoch'] = config_training['training']['last_epoch']
    epoch = config['testing']['epoch']

    # Test directory
    if config['testing']['expname'] is None:
        config['testing']['expname'] = config_training['training']['expname']
    model_name = config['testing']['expname']
    test_path = f"{data_path}/result/{model_name}/test{'_middle'*middle_voxel}/epoch_{epoch}"
    if not os.path.exists(test_path):
        print('Create new directory: {0}'.format(test_path))
        os.makedirs(test_path)

    main(config, config_training)
