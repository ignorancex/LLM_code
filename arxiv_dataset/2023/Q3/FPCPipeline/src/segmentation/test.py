"""
Script to test  trained to segment lesions in the DCE-DWI dataset

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
import sys
sys.path.append(os.path.join(os.expanduser('~'), 'false_positive_classification_pipeline', 'utils' ))
sys.path.append(os.path.join(os.expanduser('~'), 'false_positive_classification_pipeline', 'unet' ))
sys.path.append(os.path.join(os.expanduser('~'), 'false_positive_classification_pipeline', 'seg_metrics' ))
from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from model import UNet
from lits_dataset.dataset import LITSDataset
from lesionsegdataset.dataset import LesionSegMRTest
from helper_functions import *
from utils.utils import *
from utils.image_utils import post_process_predicted_mask
import numpy as np
from metrics import *
from lesion_correspondence import *
import shutil
from scipy.ndimage import binary_opening, binary_dilation, generate_binary_structure
import warnings
import joblib

warnings.filterwarnings("ignore", category=UserWarning)

BINARY_THRESHOLD=0.5
PATCH_SIZE=256
STRIDE=32
UNC_THRESH = 0.6
SEED = 42 # To choose random angles

if PATCH_SIZE == 256:
    USE_GAUSS = False
else:
    USE_GAUSS = True

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, help='Path to checkpoint directory')
    parser.add_argument('--data_dir', type=str, help='Path to test data directory', default='/home/ishaan/Desktop/UMC_Data/LesionDetection/Detection/LiverMetastasesDetectionData/DetectionData/Testing')
    parser.add_argument('--dataset', type=str, help='Dataset to pick (lits or UMC)', default='umc')
    parser.add_argument('--batch_size',type=int, help='Number of slices in a single batch', default=16)
    parser.add_argument('--iter',type=int, help='iter to choose to load from', default=-1)
    parser.add_argument('--n_orientations',type=int, help='Discretization of SE(2) group', default=1)
    parser.add_argument('--gpu_id',type=int, help='GPU to use', default=0)
    parser.add_argument('--model',type=str, help='NN model to be used: pnet or unet', default='unet')
    parser.add_argument('--dropout', action='store_true')
    parser.add_argument('--ensemble', action='store_true')
    parser.add_argument('--tta', action='store_true', help='Flag to turn on test-time augmentation')
    parser.add_argument('--mode', type=str, help='Specify which part of the data we want evaluated', default='test')
    args = parser.parse_args()
    return args




def test_model(args):

    device = torch.device('cuda:{}'.format(args.gpu_id))
    print('Using device: {}'.format(device))

    print('Enable dropout -- {}'.format(args.dropout))
    print('Do TTA -- {}'.format(args.tta))
    print('Script mode -- {}'.format(args.mode))
    print('Ensemble mode -- {}'.format(args.ensemble))

    # Both cannot be turned on at the same time!
    assert (args.ensemble & args.dropout is False)

    if args.dropout is True:
        if args.dataset  == 'umc':
            T_mcd = 20
        else:
            T_mcd = 10 # Avoid CPU crashes due to excessive RAM usage
    else:
        T_mcd = 1

    if args.tta is True:
        T_rot = 10
        np.random.seed(SEED)
        rot_angles = list(np.random.uniform(-30, 30, size=T_rot))
    else:
        T_rot = 1
        rot_angles = [0]

    # Collect patient_dirs from the test or val directory directory
    if args.ensemble is False:
        checkpoint_dir = [args.checkpoint_dir]
    else:
        checkpoint_dir = [f.path for f in os.scandir(args.checkpoint_dir) if (f.is_dir() and ('seed' in f.name))]

    if args.dropout is True:
        config = joblib.load(os.path.join(checkpoint_dir[0], 'best_config.pkl'))
        checkpoint_dir = os.path.join(checkpoint_dir[0], 'best_trial', 'checkpoints')
        print(config)
        dropout_rate = config['dropout_rate']
        structured_dropout = config['structured_dropout']
        num_blocks = config['num_blocks']
        checkpoint_dir = [checkpoint_dir]
    else:
        dropout_rate = 0.0
        structured_dropout = False
        num_blocks = 3

    if args.mode == 'test':
        if args.dataset == 'umc':
            pat_dirs = get_patient_dirs(args.data_dir)
        elif args.dataset == 'lits':
            test_patients_dir_list = os.path.join(checkpoint_dir[0], 'test_patients.pkl')

            with open(test_patients_dir_list, 'rb') as f:
                pat_dirs = pickle.load(f)

        if args.tta is True:
            folder_name = 'images_tta'
        else:
            folder_name = 'images'

        if args.iter > 0:
            res_dir = os.path.join(args.checkpoint_dir, folder_name+'_{}'.format(args.iter))
        else:
            res_dir = os.path.join(args.checkpoint_dir, folder_name)

    else:
        if args.dataset == 'lits':

            # Val patients
            val_patients_dir_list = os.path.join(checkpoint_dir[0], 'val_patients.pkl')
            with open(val_patients_dir_list, 'rb') as f:
                pat_dirs = pickle.load(f)
            # Train patients
            if args.mode == 'train': # We want predictions on train + val patients
                train_patients_dir_list = os.path.join(checkpoint_dir[0], 'train_patients.pkl')
                with open(train_patients_dir_list, 'rb') as f:
                    train_pat_dirs = pickle.load(f)

                pat_dirs.extend(train_pat_dirs)

        elif args.dataset == 'umc':
            pat_dirs = get_patient_dirs(args.data_dir)

        if args.tta is True:
            folder_name = '{}_images_tta'.format(args.mode)
        else:
            folder_name = '{}_images'.format(args.mode)

        res_dir = os.path.join(args.checkpoint_dir, folder_name)

    if args.dataset == 'lits':
        # Read the fingerprint DF
        fingerprint_df = pd.read_csv(os.path.join(args.data_dir, 'dataset_fingerprints.csv'))

    models = []

    # Load model(s)
    for model_idx, c_dir in enumerate(checkpoint_dir):
        if args.dataset == 'umc':
            model = UNet(dropout=args.dropout,
                         dropout_rate=dropout_rate,
                         structured_dropout=structured_dropout,
                         num_blocks=num_blocks,
                         n_channels=9,
                         num_classes=2,
                         use_pooling=True)

        elif args.dataset == 'lits':
            model = UNet(n_channels=1,
                         num_classes=2,
                         base_filter_num=64,
                         num_blocks=num_blocks,
                         dropout=args.dropout,
                         dropout_rate=dropout_rate,
                         structured_dropout=structured_dropout,
                         use_pooling=True)
        else:
            raise ValueError('{} dataset is invalid'.format(args.dataset))


        load_dict = load_model(model=model,
                               checkpoint_dir=c_dir,
                               training=False,
                               suffix=args.iter)

        model = load_dict['model']
        model.eval()
        set_mcd_eval_mode(model) # Set dropout layers to 'train' mode for MC-Dropout
        models.append(model)


    if args.tta is True and args.dropout is True: # To ensure different angles have the same weight setting
        seed_matrix = torch.randint(low=10000, high=1000000, size=(T_mcd, 6))
    else:
        seed_matrix= None


    # Intialize dictionary to store all the per-patient metrics
    metrics_dict = {}
    metrics_dict['Patient ID'] = []
    metrics_dict['True Lesions'] = []
    metrics_dict['Detected Lesions'] = []
    metrics_dict['False Positives'] = []
    metrics_dict['False Negatives'] = []
    metrics_dict['Precision'] = []
    metrics_dict['Recall'] = []
    metrics_dict['Dice'] = []

    if args.mode != 'train':
        metrics_dict['ECE'] = []
        metrics_dict['BB ECE'] = []

    # Model inference on a per-patient basis
    for pat_dir in pat_dirs:
        if args.dataset == 'umc':
            dataset = LesionSegMRTest(pat_dir=pat_dir,
                                      patch_size=PATCH_SIZE,
                                      stride=STRIDE)
        else:
            dataset = LITSDataset(data_dir=args.data_dir,
                                  patient_id=pat_dir,
                                  fingerprint_df=fingerprint_df,
                                  mode='test')

        mask_shape = dataset.get_mask_shape()
        if args.dataset == 'umc':
            true_mask = dataset.get_mask_np()
            liver_mask = dataset.get_liver_mask_np()
            mask_itk = dataset.get_mask_itk()
        else:
            mask_np = dataset.get_mask_np()
            mask_itk = dataset.get_mask_itk()
            true_mask = np.where(mask_np == 2, 1, 0).astype(np.uint8)
            liver_mask = np.where(mask_np != 0, 1, 0).astype(np.uint8)

        metadata = {}
        metadata['spacing'] = mask_itk.GetSpacing()
        metadata['direction'] = mask_itk.GetDirection()
        metadata['origin'] = mask_itk.GetOrigin()

        merged_softmax_output = torch.zeros(size=(T_rot, T_mcd, len(models), 2, mask_shape[0], mask_shape[1], mask_shape[2]),
                                            dtype=torch.float32)

        merged_output = torch.zeros(size=(T_rot, T_mcd, len(models), 2, mask_shape[0], mask_shape[1], mask_shape[2]),
                                            dtype=torch.float32)


        if args.dataset == 'umc':
            merged_dce = torch.zeros(size=(6, mask_shape[0], mask_shape[1], mask_shape[2]), dtype=torch.float32)
            merged_counts = torch.zeros(size=mask_shape, dtype=torch.float32)

        if USE_GAUSS is True:
            gauss_imp_map = torch.from_numpy(create_gaussian_importance_map(PATCH_SIZE))

        dataloader = DataLoader(dataset=dataset,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=4)

        print('Inference for patient {} with {} patches/slices'.format(dataset.get_patient_id(), dataset.__len__()))

        with torch.no_grad():
            for i, data in enumerate(dataloader):
                if args.dataset == 'umc':
                    dce_patches, dwi_patches, labels = data['dce'].float(), data['dwi'].float(), data['label'].float()
                    slices, patch_idxs = data['slice'], data['patch']
                    images = torch.cat([dce_patches, dwi_patches], dim=1)
                elif args.dataset == 'lits':
                    images, labels = data['image'], data['lesion_labels']
                    images = torch.unsqueeze(images, dim=1)
                    slices = data['slices']

                for rot_idx, rot_angle in enumerate(rot_angles): # TTA
                    rot_images = rotate_tensor(img=images,
                                               order=1,
                                               rot_angle=rot_angle)

                    for i in range(T_mcd): # MC-Dropout
                        # Multiple models in case of an ensemble
                        for e_idx, model in enumerate(models): # Ensemble

                            model.to(device)

                            if args.tta is True and args.dropout is True:
                                preds = model(rot_images.to(device),
                                              seeds=seed_matrix[i])
                            else:
                                preds = model(rot_images.to(device))

                            # Inverse rotate the prediction
                            preds = rotate_tensor(img=preds.cpu(),
                                                  rot_angle=-1*rot_angle,
                                                  order=0,
                                                  verbose=False)

                            norm_preds = F.softmax(preds, dim=1)

                            # Put it into the right "slot" (w.r.t Rotation iteration, MC-D iteration and slice ID)
                            for batch_idx,slice_id in enumerate(slices):
                                if args.dataset == 'lits':
                                    merged_softmax_output[rot_idx, i, e_idx, :, slice_id, :, :] = norm_preds[batch_idx, :, :, :]
                                    merged_output[rot_idx, i, e_idx, :, slice_id, :, :] = preds[batch_idx, :, :, :]
                                elif args.dataset == 'umc':
                                    merged_softmax_output[rot_idx, i, e_idx, :, :, :, slice_id] = norm_preds[batch_idx, :, :, :]
                                    merged_output[rot_idx, i, e_idx, :, :, :, slice_id] = preds[batch_idx, :, :, :]


            merged_softmax_output = merged_softmax_output.numpy()
            merged_output = merged_output.numpy()


            # Take the mean over different MC-D iterations, rotation angles and ensemble axes
            normalized_softmax_output = np.mean(merged_softmax_output, axis=(0, 1, 2))
            logit_map = np.mean(merged_output, axis=(0, 1, 2))

            # Mean over the rotation angles, retaining the outputs from different MC-D/Ensemble iters
            if args.ensemble is False:
                normalized_softmax_output_aleatoric = np.mean(merged_softmax_output, axis=(0, 2)) # Mean over (rotation, ensemble) axes
            else:
                normalized_softmax_output_aleatoric = np.mean(merged_softmax_output, axis=(0, 1)) # Mean over (rotation, mc-d) axes


            if args.dropout is False and args.ensemble is False:
                assert(np.array_equal(normalized_softmax_output_aleatoric[0], normalized_softmax_output))

            normalized_softmax_output[1, :, :, :] = normalized_softmax_output[1, :, :, :]*liver_mask
            normalized_softmax_output[0, :, :, :] = 1 - normalized_softmax_output[1, :, :, :]

            normalized_softmax_output_aleatoric[:, 1, :, :, :] = normalized_softmax_output_aleatoric[:, 1, :, :, :]*liver_mask
            normalized_softmax_output_aleatoric[:, 0, :, :, :] = 1 - normalized_softmax_output_aleatoric[:, 1, :, :, :]

            # Calcuate normalized entropy (Scaled between 0 and 1)
            pred_uncertainty = calculate_entropy(normalized_softmax_output)
            normalized_pred_uncertainty = (pred_uncertainty - np.amin(pred_uncertainty))/(np.amax(pred_uncertainty) - np.amin(pred_uncertainty))

            # Calculate aleatoric uncertainty
            aleatoric_uncertainty = np.zeros_like(normalized_softmax_output_aleatoric[:, 1, :, :, :])

            for idx in range(T_mcd):
                aleatoric_uncertainty[idx, :, :, :] = calculate_entropy(normalized_softmax_output_aleatoric[idx,:, :, :, :])

            aleatoric_uncertainty = np.mean(aleatoric_uncertainty, axis=0)

            if (T_mcd == 1) and (args.ensemble is False): # Fixed weights => all uncertainty is aleatoric
                assert(np.array_equal(aleatoric_uncertainty, pred_uncertainty) is True)

            normalized_aleatoric_uncertainty = (aleatoric_uncertainty - np.amin(aleatoric_uncertainty))/(np.amax(aleatoric_uncertainty) - np.amin(aleatoric_uncertainty))

            # Mask out entropy in non-liver regions
            normalized_pred_uncertainty = normalized_pred_uncertainty*liver_mask
            normalized_aleatoric_uncertainty = normalized_aleatoric_uncertainty*liver_mask

            # Calculate epistemic uncertainty
            epistemic_uncertainty = pred_uncertainty - aleatoric_uncertainty

            if (T_mcd == 1) and (args.ensemble is False):
                normalized_epistemic_uncertainty = epistemic_uncertainty
            else:
                print('Minimum value in epistemic unc. map = {}'.format(np.amin(epistemic_uncertainty)))
                epistemic_uncertainty = np.where(epistemic_uncertainty < 0, 0, epistemic_uncertainty)
                normalized_epistemic_uncertainty = (epistemic_uncertainty - np.amin(epistemic_uncertainty))/(np.amax(epistemic_uncertainty) - np.amin(epistemic_uncertainty))

            normalized_epistemic_uncertainty = normalized_epistemic_uncertainty*liver_mask

            # Threshold the foreground output to get a binary mask
            binary_mask = np.where(normalized_softmax_output[1, :, :, :] > BINARY_THRESHOLD, 1, 0).astype(np.uint8)

            binary_mask = binary_mask*liver_mask

            post_proc_binary_mask = post_process_predicted_mask(binary_mask)

            # Binary opening on true mask -- Refer Marielle's code EvaluationDetection.m
            s_elem = generate_binary_structure(rank=3, connectivity=1)
            true_mask = true_mask*liver_mask
            true_mask_opened =  binary_opening(input=true_mask,
                                               structure=s_elem).astype(np.uint8)

            patient_dice = calculate_dice_score(post_proc_binary_mask, true_mask_opened)
            print('Dice score for patient {} = {}'.format(dataset.get_patient_id(), patient_dice))


            # Create lesion correspondence graph
            corr_graph = create_correspondence_graph(seg=post_proc_binary_mask,
                                                     gt=true_mask_opened)

            lesion_counts = count_detections(corr_graph,
                                             gt=true_mask_opened,
                                             seg=post_proc_binary_mask)

            # Compute Exptected Calibration Error (ECE)
            if args.mode != 'train':
                ece = expected_calibration_error(probabilities = normalized_softmax_output[1, :, :, :],
                                                 ground_truth = true_mask_opened,
                                                 bins=10)

                print('ECE for patient {} = {}'.format(dataset.get_patient_id(), ece))

                # Compute ECE for bounding boxes around the lesion
                segmented_lesions = lesion_counts['slices']
                missed_lesions = lesion_counts['fn_slices']

                # Merge both lists
                lesion_list = segmented_lesions + missed_lesions

                if len(lesion_list) > 0:
                    ece_per_bounding_box = []
                    softmax_fg = normalized_softmax_output[1, :, :, :]

                    for lesion_bb in lesion_list:
                        struct_elem = np.ones((3, 3, 3), dtype=np.uint8)

                        single_lesion_mask = np.zeros_like(post_proc_binary_mask)
                        single_lesion_mask[lesion_bb] += post_proc_binary_mask[lesion_bb]

                        # Dilate this mask
                        single_lesion_mask = binary_dilation(input=single_lesion_mask,
                                                             structure=np.ones((3,3,1), dtype=np.uint8))

                        # Get the "dilated" bounding box
                        lesion_bb_dilated_list, num_objects = return_lesion_coordinates(single_lesion_mask)

                        for lesion_bb_dilated in lesion_bb_dilated_list:
                            ece_per_bounding_box.append(expected_calibration_error(probabilities = softmax_fg[lesion_bb_dilated],
                                                                                   ground_truth = true_mask_opened[lesion_bb_dilated],
                                                                                   bins=10))


                    mean_ece_over_bbs = np.mean(np.array(ece_per_bounding_box))

                    print('ECE over bounding boxes for patient {} = {}'.format(dataset.get_patient_id(), mean_ece_over_bbs))
                else:
                    mean_ece_over_bbs = np.nan

                metrics_dict['ECE'].append(ece)
                metrics_dict['BB ECE'].append(mean_ece_over_bbs)


            metrics_dict['Patient ID'].append(dataset.get_patient_id())
            metrics_dict['True Lesions'].append(lesion_counts['true lesions'])
            metrics_dict['Detected Lesions'].append(lesion_counts['true positives'])
            metrics_dict['False Positives'].append(lesion_counts['false positives'])
            metrics_dict['False Negatives'].append(lesion_counts['false negatives'])
            metrics_dict['Precision'].append(lesion_counts['precision'])
            metrics_dict['Recall'].append(lesion_counts['recall'])
            metrics_dict['Dice'].append(patient_dice)


            # Save image, true mask and merged predictions
            img_dir = os.path.join(res_dir, '{}'.format(dataset.get_patient_id()))
            if os.path.exists(img_dir) is True:
                shutil.rmtree(img_dir)
            os.makedirs(img_dir)

            if args.dataset == 'umc':
                z_first = False
            elif args.dataset == 'lits':
                z_first = True

            if args.dataset == 'umc':
                save_itk_image(data=np.transpose(dataset.get_dce_np(), (3, 0, 1, 2)),
                               fname = os.path.join(img_dir, 'dce.nii.gz'),
                               metadata=metadata)
            elif args.dataset == 'lits':
                image_np = dataset.get_image_np()
                save_itk_image(data=image_np,
                               fname=os.path.join(img_dir, 'image.nii.gz'),
                               z_first=True,
                               metadata=metadata)

            save_itk_image(data=post_proc_binary_mask,
                           fname = os.path.join(img_dir, 'binary_post_proc_pred.nii.gz'),
                           z_first=z_first,
                           metadata=metadata)
            save_itk_image(data=liver_mask,
                           fname = os.path.join(img_dir, 'liver_mask.nii.gz'),
                           z_first=z_first,
                           metadata=metadata)
            save_itk_image(data=true_mask_opened,
                           fname=os.path.join(img_dir, 'true_mask.nii.gz'),
                           z_first=z_first,
                           metadata=metadata)
            save_itk_image(data=normalized_pred_uncertainty,
                           fname=os.path.join(img_dir, 'uncertainty_map.nii.gz'),
                           z_first=z_first,
                           metadata=metadata)
            save_itk_image(data=normalized_aleatoric_uncertainty,
                           fname=os.path.join(img_dir, 'aleatoric_uncertainty_map.nii.gz'),
                           z_first=z_first,
                           metadata=metadata)
            save_itk_image(data=normalized_epistemic_uncertainty,
                           fname=os.path.join(img_dir, 'epistemic_uncertainty_map.nii.gz'),
                           z_first=z_first,
                           metadata=metadata)
            save_itk_image(data=normalized_softmax_output[1, :, :, :],
                           fname=os.path.join(img_dir, 'softmax_pred.nii.gz'),
                           z_first=z_first,
                           metadata=metadata)

            # Save the raw uncertainty maps
            save_itk_image(data=pred_uncertainty,
                           fname=os.path.join(img_dir, 'raw_uncertainty_map.nii.gz'),
                           z_first=z_first,
                           metadata=metadata)
            save_itk_image(data=aleatoric_uncertainty,
                           fname=os.path.join(img_dir, 'raw_aleatoric_uncertainty_map.nii.gz'),
                           z_first=z_first,
                           metadata=metadata)
            save_itk_image(data=epistemic_uncertainty,
                           fname=os.path.join(img_dir, 'raw_epistemic_uncertainty_map.nii.gz'),
                           z_first=z_first,
                           metadata=metadata)

            # Save logit map (for CRF post-processing)
            np.save(os.path.join(img_dir, 'logit_map.npz'),
                    logit_map)

            if corr_graph is not None:
                visualize_lesion_correspondences(corr_graph, fname=os.path.join(img_dir, 'lesion_graph.png'))

    # Save the list of dictionaries as CSV file for later analysis
    save_metric_dict(metric_dict=metrics_dict, fname=os.path.join(res_dir, 'metrics.csv'))




if __name__ == '__main__':
    args = build_parser()
    test_model(args)
