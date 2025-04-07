import os
import numpy as np
import torch.utils.data

from spatial_regularisation import loaders
from spatial_regularisation import networks
import spatial_regularisation.losses as losses
from spatial_regularisation.utils import (
    build_subject_dict, build_xfm_dict, aff_to_field, interpolate, save_tensor, matrix_to_angles)

# set up cuda and device
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def predict(path_main_model,
            results_dir,
            testing_moving_im_dir,
            testing_fixed_im_dir,
            net_type,
            rig_only=True,
            testing_moving_lab_dir=None,
            testing_fixed_lab_dir=None,
            path_label_list=None,
            testing_xfm_dir=None,
            image_size=64,
            n_channels=32,
            n_levels=4,
            n_conv=2,
            n_feat=32,
            feat_mult=2,
            kernel_size=5,
            last_activation='softmax',
            closed_form_algo='numerical',
            recompute=False):
    """
    This function predicts rigid transforms between given pairs of fixed and moved images.

    The predicted transforms will be saved in the provided results_dir in two ways:
        1- results_dir/predicted_transforms/predicted_transforms_4x4_matrix/
            this folder will contain numpy arrays with predicted rigid transforms in homogeneous coordinates
        2- results_dir/predicted_transforms/predicted_rotation_angles.npy and predicted_translation_shifts.npy
            these are numpy matrices of shape 3xN_image_pairs with rotation and translations

    Finally, ground truth transforms can be given as well as binary masks for the moving and fixed images.
    In this case, we compute evaluation metrics which are saved under results_dir/test_scores.npy.
    This is a 3xN_image_pairs matrix where the rows correspond to 1) mean error in predicted angle 2) mean error in
    predicted translation 3) Dice score.
    """

    label_list = torch.tensor(np.load(path_label_list)) if path_label_list is not None else None

    # reformat inputs
    image_size = [image_size] * 3 if not isinstance(image_size, list) else image_size
    if ((testing_moving_lab_dir is None) & (testing_fixed_lab_dir is not None)) | \
            ((testing_moving_lab_dir is not None) & (testing_fixed_lab_dir is None)):
        raise ValueError('testing_moving_lab_dir and testing_fixed_lab_dir should either be both None or both given.')

    # create result directory
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'transforms'), exist_ok=True)
    path_results = os.path.join(results_dir, 'test_scores.npy')
    path_rotations = os.path.join(results_dir, 'rotations.npy')
    path_translations = os.path.join(results_dir, 'translations.npy')

    # check if we need to recompute
    if not os.path.isfile(path_results) or \
            (rig_only and (not os.path.isfile(path_rotations) or not os.path.isfile(path_translations))) or \
            recompute:

        # test loader
        testing_moving_subj_dict = build_subject_dict(testing_moving_im_dir, testing_moving_lab_dir)
        testing_fixed_subj_dict = build_subject_dict(testing_fixed_im_dir, testing_fixed_lab_dir)
        testing_xfm_dict = build_xfm_dict(testing_xfm_dir) if testing_xfm_dir is not None else None
        test_dataset = loaders.LoaderTesting(subj_dict_moving=testing_moving_subj_dict,
                                             subj_dict_fixed=testing_fixed_subj_dict,
                                             resize=image_size,
                                             return_masks=(testing_moving_lab_dir is not None),
                                             dict_xfm=testing_xfm_dict)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)
        list_moving_test_subjects = list(testing_moving_subj_dict.keys())
        list_fixed_test_subjects = list(testing_fixed_subj_dict.keys())

        # prepare network
        net = networks.Archi(net_type=net_type,
                             input_shape=image_size,
                             n_out_chan=n_channels,
                             n_levels=n_levels,
                             n_conv=n_conv,
                             n_feat=n_feat,
                             feat_mult=feat_mult,
                             kernel_size=kernel_size,
                             last_activation=last_activation,
                             closed_form_algo=closed_form_algo).to(device)
        net.load_state_dict(torch.load(path_main_model, map_location=torch.device(device), weights_only=True)['net_state_dict'])

        # test loop
        net.eval()
        list_scores = []
        list_rotations = []
        list_translations = []
        for i, batch in enumerate(test_loader):

            # initialise inputs
            moving = batch['scan_moving'].to(device)
            fixed = batch['scan_fixed'].to(device)

            # predict transformation
            xfm, feats_m, feats_f, means_m, means_f, covar_m, covar_f = net.forward((moving, fixed))

            # apply transformation
            grid_xfm = aff_to_field(xfm, image_size, invert_affine=True)
            moved = torch.moveaxis(interpolate(torch.moveaxis(moving, 1, -1), grid_xfm, 1, 'linear'), -1, 1)

            # save images
            list_tensors = [moving, fixed, moved, feats_m, feats_f]
            list_names = ['moving', 'fixed', 'moved', 'moving_features', 'fixed_features']
            for tens, name in zip(list_tensors, list_names):
                if 'fixed' in name:
                    path = os.path.join(results_dir, 'test_images', name, list_fixed_test_subjects[i] + '.nii.gz')
                else:
                    path = os.path.join(results_dir, 'test_images', name, list_moving_test_subjects[i] + '.nii.gz')
                save_tensor(tens, path, moveaxis='features' in name)

            # save transforms
            if rig_only:
                list_rotations.append(matrix_to_angles(xfm[:, :3, :3]).cpu().detach().numpy().squeeze() * 180 / np.pi)
                list_translations.append(xfm[:, :3, 3].cpu().detach().numpy().squeeze())
            np.save(os.path.join(results_dir, 'transforms', str(list_moving_test_subjects[i]) + '.npy'),
                    xfm.cpu().detach().numpy().squeeze())

            # evaluation
            tmp_list_scores = list()

            # compute transform similarity
            if testing_xfm_dict is not None:
                xfm_gt = batch['xfm_gt'].to(device)
                err_R = losses.l1_angle_from_matrix(xfm_gt[:, :3, :3], xfm[:, :3, :3]).mean().item()
                err_T = losses.l1_translation(xfm_gt[:, :3, 3], xfm[:, :3, 3]).mean().item()
                tmp_list_scores += [err_R, err_T]
                del xfm_gt, err_R, err_T

            # compute Dice score and save masks
            if testing_moving_lab_dir is not None:
                mask_moving = batch['mask_moving'].to(device)
                mask_fixed = batch['mask_fixed'].to(device)
                mask_moved = interpolate(mask_moving.moveaxis(1, -1), grid_xfm, 1, 'nearest').moveaxis(-1, 1)
                tmp_list_scores.append(losses.fast_dice(mask_fixed, mask_moved, label_list).mean().item())
                for tens, name in zip([mask_moving, mask_fixed, mask_moved],
                                      ['moving_mask', 'fixed_mask', 'moved_mask']):
                    if 'fixed' in name:
                        path = os.path.join(results_dir, 'test_images', name, list_fixed_test_subjects[i] + '.nii.gz')
                    else:
                        path = os.path.join(results_dir, 'test_images', name, list_moving_test_subjects[i] + '.nii.gz')
                    save_tensor(tens, path, moveaxis=False)
                del mask_moving, mask_fixed, mask_moved

            # compute topological metrics
            topo_loss = (losses.gaussian_kl_loss(feats_m, means_m, covar_m, 'kl', 'anisotropic') +
                         losses.gaussian_kl_loss(feats_f, means_f, covar_f, 'kl', 'anisotropic')).item() / 2
            var_loss = (losses.spectral_norm(covar_m).mean() + losses.spectral_norm(covar_f).mean()).item() / 2
            mean_dist = (losses.mean_point_dist(means_m) + losses.mean_point_dist(means_f)).item() / 2
            tmp_list_scores += [topo_loss, var_loss, mean_dist]
            del topo_loss, var_loss, mean_dist

            # save scores
            list_scores.append(tmp_list_scores)

            # flush cuda memory
            del moving, fixed, xfm, feats_m, feats_f, means_m, means_f, covar_m, covar_f, grid_xfm, moved, tens
            torch.cuda.empty_cache()

        # write scores
        if rig_only:
            np.save(path_rotations, np.array(list_rotations))
            np.save(path_translations, np.array(list_translations))
        np.save(path_results, np.array(list_scores))
