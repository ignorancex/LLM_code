"""
This script shows how to perform rigid motion tracking in foetal time-series after having trained a corresponding net.
This function expects the input data to be organised as follows:
main_data_dir/time_series_1/name_image_dir: image_time_frame_0.nii.gz
                                            image_time_frame_1.nii.gz
                                          ...
                            name_label_dir: label_time_frame_0.nii.gz
                                            label_time_frame_1.nii.gz
                                            ...
                            name_xfm_dir:   gt_4x4_transform_matrix_1_to_0.npy
                                          ...
              time_series_2/name_image_dir: image_time_frame_0.nii.gz
                                            image_time_frame_1.nii.gz
                                            ...
                            name_label_dir: label_time_frame_0.nii.gz
                                            label_time_frame_1.nii.gz
                                            ...
                            name_xfm_dir:   gt_4x4_transform_matrix_1_to_0.npy
                                            ...
Note that label dir and transform dirs are optional.
"""

from spatial_regularisation.predict_time_series import predict

# ------------------ INPUT DATA

# the meaning of the following parameters is given in the docstring above, which describes the expected organisation of
# teh input data

main_data_dir = '/path/to/main/dir'
name_image_dir = 'images'
name_xfm_dir = 'gt_transforms'
name_labels_dir = None

path_main_model = '/data/model_rigid_motion_tracking/best_val_loss.pth'  # path of the rigid registration model

path_label_list = None  # here we do not provide a label list since no segmentations are available

# ------------------ OUTPUT DATA

results_dir = '/data/pair_registration_results/'
rig_only = True

# ------------------ ARCHITECTURE

# same as for training
net_type = 'se3'
closed_form_algo = 'numerical'
image_size = 96
n_channels = 32
n_levels = 4
n_conv = 2
n_feat = 32
feat_mult = 2
kernel_size = 3
last_activation = 'softmax'

predict(path_main_model=path_main_model,
        results_dir=results_dir,
        main_data_dir=main_data_dir,
        name_image_dir=name_image_dir,
        name_xfm_dir=name_xfm_dir,
        name_labels_dir=name_labels_dir,
        path_label_list=path_label_list,
        net_type=net_type,
        rig_only=rig_only,
        image_size=image_size,
        n_channels=n_channels,
        n_levels=n_levels,
        n_conv=n_conv,
        n_feat=n_feat,
        feat_mult=feat_mult,
        kernel_size=kernel_size,
        last_activation=last_activation,
        closed_form_algo=closed_form_algo)
