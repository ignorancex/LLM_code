"""
This script shows how to perform affine registration between brain MRIs of two populations.
"""

from spatial_regularisation.predict import predict

# ------------------ INPUT DATA

# the meaning of the following parameters is given in the docstring above, which describes the expected organisation of
# teh input data

testing_moving_im_dir = '/path/to/adni/brains'
testing_fixed_im_dir = '/path/to/hcp/brains'
testing_moving_lab_dir = '/path/to/adni/brain_segmentation'
testing_fixed_lab_dir = '/path/to/hcp/brain_segmentation'

path_main_model = '/data/model_affine_registration/best_val_loss.pth'  # path of the rigid registration model

path_label_list = '/path/to/synthseg/labels_list.npy'

# ------------------ OUTPUT DATA

results_dir = '/data/pair_registration_results/'
rig_only = False  # here we do affine registration

# ------------------ ARCHITECTURE

# same as for training
net_type = 'conv'
closed_form_algo = 'analytical'
image_size = 160
n_channels = 32
n_levels = 4
n_conv = 2
n_feat = 32
feat_mult = 2
kernel_size = 3
last_activation = 'softmax'

predict(path_main_model=path_main_model,
        results_dir=results_dir,
        testing_moving_im_dir=testing_moving_im_dir,
        testing_fixed_im_dir=testing_fixed_im_dir,
        testing_moving_lab_dir=testing_moving_lab_dir,
        testing_fixed_lab_dir=testing_fixed_lab_dir,
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
