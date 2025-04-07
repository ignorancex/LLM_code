"""
Script to train KeyMorph for affine registration of adult brain MRI with the proposed three-fold spatial regularisation.
All parameters are explained below.
"""

from spatial_regularisation.training import training

# ------------------ INPUTS

# path to training images, which need to be in nifty format.
training_im_dir = '/data/training/images'

# every n epochs, the model is being validated on a provided folder, by computing the accuracy of the predicted
# transforms w.r.t. ground truth transforms obtained by construction. If we have segmentations, we can also use them to
# compute Dice scores.
val_im_dir = '/data/validation/images'
val_lab_dir = '/data/validation/images'
path_label_list = '/path/to/label_list.npy'

# Finally, we need to provide a directory where all the intermediate models and validation scores will be saved.
results_dir = '/data/model_affine_registration'

# ------------------ LOSS

# In addition to the image similarity loss used to drive the registration task, we use three spatial regularisation
# terms on the features/keypoints
weight_kl_loss = 1                 # weight of regularisation to make features look like point spread functions
weight_var_loss = 0.01             # weight of regularisation of the sample covariance matrix of each feature map
weight_repulsive_loss = 0.001      # weight of the repulsive loss between extracted keypoints
temperature_repulsive_loss = 0.1   # temperature of the sigmoid used in the repulsive loss

# ------------------ ARCHITECTURE

# type of network, can be 'se3" for SE(3)-equivariant network, or 'conv' for regular CNN.
# We also need to choose the type of closed-form algorithm we use to estimate transform from point-clouds
net_type = 'conv'
closed_form_algo = 'analytical'

# In both cases, we build a UNet with provided parameters
n_channels = 32                 # number of output features
n_levels = 4                    # number of levels in the UNet
n_conv = 2                      # number of convolutions per level
n_feat = 32                     # number of features for the first layer
feat_mult = 2                   # multiplicator of features across levels
kernel_size = 3                 # size of the convolutional kernels
last_activation = 'softmax'     # last activation of the UNet (all the others are ReLU)

# ------------------ PREPROCESSING AND AUGMENTATION
image_size = 160  # size of the images for training, which will be enforced by zero-padding and cropping

# augmentation
rotation_range = 90      # range for the rotations, which will be drawn in [-rotation_range, +rotation_range]
shift_range = 5          # same for translations
shear_range = 0.08       # same for shears
scale_range = 0.05       # range for the scalings, which will be drawn in [1-rotation_range, 1+rotation_range]
max_noise_std = 0.02     # maximum standard deviation of the Gaussian noise to apply (higher = stronger).
max_bias_std = 0.2       # maximum standard deviation of the bias field corruption to apply (higher = stronger).
bias_scale = 0.04        # scale of the bias field (lower = smoother).
gamma_std = 0.15         # maximum value of the gamma-exponentiation for histogram shifting (higher = stronger).


# ------------------ LEARNING

batch_size = 4
learning_rate = 1e-5
n_epochs = 200               # this is much larger than what we need in reality, we stop based on validation scores
steps_per_epoch = 1000
validate_every_n_epoch = 1


training(training_im_dir=training_im_dir,
         val_im_dir=val_im_dir,
         results_dir=results_dir,
         net_type=net_type,
         val_lab_dir=val_lab_dir,
         path_label_list=path_label_list,
         image_size=image_size,
         rotation_range=rotation_range,
         shift_range=shift_range,
         max_noise_std=max_noise_std,
         max_bias_std=max_bias_std,
         bias_scale=bias_scale,
         gamma_std=gamma_std,
         n_channels=n_channels,
         n_levels=n_levels,
         n_conv=n_conv,
         n_feat=n_feat,
         feat_mult=feat_mult,
         kernel_size=kernel_size,
         last_activation=last_activation,
         closed_form_algo=closed_form_algo,
         batch_size=batch_size,
         learning_rate=learning_rate,
         n_epochs=n_epochs,
         steps_per_epoch=steps_per_epoch,
         validate_every_n_epoch=validate_every_n_epoch,
         weight_kl_loss=weight_kl_loss,
         weight_var_loss=weight_var_loss,
         weight_repulsive_loss=weight_repulsive_loss,
         temperature_repulsive_loss=temperature_repulsive_loss)
