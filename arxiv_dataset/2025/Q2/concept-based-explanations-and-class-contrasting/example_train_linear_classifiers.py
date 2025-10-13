import os

from core.separate_features import train_linear_classifiers
from core.config import OUTPUT_PATH

#train_linear_classifiers(activation_vecs_folder=os.path.join(OUTPUT_PATH, "activations_for_gmm_training", "version_299"),
#                         output_folder=os.path.join(OUTPUT_PATH, "hyperplanes_from_attribution", "version_299"),
#                         features_list=[i for i in range(46)])

train_linear_classifiers(activation_vecs_folder=os.path.join(OUTPUT_PATH, "activations_for_gmm_training", "resnet50"),
                         output_folder=os.path.join(OUTPUT_PATH, "hyperplanes_from_attribution", "resnet50"),
                         features_list=[i for i in range(1000)])
