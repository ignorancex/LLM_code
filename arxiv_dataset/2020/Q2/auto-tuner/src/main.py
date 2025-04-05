"""
This is the main file for the "Auto Tuning of Double Quantum Dot Charge States" project

"""
import time
import os
import numpy as np

from src.data_generation import augmenter, labeler, occupation_labeler
from src.auto_tune import AutoTuner
from imgaug import augmenters as iaa


def augment_reference_data(train=False, evaluate=False):
    """

    function process the data for the reference point problem

    """
    print('Starting data processing ...')

    # define augmentation transformations
    rot = iaa.Sequential(
        [
            iaa.Affine(rotate=(-8, 8)),
        ])

    inc_capacities = iaa.Sequential(
        [
            iaa.Affine(rotate=(45, 45)),
            iaa.Affine(scale={"x": (0.6, 1.7), "y": (0.6, 1.7)}),  # before (0.7, 1.5)
            iaa.Affine(rotate=(-45, -45))
        ])

    scale = iaa.Sequential(
        [
            iaa.Affine(scale={"x": (0.6, 1.7), "y": (0.6, 1.7)}),  # before (0.65, 1.4)
        ]
    )

    # put the above transformations together and apply between 1 to all of them
    seq = iaa.SomeOf((1, None),
                     [
                         rot,
                         inc_capacities,
                         scale
                     ]
                     )

    # for swap_axes=True twice as many augmentation sequences will be performed
    # i.e. the same with swaped axes as well
    augmentation_sequence = [seq] * 8

    # Augment training data
    if train:
        # set paths, folders need to be created manually
        train_input_folder = '../../data/coarse/train/marked/'
        train_output_folder = '../../data/coarse/train/augmented/'

        A_train = augmenter.Augmenter(train_input_folder, train_output_folder,
                                      perform_augment=True, swap_axes=True)
        then = time.time()
        # perform augmentation
        while A_train.next_file():
            A_train.define_keypoints()
            A_train.augment(augmentation_sequence)
            A_train.save_maps()
        print('Augmenting training data took: {}s'.format(time.time() - then))

    # Augment evaluation data
    if evaluate:
        # set paths, folders need to be created manually
        evaluation_input_folder = '../../data/coarse/evaluation/marked/'
        evaluation_output_folder = '../../data/coarse/evaluation/augmented/'
        # do not actually augment the data, but still process them
        A_evaluation = augmenter.Augmenter(evaluation_input_folder, evaluation_output_folder,
                                           perform_augment=False, swap_axes=False)
        then = time.time()
        # perform augmentation
        while A_evaluation.next_file():
            A_evaluation.define_keypoints()
            A_evaluation.augment(augmentation_sequence)
            A_evaluation.save_maps()
        print('Augmenting evaluation data took: {}s'.format(time.time() - then))


def augment_transition_data(train=False, evaluate=False):
    """

    function to augment the data for the transition problem

    """
    print('Starting data processing ...')
    # define augmentation transformations
    rot = iaa.Sequential(
        [
            iaa.Affine(rotate=(-8, 8)),
        ])

    inc_capacities = iaa.Sequential(
        [
            iaa.Affine(rotate=(45, 45)),
            iaa.Affine(scale={"x": (0.6, 1.7), "y": (0.6, 1.7)}),  # before (0.7, 1.5)
            iaa.Affine(rotate=(-45, -45))
        ])

    scale = iaa.Sequential(
        [
            iaa.Affine(scale={"x": (0.6, 1.7), "y": (0.6, 1.7)}),  # before (0.65, 1.4)
        ]
    )

    # put the above transformations together and apply between 1 to all of them
    seq = iaa.SomeOf((1, None),
                     [
                         rot,
                         inc_capacities,
                         scale
                     ]
                     )

    # for swap_axes=True twice as many augmentation sequences will be performed
    # i.e. the same with swaped axes as well
    augmentation_sequence = [seq] * 8

    # Augment training data
    if train:
        # set paths, folders need to be created manually
        train_input_folder = '../data/fine/train/marked/'
        train_output_folder = '../data/fine/train/augmented/'

        A_train = augmenter.Augmenter(train_input_folder, train_output_folder,
                                      perform_augment=True, swap_axes=True)
        then = time.time()
        # perform augmentation
        while A_train.next_file():
            A_train.define_keypoints()
            A_train.augment(augmentation_sequence)
            A_train.save_maps()
        print('Augmenting training data took: {}s'.format(time.time() - then))

    # Augment evaluation data
    if evaluate:
        # set paths, folders need to be created manually
        evaluation_input_folder = '../data/fine/evaluation/marked/'
        evaluation_output_folder = '../data/fine/evaluation/augmented/'
        # do not actually augment the data, but still process them
        A_evaluation = augmenter.Augmenter(evaluation_input_folder, evaluation_output_folder,
                                           perform_augment=False, swap_axes=False)
        then = time.time()
        # perform augmentation
        while A_evaluation.next_file():
            A_evaluation.define_keypoints()
            A_evaluation.augment(augmentation_sequence)
            A_evaluation.save_maps()
        print('Augmenting evaluation data took: {}s'.format(time.time() - then))


def label_reference_data(train=False, evaluate=False):
    """
    Function that labels the data for the reference point task

    """
    # label training data
    if train:
        # set paths, folders need to be created manually
        training_path_in = '../../data/coarse/train/augmented/'
        training_path_out = '../../data/coarse/train/labeled/'

        then = time.time()
        label = occupation_labeler.Labeler(training_path_in, training_path_out, 4, 16)
        while label.next_file():
            label.create_shapes()
            label.create_frames(6, subtract_background=True)
            label.save_frames()
        print('TLabeling the training data took {}s'.format(time.time() - then))

    # label evaluation data
    if evaluate:
        # set paths, folders need to be created manually
        evaluation_path_in = '../../data/coarse/evaluation/augmented/'
        evaluation_path_out = '../../data/coarse/evaluation/labeled/'

        then = time.time()
        label = occupation_labeler.Labeler(evaluation_path_in, evaluation_path_out, 4, 16)
        while label.next_file():
            label.create_shapes()
            label.create_frames(6, subtract_background=False)
            label.save_frames()
        print('Labeling the evaluation data took {}s'.format(time.time() - then))


def label_transition_data(train=False, evaluate=False):
    """
    Function that labels the data for the reference point task

    """
    # label training data
    if train:
        # set paths, folders need to be created manually
        training_path_in = '../data/fine/train/augmented/'
        training_path_out = '../data/fine/train/labeled_scaled/'

        then = time.time()
        label = labeler.Labeler(training_path_in, training_path_out, 8, 12)
        while label.next_file():
            label.create_shapes()
            label.create_frames(10)
            label.save_frames()
        print('Labeling the training data took {}s'.format(time.time() - then))

    # label evaluation data
    if evaluate:
        # set paths, folders need to be created manually
        evaluation_path_in = '../data/fine/evaluation/augmented/'
        evaluation_path_out = '../data/fine/evaluation/labeled_scaled/'

        then = time.time()
        label = labeler.Labeler(evaluation_path_in, evaluation_path_out, 8, 12)
        while label.next_file():
            label.create_shapes()
            label.create_frames(10)
            label.save_frames()
        print('Labeling the evaluation data took {}s'.format(time.time() - then))


if __name__ == "__main__":
    ######################
    # Data Preprocessing #
    ######################

    # augment_reference_data(evaluate=True, train=True)
    # label_reference_data(evaluate=True, train=True)
    # augment_transition_data(evaluate=True, train=True)
    # label_transition_data(evaluate=True, train=True)

    ######################
    # Perform the tuning #
    ######################

    # define the gates
    gate_information = {
        'PG1': 'LPG0',
        'PG2': 'MPG0',
        'I_DQD': 'I TQD',
        'I_QPC': 'I QPC',
        'QPC_G': 'QPC_M'
    }

    path_in = os.path.abspath(r'C:\\Users\\Measure2\\Desktop\\measurement_series'
                              r'\\2019\\07\\Data_0716\\340_test_regime.hdf5')
    path_out = os.path.abspath(r'C:\\Users\\Measure2\\Desktop\\auto_tuner\\results\\outputs\\evaluation_run_14\\')

    then = time.time()
    # Define Auto Tuner Instance
    tune = AutoTuner(f=12, p=8,
                     gate_information=gate_information, path_in=path_in, path_out=path_out,
                     transition_recognizer='s_grid_6/s_grid_6_8', reference_recognizer='Binary_Grid_1/Binary_Grid_1_5')

    # define boundaries within which the starting point is drawn
    pg1_b = (-150e-3, -250e-3)
    pg2_b = (-200e-3, -350e-3)
    x_0 = (pg1_b[1] - pg1_b[0]) * np.random.random() + pg1_b[0]
    y_0 = (pg2_b[1] - pg2_b[0]) * np.random.random() + pg2_b[0]
    # find reference point
    tune.find_reference(16, 4, x_0=x_0, y_0=y_0, res=7.5e-3, confidence=0.5, calibrate=True, plot=True)
    # perform tuning
    tune.tune(1, 1, plot=True, calibrate=True)
    filepath = os.path.join(path_out, 'tuning_info.pkl')
    tune.save_tuning_info(filepath)
    print('Tuning took {}s'.format(time.time() - then))
