# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from scipy.stats import wilcoxon
import os
import pylab as py

def convert_string_to_floats(long_string_with_commas):
    long_string_without_brackets = long_string_with_commas.replace('[', '').replace(']', '')
    number_strings = long_string_without_brackets.split(',')
    float_numbers = [float(number.replace(' ', '')) for number in number_strings]
    return float_numbers


def compute_wilcoxon_test(set_1, set_2):
    set1 = np.array(set_1)

    set2 = np.array(set_2)

    d = np.squeeze(np.subtract(set1, set2))
    # To test the null hypothesis that there
    # is no value difference between the two sets, we can apply the two-sided test
    # that is we want to verify that the distribution underlying d is not symmetric about zero.
    res = wilcoxon(d, alternative='less')
    print(res.statistic, res.pvalue)


def compute_results(path_csv_1, path_csv_2, path_csv_3, exp_name, index_csv):
    exp_1 = pd.read_csv(path_csv_1, header=None)
    exp_2 = pd.read_csv(path_csv_2, header=None)
    exp_3 = pd.read_csv(path_csv_3, header=None)

    exp_1 = convert_string_to_floats(exp_1.iloc[index_csv][1])
    exp_2 = convert_string_to_floats(exp_2.iloc[index_csv][1])
    exp_3 = convert_string_to_floats(exp_3.iloc[index_csv][1])

    exp_1_average = np.mean(exp_1)
    exp_1_std = np.std(exp_1)

    exp_2_average = np.mean(exp_2)
    exp_2_std = np.std(exp_2)

    exp_3_average = np.mean(exp_3)
    exp_3_std = np.std(exp_3)

    overall_average = (exp_1_average + exp_2_average + exp_2_average) / 3
    overall_std = np.sqrt(((exp_1_std * exp_1_std) + (exp_2_std * exp_2_std) + (exp_3_std * exp_3_std)) / 3)

    #  print(f'{exp_name}_1: {exp_1_average} +- {exp_1_std}')
    #  print(f'{exp_name}_2: {exp_2_average} +- {exp_2_std}')
    # print(f'{exp_name}_3: {exp_3_average} +- {exp_3_std}')

    print(f'{exp_name}_overall: {overall_average} +- {overall_std}')
    print("-----------------------------------------------------------")


def compute_results_2(path_csv_1, path_csv_2, path_csv_3, exp_name, index_csv):
    exp_1 = pd.read_csv(path_csv_1, header=None)
    exp_2 = pd.read_csv(path_csv_2, header=None)
    exp_3 = pd.read_csv(path_csv_3, header=None)

    exp_1 = convert_string_to_floats(exp_1.iloc[index_csv][1])
    exp_2 = convert_string_to_floats(exp_2.iloc[index_csv][1])
    exp_3 = convert_string_to_floats(exp_3.iloc[index_csv][1])

    # print(len(exp_1))
    # print(len(exp_2))
    # print(len(exp_3))
    average = [(im_1 + im_2 + im_3) / 3 for im_1, im_2, im_3 in zip(exp_1, exp_2, exp_3)]
    # average = [(im_2 + im_3) / 2 for im_2, im_3 in zip(exp_2, exp_3)]

    std = [np.std([im_1, im_2, im_3]) for im_1, im_2, im_3 in zip(exp_1, exp_2, exp_3)]
    # std = [np.std([im_2, im_3]) for im_2, im_3 in zip(exp_2, exp_3)]

    overall_average = np.mean(average)
    squared_sum = 0
    for i in std:
        squared_sum += i * i
    overall_std = np.sqrt(squared_sum / len(std))

    # print(f'{exp_name}_1: {exp_1_average} +- {exp_1_std}')
    #  print(f'{exp_name}_2: {exp_2_average} +- {exp_2_std}')
    # print(f'{exp_name}_3: {exp_3_average} +- {exp_3_std}')

    print(f'{overall_average} +- {overall_std}')
    print("-----------------------------------------------------------")
    return average


def avg_fid(path1, path2, path3):
    exp_1 = pd.read_csv(path1, header=None)
    exp_2 = pd.read_csv(path2, header=None)
    exp_3 = pd.read_csv(path3, header=None)

    # print(exp_1[0])
    average = (exp_1.loc[0][1] + exp_2.loc[0][1] + exp_3.loc[0][1]) / 3
    std = np.std([exp_1.loc[0][1], exp_2.loc[0][1], exp_3.loc[0][1]])
    return average, std


def element_wise_average(lists):
    list_lengths = [len(sublist) for sublist in lists[0]]

    #if any(length != list_lengths[0] for length in list_lengths):
    #     raise ValueError("All lists must have the same length")

    result = []
    for sublists in zip(*lists):
        if any(len(sublist) != len(sublists[0]) for sublist in sublists):
            raise ValueError("Sublists must have the same length")

        avg_sublist = [sum(values) / len(values) for values in zip(*sublists)]
        result.append(avg_sublist)

    return result


def compute_average_profiles(folder_path, **kwargs):
    path_list = []
    for _, path in kwargs.items():
        path_list.append(os.path.join(folder_path, path))

    profiles = []
    for i in path_list:
        experiment_profile = open(i)
        profiles.append(experiment_profile)

    return element_wise_average(profiles)



if __name__ == '__main__':

    # RAPS (radially averaged-power spectrum)

    # baseline
    profile_elcap_baseline = compute_average_profiles("/Volumes/sandisk/cycleGAN_results/metrics_ep50_cycleGAN", path_1="metrics_baseline_10/raps_elcap_complete_epoch70.json",
                                                      path_2="metrics_baseline_11/raps_elcap_complete_epoch70.json",
                                                      path_3="metrics_baseline_12/raps_elcap_complete_epoch70.json")
    profile_test3_baseline = compute_average_profiles("/Volumes/sandisk/cycleGAN_results/metrics_ep50_cycleGAN", path_1="metrics_baseline_10/raps_test_3_epoch70.json",
                                                      path_2="metrics_baseline_11/raps_test_3_epoch70.json",
                                                      path_3="metrics_baseline_12/raps_test_3_epoch70.json")

    # texture max
    profile_elcap_texture_max = compute_average_profiles("/Volumes/sandisk/cycleGAN_results/metrics_ep50_cycleGAN",path_1="metrics_texture_max_10/raps_elcap_complete_epoch70.json",
                                                      path_2="metrics_texture_max_11/raps_elcap_complete_epoch70.json",
                                                      path_3="metrics_texture_max_12/raps_elcap_complete_epoch70.json")

    profile_test_3_texture_max = compute_average_profiles("/Volumes/sandisk/cycleGAN_results/metrics_ep50_cycleGAN",path_1="metrics_texture_max_10/raps_test_3_epoch70.json",
                                                      path_2="metrics_texture_max_11/raps_test_3_epoch70.json",
                                                      path_3="metrics_texture_max_12/raps_test_3_epoch70.json")
    # texture avg
    profile_elcap_texture_avg = compute_average_profiles("/Volumes/sandisk/cycleGAN_results/metrics_ep50_cycleGAN", path_1="metrics_texture_avg_10/raps_elcap_complete_epoch70.json",
                                                         path_2="metrics_texture_avg_11/raps_elcap_complete_epoch70.json",
                                                         path_3="metrics_texture_avg_12/raps_elcap_complete_epoch70.json")

    profile_test_3_texture_avg = compute_average_profiles("/Volumes/sandisk/cycleGAN_results/metrics_ep50_cycleGAN", path_1="metrics_texture_avg_10/raps_test_3_epoch70.json",
                                                          path_2="metrics_texture_avg_11/raps_test_3_epoch70.json",
                                                          path_3="metrics_texture_avg_12/raps_test_3_epoch70.json")
    # texture frobenius
    profile_elcap_texture_Frob = compute_average_profiles("/Volumes/sandisk/cycleGAN_results/metrics_ep50_cycleGAN", path_1="metrics_texture_Frob_10/raps_elcap_complete_epoch70.json",
                                                         path_2="metrics_texture_Frob_11/raps_elcap_complete_epoch70.json",
                                                         path_3="metrics_texture_Frob_12/raps_elcap_complete_epoch70.json")

    profile_test_3_texture_Frob = compute_average_profiles("/Volumes/sandisk/cycleGAN_results/metrics_ep50_cycleGAN", path_1="metrics_texture_Frob_10/raps_test_3_epoch70.json",
                                                          path_2="metrics_texture_Frob_11/raps_test_3_epoch70.json",
                                                          path_3="metrics_texture_Frob_12/raps_test_3_epoch70.json")
    # texture attention
    profile_elcap_texture_att = compute_average_profiles("/Volumes/sandisk/cycleGAN_results/metrics_ep50_cycleGAN", path_1="metrics_texture_att_13/raps_elcap_complete_epoch70.json",
                                                         path_2="metrics_texture_att_14/raps_elcap_complete_epoch70.json",
                                                         path_3="metrics_texture_att_15/raps_elcap_complete_epoch70.json")

    profile_test_3_texture_att = compute_average_profiles("/Volumes/sandisk/cycleGAN_results/metrics_ep50_cycleGAN", path_1="metrics_texture_att_13/raps_test_3_epoch70.json",
                                                          path_2="metrics_texture_att_14/raps_test_3_epoch70.json",
                                                          path_3="metrics_texture_att_15/raps_test_3_epoch70.json")
    # perceptual
    profile_elcap_perceptual = compute_average_profiles("/Volumes/sandisk/cycleGAN_results/metrics_ep50_cycleGAN", path_1="metrics_perceptual_10/raps_elcap_complete_epoch70.json",
                                                         path_2="metrics_perceptual_11/raps_elcap_complete_epoch70.json",
                                                         path_3="metrics_perceptual_12/raps_elcap_complete_epoch70.json")

    profile_test_3_perceptual = compute_average_profiles("/Volumes/sandisk/cycleGAN_results/metrics_ep50_cycleGAN", path_1="metrics_perceptual_10/raps_test_3_epoch70.json",
                                                          path_2="metrics_perceptual_11/raps_test_3_epoch70.json",
                                                          path_3="metrics_perceptual_12/raps_test_3_epoch70.json")


    '''for i in range(0, 7, 1):
        if i == 5:
            pass
        else:
            average_1 = compute_results_2(
                '/Users/francescodifeola/Desktop/downloads_alvis/window_training/metrics_pix2pix_baseline_window_4/metrics_test_1_epoch50.csv',
                '/Users/francescodifeola/Desktop/downloads_alvis/window_training/metrics_pix2pix_baseline_window_5/metrics_test_1_epoch50.csv',
                '/Users/francescodifeola/Desktop/downloads_alvis/window_training/metrics_pix2pix_baseline_window_6/metrics_test_1_epoch50.csv',
                'baseline',
                i)

            average_2 = compute_results_2(
                '/Users/francescodifeola/Desktop/downloads_alvis/window_training/metrics_pix2pix_texture_Frob_window_4/metrics_test_1_epoch50.csv',
                '/Users/francescodifeola/Desktop/downloads_alvis/window_training/metrics_pix2pix_texture_Frob_window_5/metrics_test_1_epoch50.csv',
                '/Users/francescodifeola/Desktop/downloads_alvis/window_training/metrics_pix2pix_texture_Frob_window_6/metrics_test_1_epoch50.csv',
                'baseline',
                i)

            compute_wilcoxon_test(average_1, average_2)'''

    '''done = False
    if not done:
        print('ciao')
    n = [4, 6]
    for i in range(7):
        if i == 9:
            pass
        else:
            average = compute_results_2(
                '/Users/francescodifeola/Desktop/downloads_alvis/homogenization/metrics_hom_texture_att_1/metrics_test_1_epoch100.csv',
                '/Users/francescodifeola/Desktop/downloads_alvis/homogenization/metrics_hom_texture_att_2/metrics_test_1_epoch100.csv',
                '/Users/francescodifeola/Desktop/downloads_alvis/homogenization/metrics_hom_texture_att_3/metrics_test_1_epoch100.csv',
                'baseline',
                i)'''


    '''avg_fid, std = avg_fid(
        '/Volumes/Untitled/test_pix2pix_texture_max_window_4/fid_test_2.csv',
        '/Volumes/Untitled/test_pix2pix_texture_max_window_5/fid_test_2.csv',
        '/Volumes/Untitled/test_pix2pix_texture_max_window_6/fid_test_2.csv',
    )
    print(avg_fid, std)'''

    """
    print("Pix2Pix")
    compute_results('/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_baseline_nw_1/metrics_test_2_epoch200.csv',
                    '/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_baseline_nw_2/metrics_test_2_epoch200.csv',
                    '/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_baseline_nw_3/metrics_test_2_epoch200.csv',
                    'baseline',
                    1)
    compute_results('/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_texture_max_nw_1/metrics_test_2_epoch200.csv',
                    '/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_texture_max_nw_2/metrics_test_2_epoch200.csv',
                    '/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_texture_max_nw_3/metrics_test_2_epoch200.csv',
                    'texture_max',
                    1)
    compute_results('/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_texture_avg_nw_1/metrics_test_2_epoch200.csv',
                    '/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_texture_avg_nw_2/metrics_test_2_epoch200.csv',
                    '/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_texture_avg_nw_3/metrics_test_2_epoch200.csv',
                    'texture_average',
                    1)
    compute_results('/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_texture_Frob_nw_1/metrics_test_2_epoch200.csv',
                    '/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_texture_Frob_nw_2/metrics_test_2_epoch200.csv',
                    '/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_texture_Frob_nw_3/metrics_test_2_epoch200.csv',
                    'texture_Frobenius',
                    1)
    compute_results('/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_perceptual_nw_1/metrics_test_2_epoch200.csv',
                    '/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_perceptual_nw_2/metrics_test_2_epoch200.csv',
                    '/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_perceptual_nw_3/metrics_test_2_epoch200.csv',
                    'perceptual',
                    1)

    compute_results('/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_baseline_nw_1/metrics_test_2_epoch200.csv',
                    '/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_baseline_nw_2/metrics_test_2_epoch200.csv',
                    '/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_baseline_nw_3/metrics_test_2_epoch200.csv',
                    'baseline',
                    0)
    compute_results('/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_texture_max_nw_1/metrics_test_2_epoch200.csv',
                    '/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_texture_max_nw_2/metrics_test_2_epoch200.csv',
                    '/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_texture_max_nw_3/metrics_test_2_epoch200.csv',
                    'texture_max',
                    0)
    compute_results('/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_texture_avg_nw_1/metrics_test_2_epoch200.csv',
                    '/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_texture_avg_nw_2/metrics_test_2_epoch200.csv',
                    '/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_texture_avg_nw_3/metrics_test_2_epoch200.csv',
                    'texture_average',
                    0)
    compute_results('/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_texture_Frob_nw_1/metrics_test_2_epoch200.csv',
                    '/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_texture_Frob_nw_2/metrics_test_2_epoch200.csv',
                    '/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_texture_Frob_nw_3/metrics_test_2_epoch200.csv',
                    'texture_Frobenius',
                    0)
    compute_results('/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_perceptual_nw_1/metrics_test_2_epoch200.csv',
                    '/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_perceptual_nw_2/metrics_test_2_epoch200.csv',
                    '/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_perceptual_nw_3/metrics_test_2_epoch200.csv',
                    'perceptual',
                    0)
    print("CycleGAN")
    compute_results(
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_baseline_nw_10/metrics_test_2_epoch200.csv',
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_baseline_nw_11/metrics_test_2_epoch200.csv',
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_baseline_nw_12/metrics_test_2_epoch200.csv',
        'baseline',
        1)
    compute_results(
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_texture_max_nw_1/metrics_test_2_epoch200.csv',
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_texture_max_nw_2/metrics_test_2_epoch200.csv',
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_texture_max_nw_3/metrics_test_2_epoch200.csv',
        'texture_max',
        1)
    compute_results(
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_texture_avg_nw_1/metrics_test_2_epoch200.csv',
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_texture_avg_nw_2/metrics_test_2_epoch200.csv',
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_texture_avg_nw_3/metrics_test_2_epoch200.csv',
        'texture_average',
        1)
    compute_results(
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_texture_frob_nw_1/metrics_test_2_epoch200.csv',
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_texture_frob_nw_2/metrics_test_2_epoch200.csv',
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_texture_frob_nw_3/metrics_test_2_epoch200.csv',
        'texture_Frobenius',
        1)
    compute_results(
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_perceptual_nw_1/metrics_test_2_epoch200.csv',
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_perceptual_nw_2/metrics_test_2_epoch200.csv',
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_perceptual_nw_3/metrics_test_2_epoch200.csv',
        'perceptual',
        1)
    compute_results(
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_baseline_nw_10/metrics_test_2_epoch200.csv',
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_baseline_nw_11/metrics_test_2_epoch200.csv',
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_baseline_nw_12/metrics_test_2_epoch200.csv',
        'baseline',
        0)
    compute_results(
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_texture_max_nw_1/metrics_test_2_epoch200.csv',
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_texture_max_nw_2/metrics_test_2_epoch200.csv',
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_texture_max_nw_3/metrics_test_2_epoch200.csv',
        'texture_max',
        0)
    compute_results(
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_texture_avg_nw_1/metrics_test_2_epoch200.csv',
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_texture_avg_nw_2/metrics_test_2_epoch200.csv',
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_texture_avg_nw_3/metrics_test_2_epoch200.csv',
        'texture_average',
        0)
    compute_results(
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_texture_frob_nw_1/metrics_test_2_epoch200.csv',
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_texture_frob_nw_2/metrics_test_2_epoch200.csv',
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_texture_frob_nw_3/metrics_test_2_epoch200.csv',
        'texture_Frobenius',
        0)
    compute_results(
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_perceptual_nw_1/metrics_test_2_epoch200.csv',
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_perceptual_nw_2/metrics_test_2_epoch200.csv',
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_perceptual_nw_3/metrics_test_2_epoch200.csv',
        'perceptual',
        0)
    """

    '''# Set the font size
    font_size = 12

    # Create the figure and subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Flatten the axs array to make it easier to work with
    axs = axs.flatten()

    # Loop through the subplots and plot the scatter plots
    for i, ax in enumerate(axs):
        if i==0:
            ax.scatter([0.00193, 0.00148, 0.00205, 0.00243, 0.00163], [4.47735, 4.32139, 4.47464, 4.51363, 4.48758])
            ax.annotate('baseline', (0.00193, 4.47735))
            ax.annotate('Texture max', (0.00148, 4.32139))
            ax.annotate('Texture average', (0.00205, 4.47464))
            ax.annotate('Texture frobenius', (0.00243, 4.51363))
            ax.annotate('Perceptual', (0.00163, 4.48758))
            ax.legend(['CycleGAN'], fontsize=font_size)
            ax.set_xlabel('MSE', fontsize=font_size)
            ax.set_ylabel('NIQE', fontsize=font_size)
            ax.grid()
            ax.tick_params(axis='both', labelsize=font_size)
        elif i == 1:
            ax.scatter([0.00072, 0.00068, 0.00070, 0.00067, 0.00071], [4.54087, 4.56291, 4.53187, 4.44389, 4.55222])
            ax.annotate('baseline', (0.00072, 4.54087))
            ax.annotate('Texture max', (0.00068, 4.56291))
            ax.annotate('Texture average', (0.00070, 4.53187))
            ax.annotate('Texture frobenius', (0.00067, 4.44389))
            ax.annotate('Perceptual', (0.00071, 4.55222))
            ax.legend(['Pix2Pix'], fontsize=font_size)
            ax.set_xlabel('MSE', fontsize=font_size)
            ax.set_ylabel('NIQE', fontsize=font_size)
            ax.grid()
            ax.tick_params(axis='both', labelsize=font_size)
        elif i == 2:
            ax.scatter([0.00193, 0.00148, 0.00205, 0.00243, 0.00163], [34.19983, 32.59030, 35.49717, 36.07929, 28.93616])
            ax.annotate('baseline', (0.00193, 34.19983))
            ax.annotate('Texture max', (0.00148, 32.59030))
            ax.annotate('Texture average', (0.00205, 35.49717))
            ax.annotate('Texture frobenius', (0.00243, 36.07929))
            ax.annotate('Perceptual', (0.00163, 28.93616))
            ax.legend(['CycleGAN'], fontsize=font_size)
            ax.set_xlabel('MSE', fontsize=font_size)
            ax.set_ylabel('FID', fontsize=font_size)
            ax.grid()
            ax.tick_params(axis='both', labelsize=font_size)
        elif i == 3:
            ax.scatter([0.00072, 0.00068, 0.00070, 0.00067, 0.00071], [44.53723, 44.86776, 46.36452, 48.32258, 48.63981])
            ax.annotate('baseline', (0.00072, 44.53723))
            ax.annotate('Texture max', (0.00068, 44.86776))
            ax.annotate('Texture average', (0.00070, 46.36452))
            ax.annotate('Texture frobenius', (0.00067, 48.32258))
            ax.annotate('Perceptual', (0.00071, 48.63981))
            ax.legend(['Pix2Pix'], fontsize=font_size)
            ax.set_xlabel('MSE', fontsize=font_size)
            ax.set_ylabel('FID', fontsize=font_size)
            ax.grid()
            ax.tick_params(axis='both', labelsize=font_size)


    plt.suptitle('Average metrics at epoch 200 over Test 1', fontsize=12)
    # Adjust the spacing between subplots
    plt.tight_layout()

    # Display the plot
    plt.show()
    plt.figure()

    # Create the figure and subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Flatten the axs array to make it easier to work with
    axs = axs.flatten()

    # Loop through the subplots and plot the scatter plots
    for i, ax in enumerate(axs):
        if i == 0:
            ax.scatter([0.00457, 0.00443, 0.00453, 0.00725, 0.00502], [7.13569, 6.95414, 7.02405, 7.36896, 7.54868])
            ax.annotate('baseline', (0.00457, 7.13569))
            ax.annotate('Texture max', (0.00443, 6.95414))
            ax.annotate('Texture average', (0.00453, 7.02405))
            ax.annotate('Texture frobenius', (0.00725, 7.36896))
            ax.annotate('Perceptual', (0.00502, 7.54868))
            ax.legend(['CycleGAN'], fontsize=font_size)
            ax.set_xlabel('MSE', fontsize=font_size)
            ax.set_ylabel('NIQE', fontsize=font_size)
            ax.grid()
            ax.tick_params(axis='both', labelsize=font_size)
        elif i == 1:
            ax.scatter([0.00987, 0.00986, 0.00998, 0.00983, 0.00974], [6.41072, 6.34904, 6.44700, 6.35169, 6.34527])
            ax.annotate('baseline', (0.00987, 6.41072))
            ax.annotate('Texture max', (0.00986, 6.34904))
            ax.annotate('Texture average', (0.00998, 6.44700))
            ax.annotate('Texture frobenius', (0.00983, 6.35169))
            ax.annotate('Perceptual', (0.00974, 6.34527))
            ax.legend(['Pix2Pix'], fontsize=font_size)
            ax.set_xlabel('MSE', fontsize=font_size)
            ax.set_ylabel('NIQE', fontsize=font_size)
            ax.grid()
            ax.tick_params(axis='both', labelsize=font_size)
        elif i == 2:
            ax.scatter([0.00457, 0.00443, 0.00453, 0.00725, 0.00502],
                       [32.72679, 25.27028, 32.36042, 34.06509, 23.67524])
            ax.annotate('baseline', (0.00457, 32.72679))
            ax.annotate('Texture max', (0.00443, 25.27028))
            ax.annotate('Texture average', (0.00453, 32.36042))
            ax.annotate('Texture frobenius', (0.00725, 34.06509))
            ax.annotate('Perceptual', (0.00502, 23.67524))
            ax.legend(['CycleGAN'], fontsize=font_size)
            ax.set_xlabel('MSE', fontsize=font_size)
            ax.set_ylabel('FID', fontsize=font_size)
            ax.grid()
            ax.tick_params(axis='both', labelsize=font_size)
        elif i == 3:
            ax.scatter([0.00987, 0.00986, 0.00998, 0.00983, 0.00974],
                       [35.86163, 35.47996, 35.40051, 35.10810, 36.70606])
            ax.annotate('baseline', (0.00987, 35.86163))
            ax.annotate('Texture max', (0.00986, 35.47996))
            ax.annotate('Texture average', (0.00998, 35.40051))
            ax.annotate('Texture frobenius', (0.00983, 35.10810))
            ax.annotate('Perceptual', (0.00974, 36.70606))
            ax.legend(['Pix2Pix'], fontsize=font_size)
            ax.set_xlabel('MSE', fontsize=font_size)
            ax.set_ylabel('FID', fontsize=font_size)
            ax.grid()
            ax.tick_params(axis='both', labelsize=font_size)

    plt.suptitle('Average metrics at epoch 200 over Test 2', fontsize=12)
    # Adjust the spacing between subplots
    plt.tight_layout()

    # Display the plot
    plt.show()
    plt.figure()'''

    '''########################

    texture_max_1 = pd.read_csv(
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_texture_max_nw_1/metrics_test_1_epoch200.csv',
        header=None)
    texture_max_2 = pd.read_csv(
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_texture_max_nw_2/metrics_test_1_epoch200.csv',
        header=None)
    texture_max_3 = pd.read_csv(
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_texture_max_nw_3/metrics_test_1_epoch200.csv',
        header=None)

    texture_max_1 = convert_string_to_floats(texture_max_1.iloc[1][1])
    texture_max_2 = convert_string_to_floats(texture_max_2.iloc[1][1])
    texture_max_3 = convert_string_to_floats(texture_max_3.iloc[1][1])

    texture_max_avg_1 = np.mean(texture_max_1)
    texture_max_avg_2 = np.mean(texture_max_2)
    texture_max_avg_3 = np.mean(texture_max_3)
    overall_texture_max = (texture_max_avg_1 + texture_max_avg_2 + texture_max_avg_3) / 3
    std = np.std(np.array([texture_max_avg_1, texture_max_avg_2, texture_max_avg_3]))

    print(f'Texture_max1: {texture_max_avg_1}')
    print(f'Texture_max2: {texture_max_avg_2}')
    print(f'Texture_max3: {texture_max_avg_3}')
    print(f'Overall: {overall_texture_max}, Std: {std}')

    #########################

    texture_avg_1 = pd.read_csv(
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_texture_avg_nw_1/metrics_test_1_epoch200.csv',
        header=None)
    texture_avg_2 = pd.read_csv(
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_texture_avg_nw_2/metrics_test_1_epoch200.csv',
        header=None)
    texture_avg_3 = pd.read_csv(
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_texture_avg_nw_3/metrics_test_1_epoch200.csv',
        header=None)

    texture_avg_1 = convert_string_to_floats(texture_avg_1.iloc[1][1])
    texture_avg_2 = convert_string_to_floats(texture_avg_2.iloc[1][1])
    texture_avg_3 = convert_string_to_floats(texture_avg_3.iloc[1][1])

    texture_avg_avg_1 = np.mean(texture_avg_1)
    texture_avg_avg_2 = np.mean(texture_avg_2)
    texture_avg_avg_3 = np.mean(texture_avg_3)
    overall_texture_avg = (texture_avg_avg_1 + texture_avg_avg_2 + texture_avg_avg_3) / 3
    std = np.std(np.array([texture_avg_avg_1, texture_avg_avg_2, texture_avg_avg_3]))

    print(f'Texture_avg1: {texture_avg_avg_1}')
    print(f'Texture_avg2: {texture_avg_avg_2}')
    print(f'Texture_avg3: {texture_avg_avg_3}')
    print(f'Overall: {overall_texture_avg}, Std: {std}')

    ##############################

    texture_frob_1 = pd.read_csv(
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_texture_Frob_nw_1/metrics_test_1_epoch200.csv',
        header=None)
    texture_frob_2 = pd.read_csv(
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_texture_Frob_nw_2/metrics_test_1_epoch200.csv',
        header=None)
    texture_frob_3 = pd.read_csv(
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_texture_Frob_nw_3/metrics_test_1_epoch200.csv',
        header=None)

    texture_frob_1 = convert_string_to_floats(texture_frob_1.iloc[1][1])
    texture_frob_2 = convert_string_to_floats(texture_frob_2.iloc[1][1])
    texture_frob_3 = convert_string_to_floats(texture_frob_3.iloc[1][1])

    texture_frob_avg_1 = np.mean(texture_frob_1)
    texture_frob_avg_2 = np.mean(texture_frob_2)
    texture_frob_avg_3 = np.mean(texture_frob_3)
    overall_texture_frob = (texture_frob_avg_1 + texture_frob_avg_2 + texture_frob_avg_3) / 3
    std = np.std(np.array([texture_frob_avg_1, texture_frob_avg_2, texture_frob_avg_3]))

    print(f'Texture_frob1: {texture_frob_avg_1}')
    print(f'Texture_frob2: {texture_frob_avg_2}')
    print(f'Texture_frob3: {texture_frob_avg_3}')
    print(f'Overall: {overall_texture_frob}, Std: {std}')
    # average_baseline = [(x + y) / 2 for x, y in zip(baseline_1, baseline_2, baseline_3)]

    plt.figure()
    # plt.scatter([31.9714, 31.8105, 31.8427, 32.1373, 31.8319], [0.9626, 0.9594, 0.9580, 0.9614, 0.9588], marker='o')
    plt.scatter([0.00072, 0.00068, 0.0007, 0.00067], [4.5409, 4.5629, 4.5319, 4.4439], marker='o',
                color='orange')
    plt.annotate('baseline', (0.00072, 4.5409))
    plt.annotate('Texture max', (0.00068, 4.5629))
    plt.annotate('Texture avg', (0.0007, 4.5319))
    plt.annotate('Texture frob', (0.00067, 4.4439))

    # plt.ylim([0.925, 1.02])
    plt.legend(['Pix2Pix'])
    plt.ylabel("NIQE")
    plt.xlabel("MSE")
    plt.grid()
    plt.title('Average metrics at epoch 200 over 1 test patient (Mayo)')
    plt.show()'''

    """baseline_1 = pd.read_csv(
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_baseline_wind_1/metrics_test_1.csv', header=None)
    baseline_2 = pd.read_csv(
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_baseline_wind_2/metrics_test_1.csv', header=None)
    baseline_3 = pd.read_csv(
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_baseline_wind_3/metrics_test_1.csv', header=None)

    baseline_1 = convert_string_to_floats(baseline_1.iloc[0][1])
    baseline_2 = convert_string_to_floats(baseline_2.iloc[0][1])
    baseline_3 = convert_string_to_floats(baseline_3.iloc[0][1])

    average_baseline = [(x + y) / 2 for x, y in zip(baseline_1, baseline_3)]
    print(f'Baseline: {average_baseline[-1]}')

    texture_max_1 = pd.read_csv(
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_texture_max_wind_1/metrics_test_1.csv',
        header=None)
    texture_max_2 = pd.read_csv(
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_texture_max_wind_2/metrics_test_1.csv',
        header=None)
    texture_max_3 = pd.read_csv(
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_texture_max_wind_3/metrics_test_1.csv',
        header=None)

    texture_max_1 = convert_string_to_floats(texture_max_1.iloc[0][1])
    texture_max_2 = convert_string_to_floats(texture_max_2.iloc[0][1])
    texture_max_3 = convert_string_to_floats(texture_max_3.iloc[0][1])


    average_texture_max = [(x + y + z) / 3 for x, y, z in zip(texture_max_1, texture_max_2, texture_max_3)]
    print(baseline_3)
    print(f'Texture max: {average_texture_max[-1]}')

    plt.figure()
    plt.plot(range(1, 201), baseline_1)
    plt.plot(range(1, 201), average_texture_max)
    plt.legend(['baseline', 'texture_max'])
    plt.title('PSNR over epochs on the Mayo test (1 patient)')
    plt.show()"""

    # texture_norm_1 = pd.read_csv(
    #    '/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_texture_normalized_1/metrics_epoch200.csv',
    #    header=None)
    # texture_norm_2 = pd.read_csv(
    #     '/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_texture_normalized_2/metrics_epoch200.csv',
    #     header=None)
    # texture_norm_3 = pd.read_csv(
    #     '/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_texture_normalized_3/metrics_epoch200.csv',
    #     header=None)

    # texture_norm_1 = convert_string_to_floats(texture_norm_1.iloc[0][1])
    # texture_norm_2 = convert_string_to_floats(texture_norm_2.iloc[0][1])
    # texture_norm_3 = convert_string_to_floats(texture_norm_3.iloc[0][1])

    # average_texture_norm = [(x + y + z) / 3 for x, y, z in zip(texture_norm_1, texture_norm_2, texture_norm_3)]

    # print(f'Texture norm: {average_texture_norm[-1]}')

    # texture_averaged_d5_1 = pd.read_csv(
    #     '/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_texture_offset5_1/metrics_epoch200.csv',
    #     header=None)
    # texture_averaged_d5_2 = pd.read_csv(
    #     '/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_texture_offset5_2/metrics_epoch200.csv',
    #     header=None)
    # texture_averaged_d5_3 = pd.read_csv(
    #     '/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_texture_offset5_3/metrics_epoch200.csv',
    #     header=None)

    # texture_averaged_d5_1 = convert_string_to_floats(texture_averaged_d5_1.iloc[0][1])
    # texture_averaged_d5_2 = convert_string_to_floats(texture_averaged_d5_2.iloc[0][1])
    # texture_averaged_d5_3 = convert_string_to_floats(texture_averaged_d5_3.iloc[0][1])

    # average_texture_averaged_d5 = [(x + y + z) / 3 for x, y, z in
    #                               zip(texture_averaged_d5_1, texture_averaged_d5_2, texture_averaged_d5_3)]

    # print(f'Texture avg d5: {average_texture_averaged_d5[-1]}')

    # perceptual_1 = pd.read_csv(
    #     '/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_perceptual_1/metrics_epoch200.csv',
    #     header=None)
    # perceptual_2 = pd.read_csv(
    #     '/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_perceptual_2/metrics_epoch200.csv',
    #     header=None)
    # perceptual_3 = pd.read_csv(
    #     '/Users/francescodifeola/Desktop/downloads_alvis/metrics_pix2pix_perceptual_3/metrics_epoch200.csv',
    #     header=None)

    # perceptual_1 = convert_string_to_floats(perceptual_1.iloc[0][1])
    # perceptual_2 = convert_string_to_floats(perceptual_2.iloc[0][1])
    # perceptual_3 = convert_string_to_floats(perceptual_3.iloc[0][1])

    # average_perceptual = [(x + y + z) / 3 for x, y, z in zip(perceptual_1, perceptual_2, perceptual_3)]

    # print(f'Perceptual: {average_perceptual[-1]}')

    '''perceptual_l45_1 = pd.read_csv(
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_perceptual_l45_1/metrics_test.csv',
        header=None)
    perceptual_l45_2 = pd.read_csv(
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_perceptual_l45_2/metrics_test.csv',
        header=None)
    perceptual_l45_3 = pd.read_csv(
        '/Users/francescodifeola/Desktop/downloads_alvis/metrics_perceptual_l45_3/metrics_test.csv',
        header=None)

    perceptual_l45_1 = convert_string_to_floats(perceptual_l45_1.iloc[2][1])
    perceptual_l45_2 = convert_string_to_floats(perceptual_l45_2.iloc[2][1])
    perceptual_l45_3 = convert_string_to_floats(perceptual_l45_3.iloc[2][1])

    average_perceptual_l45 = [(x + y + z) / 3 for x, y, z in zip(perceptual_l45_1, perceptual_l45_2, perceptual_l45_3)]

    print(f'Perceptual_l45: {average_perceptual_l45[-1]}')

    # compute_wilcoxon_test(average_texture_norm, average_texture_max)
    plt.figure()
    # plt.scatter([31.9714, 31.8105, 31.8427, 32.1373, 31.8319], [0.9626, 0.9594, 0.9580, 0.9614, 0.9588], marker='o')
    plt.scatter([29.2322, 30.6947, 29.3731, 30.7706, 29.7606], [1.0021, 1.0044, 0.9393, 1.0165, 0.9564], marker='o', color='orange')
    plt.annotate('baseline', (31.9714, 0.9626))
    plt.annotate('Texture max', (31.8105, 0.9594))
    plt.annotate('Texture norm', (31.8427, 0.9580))
    plt.annotate('Texture avg d5', (32.1373, 0.9614))
    plt.annotate('Perceptual ImNet', (31.8319, 0.9588))
    plt.annotate('baseline', (29.2322, 1.0021))
    plt.annotate('Texture max', (30.6947, 1.0044))
    plt.annotate('Texture norm', (29.3731, 0.9393))
    plt.annotate('Texture avg d5', (30.7706, 1.0165))
    plt.annotate('Perceptual ImNet', (29.7606, 0.9564))
    #plt.ylim([0.925, 1.02])
    plt.legend(['CycleGAN'])
    plt.ylabel("VIF")
    plt.xlabel("PSNR")
    plt.grid()
    plt.title('Average metrics at epoch 200 over 1 test patient (Mayo)')
    plt.show()
    plt.figure()
    plt.plot(range(1, 201), average_baseline)
    plt.plot(range(1, 201), average_texture_max)
    plt.plot(range(1, 201), average_texture_norm)
    plt.plot(range(1, 201), average_texture_averaged_d5)
    plt.plot(range(1, 201), average_perceptual)
    # plt.plot(range(1, 201), average_perceptual_l45)
    # plt.text(110, average_baseline[0], f'Baseline epoch 200: {round(average_baseline[-1], 4)}', fontsize=10, color='black')
    # plt.text(110, average_baseline[0]-0.01, f'Texture max epoch 200: {round(average_texture_max[-1], 4)}', fontsize=10, color='black')
    # plt.text(110, average_baseline[0] - 0.02, f'Texture norm epoch 200: {round(average_texture_norm[-1], 4)}', fontsize=10, color='black')
    # plt.text(110, average_baseline[0] - 0.03, f'Texture norm epoch 200: {round(average_texture_averaged_d5[-1], 4)}', fontsize=10, color='black')
    # plt.text(110, average_baseline[0]-0.04, f'Perceptual epoch 200: {round(average_perceptual[-1], 4)}', fontsize=10, color='black')
    plt.legend(['baseline', 'texture_max', 'texture_norm', 'texture_averaged_d5', 'perceptual_ImNet'])
    plt.title('VIF over epochs on the Mayo test (1 patient)')
    plt.show()

    # Define a function to convert string to tensor
    def convert_to_tensor(string):
        if string.startswith('tensor(') and string.endswith(')'):
            tensor_str = string.replace('tensor(', '').replace(')', '')
            return torch.tensor(eval(tensor_str))
        else:
            return string

    exp_1 = pd.read_csv('/Users/francescodifeola/Desktop/downloads_alvis/loss_texture_8/idx_texture_B.csv')
    id_d_1 = exp_1['id_d'].map(convert_to_tensor)
    id_theta_1 = exp_1['id_theta'].map(convert_to_tensor)



    id_d = list()
    id_theta = list()
    for i in id_d_1:
        if len(i) > 1:
            id_d.append(i[0].item())
        else:
            id_d.append(i.item())

    for j in id_theta_1:
        if len(j) > 1:
            id_theta.append(j[0].item())
        else:
            id_theta.append(j.item())

    histogram = {
        '1-0°': 0,
        '1-45°': 0,
        '1-90°': 0,
        '1-135°': 0,
         '3-0°': 0,
        '3-45°': 0,
        '3-90°': 0,
        '3-135°': 0,
        '5-0°': 0,
        '5-45°': 0,
        '5-90°': 0,
        '5-135°': 0,
        '7-0°': 0,
        '7-45°': 0,
        '7-90°': 0,
        '7-135°': 0,
    }
    for k in zip(id_d, id_theta):
          if k == (0, 0):
              histogram['1-0°'] += 1
          elif  k == (0, 1):
              histogram['1-45°'] += 1
          elif  k == (0, 2):
              histogram['1-90°'] += 1
          elif  k == (0, 3):
              histogram['1-135°'] += 1
          elif  k == (1, 0):
              histogram['3-0°'] += 1
          elif  k == (1, 1):
              histogram['3-45°'] += 1
          elif  k == (1, 2):
              histogram['3-90°'] += 1
          elif  k == (1, 3):
              histogram['3-135°'] += 1
          elif  k == (2, 0):
              histogram['5-0°'] += 1
          elif  k == (2, 1):
              histogram['5-45°'] += 1
          elif  k == (2, 2):
              histogram['5-90°'] += 1
          elif  k == (2, 3):
              histogram['5-135°'] += 1
          elif  k == (3, 0):
              histogram['7-0°'] += 1
          elif  k == (3, 1):
              histogram['7-45°'] += 1
          elif  k == (3, 2):
              histogram['7-90°'] += 1
          elif  k == (3, 3):
             histogram['7-135°'] += 1

    plt.figure()
    plt.bar(histogram.keys(), histogram.values())
    plt.show()'''
