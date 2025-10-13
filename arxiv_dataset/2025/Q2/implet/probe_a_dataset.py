import argparse

import numpy as np
import warnings
import pandas as pd

from main import MainExperiment
from utils import *

warnings.filterwarnings("ignore")

torch.manual_seed(42)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Specify datasets')

    parser.add_argument('--dataset_name', type=str, default='GunPoint', help='dataset_name')
    parser.add_argument('--verbose', type=bool, default=False, help='To show more info or not')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='On which device')

    args = parser.parse_args()
    dataset = args.dataset_name
    verbose = args.verbose
    random_seed = args.random_seed
    device = args.device
    xai_name = 'IntegratedGradients'
    dataset_results = []
    target_class = None  # attributions's class
    sample_filter_by_pred_y = True  # either filter by ground truth or prediction
    feature_shapelet = None  # for a real dataset we don't have a ground truth

    exp_path = 'experiments/' + dataset
    model_dataset_path = os.path.join('experiments/' + dataset, 'model')
    attr_save_dir = os.path.join(exp_path, 'attr', f'class_{str(target_class)}/{xai_name}/exp.pkl')

    se = MainExperiment(feature_shapelet,
                        model_dataset_path,
                        attr_save_dir,
                        probe_dir=None,
                        target_class=target_class,
                        sample_filter_class=None,
                        sample_filter_by_pred_y=sample_filter_by_pred_y,
                        seed=random_seed, device=device, model_type='FCN', xai_name=xai_name,
                        train_model=False,  # True,
                        dataset_name=dataset,
                        verbose=verbose
                        )
    for sample_filter_class in [0, 1]:

        if sample_filter_class is None:
            sample_filter_type = 'all_samples'
        else:
            sample_filter_type = f'sample_{"pred" if sample_filter_by_pred_y else "true"}_{sample_filter_class}'
        probe_dir = os.path.join(exp_path, 'probe', f'class_{str(target_class)}_{sample_filter_type}/{xai_name}/')

        se.probe_dir = probe_dir
        se.sample_filter_class = sample_filter_class
        implets, implet_cluster_results = se.get_implets(dataset, is_read_implet=True, lamb=0.1, thresh_factor=1.5)
        attrs = se.get_attr(is_read=True, is_plot=False)

        implets_length_list = [implet[-1] - implet[-2] + 1 for implet in implets]
        avg_implet_length = np.mean(implets_length_list)  # average length
        inst = se.test_x[se.test_y == 0]

        attr = attrs[se.test_y == 0]

        seen_nonfeature, _ = se.get_seen_nonfeature(inst, attr, int(avg_implet_length))
        print('capturing non_feature')
        print('avg_implet_length:', avg_implet_length, 'seen_nonfeature:', seen_nonfeature.shape)
        print('Start probing')
        probe_choices = [
                         'nonfeature',
                         'implet_cluster', 'implet_cluster_untrained',
                         'one_implet', 'one_implet_untrained',
                         'implet_centroid', 'implet_centroid_untrained',
                        ]
        # probe_choices = ['implet_centroid', 'implet_centroid_untrained']
        for probe_choice in probe_choices:
            if probe_choice == 'implet_cluster' or probe_choice == 'implet_cluster_untrained':
                print(f'-----------------------------')
                print(f'{probe_choice}')
                is_model_untrained = True if (probe_choice == 'implet_cluster_untrained') else False
                prob_results, tcav_scores, cf_scores = se.probe_clusters('s1', probe_dir + 'implet_cluster',
                                                                         is_read_pdata=False,
                                                                         is_read_implet=True,
                                                                         is_model_untrained=is_model_untrained)
                for i in range(len(prob_results)):
                    prob_result = prob_results[i]
                    train_acc = prob_result['train_acc']
                    test_acc = prob_result['test_acc']
                    original_acc = prob_result['accuracy']
                    dataset_results.append(
                        [dataset + f'c_{sample_filter_class}', probe_choice + str(i), train_acc, test_acc, original_acc,
                         tcav_scores[i],
                         cf_scores[i]])
            elif probe_choice == 'implet_centroid' or probe_choice == 'implet_centroid_untrained' \
                    or probe_choice == 'one_implet' or probe_choice == 'one_implet_untrained':
                implets, implet_cluster_results = se.get_implets(implet_names=None, is_read_implet=True)
                cluster_indices = implet_cluster_results['best_indices_dep']
                cluster_centroid = implet_cluster_results['best_centroids_dep']

                is_model_untrained = True if 'untrained' in probe_choice else False
                print(is_model_untrained, probe_choice)
                for i, (cluster_index, cluster_instance) in enumerate(cluster_indices.items()):
                    print(f'-----------------------------')
                    print(f'{probe_choice}, cluster: {i}')
                    if 'centroid' in probe_choice:
                        insert_feature = cluster_centroid[i][:, 0]
                        print('implet cluster centroid', insert_feature.shape)
                    else:
                        clsuter_implets = [implets[i][1].flatten() for i in cluster_instance]
                        avg_length_c = int(np.mean(np.array([len(implet) for implet in clsuter_implets])))
                        # just have one that have avg length
                        index = np.argmin([np.abs(len(implet) - avg_length_c) for implet in clsuter_implets])
                        insert_feature = clsuter_implets[index]

                        print('Find a implet that has average length', insert_feature.shape)
                    prob_result, tcav_score, cf_score = se.probe_shapelet(insert_feature,
                                                                          probe_dir + probe_choice + str(i),
                                                                          is_read_pdata=False,
                                                                          shapelet_labels_ori=None,
                                                                          is_model_untrained=is_model_untrained)

                    train_acc = prob_result['train_acc']
                    test_acc = prob_result['test_acc']
                    original_acc = prob_result['accuracy']
                    dataset_results.append(
                        [dataset + f'c_{sample_filter_class}', probe_choice + str(i), train_acc, test_acc, original_acc,
                         tcav_score, cf_score]
                    )

            else:
                print(f'-----------------------------')
                print(f'{probe_choice}')
                if probe_choice == 'feature':
                    insert_feature = feature_shapelet
                    is_model_untrained = False
                elif probe_choice == 'nonfeature':
                    insert_feature = seen_nonfeature
                    is_model_untrained = False
                elif probe_choice == 'feature_untrained':
                    insert_feature = feature_shapelet
                    is_model_untrained = True

                else:
                    raise ValueError('illegal choice of probe')
                prob_result, tcav_score, cf_score = se.probe_shapelet(insert_feature, probe_dir + probe_choice,
                                                                      is_read_pdata=False,
                                                                      shapelet_labels_ori=None,
                                                                      is_model_untrained=is_model_untrained)
                train_acc = prob_result['train_acc']
                test_acc = prob_result['test_acc']
                original_acc = prob_result['accuracy']
                dataset_results.append(
                    [dataset + f'c_{sample_filter_class}', probe_choice, train_acc, test_acc, original_acc, tcav_score,
                     cf_score])

        dataset_results_df = pd.DataFrame(dataset_results)
        dataset_results_df.columns = ['dataset', 'probe_choice', 'train_acc', "test_acc", "original_acc", "tcav_score",
                                      "cf_score"]
        dataset_results_df.to_csv(f'./{exp_path}/df.csv')
