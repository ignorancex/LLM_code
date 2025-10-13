import numpy as np
import warnings
import pandas as pd

from main import MainExperiment
from utils import *

warnings.filterwarnings("ignore")

torch.manual_seed(42)


def create_zigzag_with_len_cycle(seq_len, cycle_len):
    """
    Generate a zigzag pattern for a sequence.

    Parameters:
        seq_len (int): Total sequence length.
        cycle_len (int): Length of one zigzag cycle (up and down).

    Returns:
        np.ndarray: An array containing the zigzag pattern.
    """
    # Ensure cycle_len is at least 2 to create a zigzag pattern
    if cycle_len < 2:
        raise ValueError("cycle_len must be at least 2.")

    # Generate one cycle of the zigzag (up and down)
    half_cycle = cycle_len // 2
    upward = np.arange(half_cycle) / half_cycle  # e.g., [0, 1, 2, ...]
    downward = np.arange(half_cycle, 0, -1) / half_cycle  # e.g., [2, 1]
    full_cycle = np.concatenate((upward, downward[:cycle_len - len(upward)]))

    # Tile the cycles to cover the full sequence length
    num_repeats = (seq_len + len(full_cycle) - 1) // len(full_cycle)
    zigzag = np.tile(full_cycle, num_repeats)[:seq_len]

    return zigzag


if __name__ == '__main__':
    s1 = np.sin(np.linspace(0, 2 * np.pi, 21)) + np.sin(np.linspace(0, 4 * np.pi, 21))
    s2 = np.cos(np.linspace(0, 2 * np.pi, 21)) * 2
    s3 = 0.5 * np.sin(np.linspace(0, 2 * np.pi, 21)) + 1.5 * np.sin(np.linspace(0, 4 * np.pi, 21))
    s4 = create_zigzag_with_len_cycle(21, 7) * 4 - 2
    s5 = create_zigzag_with_len_cycle(21, 5) * 4 - 2

    dataset_results = []
    datasets = {
        "s1": s1,
        "s2": s2,
        "s3": s3,
        "s4": s4,
        "s5": s5,
    }

    for i, (d, shapelet) in enumerate(datasets.items()):
        print(i, d)
        feature_shapelet = shapelet
        dataset = d

        xai_name = 'IntegratedGradients'
        exp_path = 'experiments/simu9/' + dataset

        target_class = None  # attributions's class
        sample_filter_class = 0  # implets class filter
        sample_filter_by_pred_y = True  # either filter by ground truth or prediction

        model_dataset_path = os.path.join(exp_path, 'model')
        attr_save_dir = os.path.join(exp_path, 'attr', f'class_{str(target_class)}/{xai_name}/exp.pkl')

        if sample_filter_class is None:
            sample_filter_type = 'all_samples'
        else:
            sample_filter_type = f'sample_{"pred" if sample_filter_by_pred_y else "true"}_{sample_filter_class}'
        probe_dir = os.path.join(exp_path, 'probe', f'class_{str(target_class)}_{sample_filter_type}/{xai_name}/')
        sample_filter_class = 0
        sample_filter_by_pred_y = True
        se = MainExperiment(feature_shapelet,
                            model_dataset_path,
                            attr_save_dir,
                            probe_dir,
                            target_class,
                            sample_filter_class,
                            sample_filter_by_pred_y,
                            seed=42, device='cuda', model_type='FCN', xai_name=xai_name,
                            train_model=False,  # True,
                            instance_length=150,
                            )

        implets, implet_cluster_results = se.get_implets(dataset, is_read_implet=True, lamb=0.1, thresh_factor=1)
        attrs = se.get_attr(is_read=True, is_plot=False)

        implets_length_list = [implet[-1] - implet[-2] + 1 for implet in implets]
        avg_implet_length = np.mean(implets_length_list)
        inst = se.test_x[se.test_y == 0]
        print(inst.shape)
        attr = attrs[se.test_y == 0]
        print(inst.shape, attrs.shape)
        seen_nonfeature, _ = se.get_seen_nonfeature(inst, attr, int(avg_implet_length))

        print(avg_implet_length, inst.shape, attr.shape, seen_nonfeature)
        probe_choices = ['feature', 'nonfeature', 'feature_untrained', 'implet_cluster', 'implet_cluster_untrained']
        for probe_choice in probe_choices:
            if probe_choice == 'implet_cluster' or probe_choice == 'implet_cluster_untrained':

                is_model_untrained = True if (probe_choice == 'implet_cluster_untrained') else False
                prob_results, tcav_scores, cf_scores = se.probe_clusters('s1', probe_dir + 'implet_cluster',
                                                                         is_read_pdata=False,
                                                                         is_read_implet=False,
                                                                         is_model_untrained=is_model_untrained)
                for i in range(len(prob_results)):
                    prob_result = prob_results[i]
                    train_acc = prob_result['train_acc']
                    test_acc = prob_result['test_acc']
                    original_acc = prob_result['accuracy']
                    dataset_results.append(
                        [dataset, probe_choice + str(i), train_acc, test_acc, original_acc, tcav_scores[i],
                         cf_scores[i]])
            else:
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
                prob_result, tcav_score, cf_score = se.probe_shapelet(insert_feature, probe_dir + 'feature',
                                                                      is_read_pdata=False,
                                                                      shapelet_labels_ori=None,
                                                                      is_model_untrained=is_model_untrained)
                train_acc = prob_result['train_acc']
                test_acc = prob_result['test_acc']
                original_acc = prob_result['accuracy']
                dataset_results.append([dataset, probe_choice, train_acc, test_acc, original_acc, tcav_score, cf_score])

    dataset_results_df = pd.DataFrame(dataset_results)
    dataset_results_df.columns = ['dataset', 'probe_choice', 'train_acc', "test_acc", "original_acc", "tcav_score",
                                  "cf_score"]
    dataset_results_df.to_csv(f'./{exp_path}/df.csv')
