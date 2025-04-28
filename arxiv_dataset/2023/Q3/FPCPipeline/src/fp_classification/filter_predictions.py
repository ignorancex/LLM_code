"""

Script to filter predictions from masks and reconstruct lesion matching graph after filtering to
re-calculate precision and recall

@author: Ishaan Bhat
@email: ishaan@isi.uu.nl

"""
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import joblib
from segmentation_metrics.lesion_correspondence import *
import os
from helper_functions import *
from utils.image_utils import read_itk_image

def filter_predictions_for_patient(seg, pat_df, model, pred_lesion_nodes=None, selected_features=None):


    assert(isinstance(seg, np.ndarray))

    # Get features matrix and slices
    features_matrix, _, _ = create_features_matrix(pat_df,
                                                   selected_features=selected_features)

    slices = pat_df['slice'].to_numpy()

    assert(features_matrix.ndim == 2) # 2 dimensions

    try:
        assert(features_matrix.shape[0] == slices.shape[0]) # Number of features == Number of slices
    except AssertionError:
        print('Number of slices != number of features')
        return None


    num_slices = len(slices)

    pred_labels = model.predict(features_matrix)

    filtered_pred = seg.copy()

    nodes_to_remove = []
    for idx in range(num_slices):
        lesion_slice = slices[idx]
        pred_label = pred_labels[idx]
        if pred_label == 1: # The slice has been classified as a false positive
            filtered_pred[lesion_slice] = 0
            removed_lesion_node = None
            for lesion_node in pred_lesion_nodes:
                if lesion_node.get_coordinates() == lesion_slice:
                    nodes_to_remove.append(lesion_node)
                    removed_lesion_node = lesion_node

            # Iterations get shorter as we progress with the filtering
            if removed_lesion_node is not None:
                pred_lesion_nodes.remove(removed_lesion_node)


    return filtered_pred, nodes_to_remove


def compute_detection_statistics(seg, gt):

    assert(isinstance(seg, np.ndarray))
    assert(isinstance(gt, np.ndarray))

    dgraph = create_correspondence_graph(seg, gt)

    lesion_counts = count_detections(dgraph,
                                     gt=gt,
                                     seg=seg)

    return lesion_counts


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--out_dir', type=str, help='Directory where the feature file is saved')
    parser.add_argument('--seed', type=int, help='Select trained model based on seed used to initialize the ERF')
    parser.add_argument('--images_dir', type=str, help='Directory where NN outputs are saved')
    parser.add_argument('--cross', action='store_true')
    parser.add_argument('--reduce_features', action='store_true')
    args = parser.parse_args()

    # Get the test features
    test_features = pd.read_pickle(os.path.join(args.out_dir, 'test_dataset.pkl'))

    # Get list of patient IDs
    if args.images_dir.split('/')[-1] == '':
        dataset = args.images_dir.split('/')[-4]
    else:
        dataset = args.images_dir.split('/')[-3]

    pat_dirs = [f.path for f in os.scandir(args.images_dir) if f.is_dir()]

    # Get trained classifier
    if args.cross is False:
        out_dir = args.out_dir
    else:
        # Get dataset
        out_dir_elems = args.out_dir.split('/')
        if out_dir_elems[-1] == '':
            curr_dataset = out_dir_elems[-5]
            unc_type = out_dir_elems[-2]
            est_method = out_dir_elems[-4]
        else:
            curr_dataset = out_dir_elems[-4]
            unc_type = out_dir_elems[-1]
            est_method = out_dir_elems[-3]

        print('Current dataset : {}'.format(curr_dataset))
        if curr_dataset == 'umc':
            new_dataset = 'lits'
        elif curr_dataset == 'lits':
            new_dataset = 'umc'
        else:
            raise RuntimeError('{} is an invalid dataset'.format(curr_dataset))

        # Form the directory to find the output directory
        out_dir = '/data/ishaan/{}/{}/fp_results/{}'.format(new_dataset, est_method, unc_type)
        print('Model directory: {}'.format(out_dir))

    # If 'cross' is False, model dir and res dir point to the same directory
    model_dir = os.path.join(out_dir, 'seed_{}'.format(args.seed))
    res_dir = os.path.join(args.out_dir, 'seed_{}'.format(args.seed))

    if args.reduce_features is True:
        clf = joblib.load(os.path.join(model_dir, 'model_red_feat.pkl'))
        selected_features = joblib.load(os.path.join(out_dir, 'selected_features.pkl'))
    else:
        clf = joblib.load(os.path.join(model_dir, 'model.pkl'))
        selected_features = None

    tp_before = 0
    fp_before = 0
    fn_before = 0

    tp_after = 0
    fp_after = 0
    fn_after = 0

    # Directory to save output graphs
    graph_dir = os.path.join(model_dir, '{}_graphs'.format(dataset))

    if os.path.exists(graph_dir) is True:
        shutil.rmtree(graph_dir)

    os.makedirs(graph_dir)

    for pat_dir in pat_dirs:

        pat_id = pat_dir.split('/')[-1]
        print('Filtering lesions for Patient {}'.format(pat_id))

        # Create directory to store lesion graph(s) for patient
        pat_graph_dir = os.path.join(graph_dir, pat_id)
        os.makedirs(pat_graph_dir)

        pat_df = test_features[test_features['Patient ID'] == pat_id]

        print('Number of detected objects : {}'.format(pat_df.shape[0]))

        # Get the NN prediction and reference segmentation
        seg_path = os.path.join(pat_dir, 'binary_post_proc_pred.nii.gz')
        gt_path = os.path.join(pat_dir, 'true_mask.nii.gz')

        _, gt_np = read_itk_image(gt_path)
        _, seg_np = read_itk_image(seg_path)

        lesion_counts_before = compute_detection_statistics(seg_np, gt_np)

        tp_before += lesion_counts_before['true positives']
        fp_before += lesion_counts_before['false positives']
        fn_before += lesion_counts_before['false negatives']

        graph_before = lesion_counts_before['graph']
        if graph_before is None:
            continue

        # Filter predictions
        filtered_pred, nodes_to_remove = filter_predictions_for_patient(seg=seg_np,
                                                                        pat_df=pat_df,
                                                                        model=clf,
                                                                        pred_lesion_nodes=lesion_counts_before['pred lesion nodes'],
                                                                        selected_features=selected_features)

        if filtered_pred is None:
            continue

        lesion_counts_after = compute_detection_statistics(filtered_pred, gt_np)

        tp_after += lesion_counts_after['true positives']
        fp_after += lesion_counts_after['false positives']
        fn_after += lesion_counts_after['false negatives']

        # This is to check if anything if the prediction partition is empty (will not be drawn then!)
        graph_after = lesion_counts_after['graph']

        # Save both graphs
        if graph_before is not None:
            visualize_lesion_correspondences(graph_before,
                                             fname=os.path.join(pat_graph_dir, 'before_filtering.pdf'))
        if graph_after is not None:
            visualize_lesion_correspondences(graph_before, # Filtered nodes will be removed internally before drawing the graph
                                             fname=os.path.join(pat_graph_dir, 'after_filtering.pdf'),
                                             remove_list=nodes_to_remove)


    print('Before filtering :: TP = {} FP = {} FN = {}'.format(tp_before, fp_before, fn_before))
    print('After filtering ::  TP = {} FP = {} FN = {}'.format(tp_after, fp_after, fn_after))

    # Compute precision and recall
    precision_before = tp_before/(tp_before + fp_before)
    recall_before = tp_before/(tp_before + fn_before)
    f1_before = (2*(precision_before*recall_before))/(precision_before + recall_before)

    precision_after = tp_after/(tp_after + fp_after)
    recall_after = tp_after/(tp_after + fn_after)
    f1_after = (2*(precision_after*recall_after))/(precision_after + recall_after)

    # Compute relative FP and TP reduction
    fp_reduced_relative = (fp_before - fp_after)/fp_before
    tp_reduced_relative = (tp_before - tp_after)/tp_before


    # Save as pandas DF
    det_metrics_dict = {}
    det_metrics_dict['Precition'] = [precision_before, precision_after]
    det_metrics_dict['Recall'] = [recall_before, recall_after]
    det_metrics_dict['F1'] = [f1_before, f1_after]

    relative_reduction_dict = {}
    relative_reduction_dict['FP reduced'] = [fp_reduced_relative]
    relative_reduction_dict['TP reduced'] = [tp_reduced_relative]

    metrics_df = pd.DataFrame.from_dict(det_metrics_dict,
                                        orient='index',
                                        columns=['Before', 'After'])

    relative_reduction_df = pd.DataFrame.from_dict(data=relative_reduction_dict)
    if args.cross is False:
        relative_reduction_df.to_csv(os.path.join(res_dir, 'relative_reduction.csv'.format(dataset)))
    else:
        relative_reduction_df.to_csv(os.path.join(res_dir, 'relative_reduction_cross.csv'.format(dataset)))

    print(metrics_df)

    print(relative_reduction_df)

    if args.cross is False:
        metrics_df.to_csv(os.path.join(res_dir, 'lesion_metrics.csv'))
    else:
        metrics_df.to_csv(os.path.join(res_dir, 'lesion_metrics_cross.csv'))







