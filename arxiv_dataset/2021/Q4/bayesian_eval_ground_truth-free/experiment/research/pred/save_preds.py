"""Obtain predictions from a FacialBeauty ResNeXt50 model."""
import logging
import os

from experiment import io
from experiment.research import predictors
from experiment.research.sjd import sjd_log_prob_exp


def load_fold_save_pred(
    dir_path,
    weights_file,
    label_src,
    summary_name='summary.json',
    pred_name='pred.json',
    data=None,
    save_labels=False,
):
    # load the data and model
    model, features, labels = sjd_log_prob_exp.load_eval_fold(
        dir_path,
        weights_file,
        label_src,
        summary_name=summary_name,
        data=data,
        load_model=True,
        pred_name=pred_name,
    )

    # Get predictions and save to file
    file_to_save = {
        'train': model.predict(features[0]),
        'test': model.predict(features[1]),
    }

    if save_labels:
        file_to_save['labels'] = {
            'train': labels[0],
            'test': labels[1],
        }

    io.save_json(os.path.join(dir_path, pred_name), file_to_save)


def load_kfolds_save_preds():
    # TODO given dir containing the different folds' results directories, save
    return


def add_custom_args(parser):
    # TODO add the str id for the target to compare to: 'frequency', 'ground_truth'
    # Can add other annotator aggregation methods as necessary, ie. D&S.
    sjd_log_prob_exp.add_human_sjd_args(parser)

    parser.add_argument(
        '--whole_kfold_exp',
        action='store_true',
        help=' '.join([
            'If given, then expects the directory to contain all the kfold',
            'results directories to each have their predictions saved.',
        ]),
    )

    parser.add_argument(
        '--save_labels',
        action='store_true',
        help='Save the labels in the JSON of the predictions.'
    )


if __name__ == '__main__':
    args = io.parse_args(['mle', 'sjd'], add_custom_args)

    # NOTE be aware that the defaults of SJD args will overwrite src candidates
    del args.sjd.target_distrib
    del args.sjd.transform_distrib
    del args.sjd.independent
    del args.sjd.mle_args

    logging.info('Loading data for efficiency')
    # Load data once: features, labels, label_bin
    data = predictors.load_prep_data(
        args.dataset_id,
        vars(args.data),
        args.label_src,
        args.model.parts,
    )

    if args.whole_kfold_exp:
        load_kfolds_save_preds()
    else:
        load_fold_save_pred(
            dir_path=args.human_sjd.dir_path,
            weights_file=args.human_sjd.model_weights_file,
            label_src=args.label_src,
            summary_name=args.human_sjd.summary_name,
            data=data,
            save_labels=args.save_labels,
            pred_name=args.human_sjd.pred_name,
        )
