"""Performs the Experiment 1 of paper that confirms that the BNN MCMC captures
the conditional property of the predictor's predictions given the target label.

Notes
-----
Experiment 1 when target is actual predictor's prediction:
    Performing this where the target is the predictor's actual prediciton and
    the preds is the BNN MCMC outputs is performing experiment 1, for
    comparison to iid distributions to show our method captures the conditional
    relationship between (implictly showing this, explicitly showing out method
    closer matches the actual predictor's predictions than the iid methods.

Experiement 2 for residuals when target is given target label to predictor:
    Performing this where the target is the actual target label of the task and
    pred is the estimated predictions of the predictor via the BNN MCMC
    generates the distribution of residuals, which is a distribution of a
    measure and part of experiment 2.
"""
from experiment import io
from experiment.research.bnn.bnn_mcmc_fwd import load_bnn_fwd
from experiment.research.measure import kldiv

if __name__ == "__main__":
    # Create argparser
    args = io.parse_args(
        ['sjd'],
        custom_args=kldiv.add_custom_args,
        description=' '.join([
            'Runs Euclidean Distance on ouputs of euclidean BNN given the',
            'sampled weights. Either expeirment 1 or 2 completed with this',
            'script.',
        ]),
    )

    output_dir = io.create_dirs(args.output_dir)

    # Manage bnn mcmc args from argparse
    bnn_mcmc_args = vars(args.bnn)
    bnn_mcmc_args['sess_config'] = io.get_tf_config(
        args.cpu_cores,
        args.cpu,
        args.gpu,
    )

    givens, pred, weights_sets, bnn = load_bnn_fwd(
        args.data.dataset_filepath,
        args.bnn_weights_file,
        bnn_mcmc_args,
    )

    if not args.target_is_task_target:
        # Exp 1: predictor's prediction is the target: t(y) = y_hat;
        # L2(pred - bnn draw)
        targets = pred
    else:
        # Exp 2: Original label is target where measurement is residuals.
        # L2(given - bnn_draw)
        targets = givens

        # TODO use the chosen distribution of the givens and sample from it
        # (ie. Dirichlet)
    del pred

    # fwd pass of BNN if loaded weights.
    # Perform the measurement of euclidean distance on the BNN MCMC output to
    # the actual prediction
    euclid_dists = kldiv.get_l2dists(
        targets,
        bnn.predict(givens, weights_sets),
        args.normalize,
    )

    kldiv.save_measures(
        output_dir,
        'euclid_dists',
        euclid_dists,
        args.quantiles_frac,
        save_raw=not args.do_not_save_raw,
    )
