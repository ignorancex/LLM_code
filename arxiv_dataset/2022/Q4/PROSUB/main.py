import argparse
import os
import pickle
import timeit

import autokeras as ak
import numpy as np
import tensorflow as tf

import trainer
from utils import data_dict_to_array, set_random_seed_tf

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Code for: Progressive Subsampling for Oversampled Data - Application to Quantitative MRI"
    )
    paradd = parser.add_argument

    # Data loading and processing
    paradd(
        "--data_fil",
        type=str,
        help="Path to pickle, dict each numpy matrices no_samples x no_measurements",
    )
    paradd(
        "--data_train_subjs",
        type=str,
        nargs="*",
        default=["0011", "0012", "0013"],
        help="Data keys used for training",
    )
    paradd(
        "--data_val_subjs",
        type=str,
        nargs="*",
        default=["0014"],
        help="Data keys used for validation",
    )
    paradd(
        "--data_test_subjs",
        type=str,
        nargs="*",
        default=["0015"],
        help="Data keys for testing",
    )
    paradd("--data_normalization", type=str, default="original-measurement")
    paradd(
        "--prctsig",
        type=float,
        default="99",
        help="Percentile for log-signal normalisation",
    )
    paradd(
        "--smallsig", type=float, default="1e-6", help="Minimum signal level allowed"
    )

    # Project name and output
    paradd("--proj_name", type=str, default="tst", help="project name")
    paradd("--project_name", type=str, default="default", help="job name")
    paradd("--out_base", type=str, help="Base dir to save output")
    paradd(
        "--network_name",
        type=str,
        default="prosub",
        help="Approach {prosub|sardunet-nas",
    )

    # Training hyperparameters
    paradd("--noepoch", type=int, default=200, help="Training epochs E")
    paradd(
        "--epochs_decay",
        "-e_d",
        type=int,
        default=20,
        help="Alter mask ascross epochs E_d",
    )
    paradd("--lrate", type=float, default="0.001", help="learning rate")
    paradd("--batch_size", type=int, default=1500, help="Training batch size")
    paradd("--seed", type=int, default=0, help="Random seed")
    paradd("--workers", type=int, default=0, help="No. workers for dataloader")

    # NAS, RFE hyperparameters
    paradd(
        "--T_values",
        nargs="*",
        type=int,
        default=[1, 2, 3],
        help="Steps for different subsampling rates T_1...T",
    )
    paradd(
        "--M_values",
        nargs="*",
        type=int,
        default=[1344, 500, 250],
        help="Subsampling rates for different steps M",
    )
    paradd(
        "--num_units_nas",
        type=int,
        nargs="*",
        default=[64, 128, 256, 512, 1024, 2048],
        help="Number of neurons for NAS layers",
    )
    paradd(
        "--num_units_init",
        type=int,
        nargs=4,
        default=[1063, 781, 781, 1063],
        help="Number initial neurons two for selector, two for predictor",
    )
    paradd(
        "--tuner",
        type=str,
        default="greedy",
        help='Tuner for AutoKERAS: {"greedy","random","bayesian","hyperband"',
    )
    paradd(
        "--num_layers_nas",
        type=int,
        nargs="*",
        default=[1, 2, 3],
        help="NAS no. layers for dense block, first entry is default value.",
    )
    paradd("--overwrite", action="store_true", help="overwrite the searched model")
    paradd(
        "--use_batchnorm",
        "-use_bn",
        type=bool,
        default=False,
        help="Use batch normalisation",
    )
    paradd(
        "--dropout_nas",
        type=float,
        nargs="+",
        default=[0.0],
        help="Dropout for NAS layers, first entry is default value.",
    )

    # Other
    paradd("--notrain", action="store_true", help="Whether to train with NAS.")
    paradd(
        "--nas_fixed",
        action="store_true",
        help="Use NAS w. fixed network hyperparameters",
    )
    paradd(
        "--tensorboard",
        "-tb",
        action="store_true",
        help="Use tensorboard with $ tensorboard --logdir <save_dir>/logs",
    )

    args = parser.parse_args()

    assert args.epochs_decay * 2 <= args.noepoch, "epochs_decay <= noepoch"

    # load data
    with open(args.data_fil, "rb") as f:
        data_dict = pickle.load(f)

    datatrain = data_dict_to_array(data_dict, args.data_train_subjs)
    dataval = data_dict_to_array(data_dict, args.data_val_subjs)
    datatest = data_dict_to_array(data_dict, args.data_test_subjs)
    nmeas_out = datatrain.shape[1]

    out_base_dir = os.path.join(args.out_base, args.proj_name)
    os.makedirs(out_base_dir, exist_ok=True)

    for key, val in args.__dict__.items():
        print(key + ":", val)

    save_model_path = os.path.join(out_base_dir, args.project_name)
    os.makedirs(save_model_path, exist_ok=True)

    set_random_seed_tf(args.seed)

    ### Normalise data
    datatrain[datatrain < args.smallsig] = args.smallsig
    dataval[dataval < args.smallsig] = args.smallsig
    datatest[datatest < args.smallsig] = args.smallsig

    min_val = np.float32(args.smallsig)
    min_val_file = open(os.path.join(save_model_path, "min_val.bin"), "wb")
    pickle.dump(min_val, min_val_file, pickle.HIGHEST_PROTOCOL)
    min_val_file.close()

    if args.data_normalization == "original":
        max_val = np.float32(np.percentile(datatrain, args.prctsig))
    elif args.data_normalization == "original-measurement":
        max_val = np.float32(np.percentile(datatrain, args.prctsig, axis=0))
    else:
        assert False

    max_val_file = open(os.path.join(save_model_path, "max_val.bin"), "wb")
    pickle.dump(max_val, max_val_file, pickle.HIGHEST_PROTOCOL)
    max_val_file.close()

    datatrain = np.float32((datatrain - min_val) / (max_val - min_val))
    dataval = np.float32((dataval - min_val) / (max_val - min_val))
    datatest = np.float32((datatest - min_val) / (max_val - min_val))

    ## Hyperparameters
    options = dict(
        batch_size=args.batch_size,
        loss_affine=[max_val - min_val, min_val],
        tuner=args.tuner,
        seed=args.seed,
        out_base=args.out_base,
        proj_name=args.proj_name,
    )

    ## NAS hyperparameters
    nas_params = dict(
        num_layers_nas=args.num_layers_nas,
        num_units_nas=args.num_units_nas,
        num_units_init=args.num_units_init,
        use_batchnorm=args.use_batchnorm,
        dropout_nas=args.dropout_nas,
        nas_fixed=args.nas_fixed,
        seed=args.seed,
        network_name=args.network_name,
        n_features=nmeas_out,
        n_outputs=nmeas_out,
    )

    # Callback hyperparameters
    update_params = dict(
        epochs=args.noepoch,
        epochs_decay=args.epochs_decay,
        measurements=nmeas_out,
        T_values=args.T_values,
        M_values=args.M_values,
        tensorboard=args.tensorboard,
        save_model_path=save_model_path,
        network_name=args.network_name,
    )

    fit_params = dict(
        batch_size=args.batch_size,
        epochs=args.noepoch,
        workers=args.workers,
        verbose=2,
        shuffle=True,
    )

    auto_model_params = dict(
        project_name=args.project_name,
        max_trials=args.T_values[-1] - 1,  # last iteration on main script
        tuner=args.tuner,
        overwrite=args.overwrite,
        seed=args.seed,
        directory=os.path.dirname(save_model_path),
    )

    nnet = trainer.Trainer(
        save_model_path,
        network_name=args.network_name,
        auto_model_params=auto_model_params,
        update_params=update_params,
        nas_params=nas_params,
        fit_params=fit_params,
        options=options,
    )

    if not args.notrain:
        start_train_timer = timeit.default_timer()
        nnet.train(datatrain, datatrain, dataval, dataval)
        print(
            "Total NAS training time is",
            timeit.default_timer() - start_train_timer,
            "s",
            (timeit.default_timer() - start_train_timer) / 3600,
            "h",
        )

    results = {
        "name": args.proj_name,
        "test_subj": args.data_test_subjs[0],
        "val_scores": {},
        "test_scores": {},
    }  # assume only a single test subj

    # Run the last trial for each M
    for M in args.M_values[1:]:  # Don't need result when M=N
        print("Start final step for M=", M)

        # Can't copy this for some reason and load only once
        save_dir = os.path.join(save_model_path, "m" + str(M))
        best_model = tf.keras.models.load_model(
            save_dir, custom_objects=ak.CUSTOM_OBJECTS, compile=False
        )
        print("loaded model for", str(M), best_model.summary())

        # Load parameters, same as weight order in DownsamplingMultLayer init
        save_vars = np.load(save_dir + "/vars.npy", allow_pickle=True).item()
        vars_load = best_model.get_layer("downsampling_mult_layer").get_weights()
        vars_load[0] = save_vars["m"]  # m
        vars_load[2] = save_vars["sigma_bar"]  # overline{sigma}
        vars_load[3] = np.array(0.0)  # \alpha_t
        vars_load[4] = np.array(args.T_values[-1])  # T
        best_model.get_layer("downsampling_mult_layer").set_weights(vars_load)
        print("Replacing m, sigma_bar in best model")

        best_model.compile(**nnet.compile_params)

        if args.network_name == "prosub":
            callbacks = (
                [
                    tf.keras.callbacks.EarlyStopping(
                        patience=10, restore_best_weights=True
                    )
                ],
            )
        # better w/o early stopping
        elif args.network_name == "sardunet-nas":
            callbacks = []
        else:
            assert False, "network_name in {prosub|sardunet-nas}"

        best_model.fit(
            x=datatrain,
            y=datatrain,
            validation_data=(dataval, dataval),
            callbacks=callbacks,
            **nnet.fit_params,
        )
        val_loss = nnet.evaluate(dataval, dataval, model=best_model)
        test_loss = nnet.evaluate(datatest, datatest, model=best_model)
        print("M=", M, "val_loss and test_loss", val_loss, test_loss)
        results["val_scores"][M] = val_loss
        results["test_scores"][M] = test_loss
    print("test losses", results, "on test subj", args.data_test_subjs[0])
    os.makedirs(os.path.join(options["out_base"], "results"), exist_ok=True)
    all_results_fil = os.path.join(
        options["out_base"], "results", args.proj_name + "_all.npy"
    )
    print("Saving final results in", all_results_fil)
    np.save(all_results_fil, results)

    nnet.clean_up_trials(save_model_path=save_model_path)
