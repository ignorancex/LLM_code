# main.py
"""Do a single numerical experiment from start to finish."""

import os
import numpy as np

import opinf

import config
import utils
import step1_generate_data as step1
import step2_fitgps as step2
import step3_estimate as step3
import step4_plot as step4


def main(
    training_span: tuple[float, float],
    num_samples: int,
    noiselevel: float,
    num_regression_points: int,
    numPODmodes: list[int],
    gp_regularizer: float = 1e-8,
    ndraws: int = 100,
    exportto: str = None,
    openonsave: bool = True,
):
    r"""Do a single trial from start to finish (do not save intermediate data).

    Parameters
    ----------
    training_span : (float, float)
        Time domain over which to sample solution data.
    num_samples : int > 0
        Number of snapshots to sample.
    noiselevel : float >= 0
        Percentage of noise applied to the training snapshots.
    num_regression_points : int > 0
        Number of points at which to evaluate the GP state and derivative
        estimates.
    numPODmodes : int or (int, int, int)
        Number of POD modes (left singular vectors) to use in the
        dimensionality reduction of the training data; this is the dimension
        of the reduced-order model.
    gp_regularizer : float >= 0
        Regularization hyperparameter for the GP inference in inverting for
        the least-squares weight matrix.
    ndraws : int
        Number of draws from the posterior distribution.
    exportto : str
        If given, write experiment data to an HDF5 files with ``exportto`` as
        the prefix.
    openonsave : bool
        If ``True`` (default), open figures as they are created.
    """
    if isinstance(numPODmodes, (list, tuple)) and len(numPODmodes) == 1:
        numPODmodes = numPODmodes[0]

    # Report on experimental scenario.
    utils.summarize_experiment(
        training_span=training_span,
        num_samples=num_samples,
        noiselevel=noiselevel,
        num_regression_points=num_regression_points,
        numPODmodes=numPODmodes,
        gp_regularizer=gp_regularizer,
        ndraws=ndraws,
    )

    # Step 1: Generate data ---------------------------------------------------
    sampler = step1.TrajectorySampler(
        training_span,
        num_samples,
        noiselevel,
        num_regression_points,
        synced=False,
    )
    (
        true_states,
        time_domain_sampled,
        snapshots_sampled,
        training_inputs,
    ) = sampler.multisample(config.input_parameters, plot=openonsave)

    # Step 2: Fit Gaussian processes to data ----------------------------------
    # Dimensionality reduction with POD.
    with opinf.utils.TimedBlock(
        f"reducing noisy training states to {numPODmodes} dimensions"
    ):
        basis = config.Basis(num_vectors=numPODmodes)
        basis.fit(np.hstack(snapshots_sampled))
        ax = basis.plot_svdval_decay()
        ax.set_xlim(right=20)
        ax.set_ylim(bottom=1e-4)
        utils.save_figure("svdvals.pdf", andopen=openonsave)
        snapshots_compressed = [basis.compress(Q) for Q in snapshots_sampled]

    # Fit Gaussian process kernels to the compressed training data.
    gps = []  # indexed by trajectory, then reduced mode.
    for i, (t, Q) in enumerate(zip(time_domain_sampled, snapshots_compressed)):
        print(f"\n*** Trajectory {i+1} ***")
        gps.append(
            step2.fit_gaussian_processes(
                time_domain_training=sampler.training_time_domain,
                time_domain_sampled=t,
                snapshots_sampled=Q,
                gp_regularizer=gp_regularizer,
            )
        )
    q0s = [
        np.array([gp.state_estimate[0] for gp in gps[ell]])
        for ell in range(len(config.input_parameters))
    ]

    # Step 3: Construct the posterior hyperparameters -------------------------
    bayesian_model = step3.estimate_posterior(
        prediction_time_domain=sampler.prediction_time_domain,
        gps=gps,
        training_inputs=training_inputs,
        initial_conditions=q0s,
    )

    def draws_for_single_trajectory(params, q0, snaps_compressed):
        input_func = config.input_func_factory(params)
        qbar = snaps_compressed.mean(axis=1).reshape((-1, 1))
        bound = 5 * np.max(np.abs(snaps_compressed - qbar), axis=1)

        num_unstables = 0
        draws_compressed = []
        for _ in range(ndraws):
            draw = bayesian_model.predict(
                q0,
                sampler.prediction_time_domain,
                input_func=input_func,
            )
            if draw.shape[1] != sampler.prediction_time_domain.size:
                num_unstables += 1
                continue
            if np.any(np.abs(draw - qbar).max(axis=1) > bound):
                num_unstables += 1
                continue
            draws_compressed.append(draw)
        if num_unstables:
            print(f"\n{num_unstables}/{ndraws} draws unstable")

        # Translate results back to original state space.
        draws = [basis.decompress(draw) for draw in draws_compressed]
        return draws_compressed, draws

    # Draw samples from the posterior (for now at the same model parameters).
    with opinf.utils.TimedBlock("sampling posterior distribution"):
        all_draws_compressed, all_draws_decompressed = [], []
        for ell, params in enumerate(config.input_parameters):
            draws_compressed, draws_decompressed = draws_for_single_trajectory(
                params,
                q0s[ell],
                snapshots_compressed[ell],
            )
            all_draws_compressed.append(draws_compressed)
            all_draws_decompressed.append(draws_decompressed)

    # Step 4: plot results ----------------------------------------------------
    # Initialize reduced plotter.
    gp_predictions = [
        [gp.predict(sampler.training_time_domain) for gp in gps[ell]]
        for ell in range(len(config.input_parameters))
    ]
    true_states_compressed = [basis.compress(Q) for Q in true_states]
    true_states_projected = [
        basis.decompress(Q_) for Q_ in true_states_compressed
    ]

    romplotter = step4.ReducedPlotter(
        trajectory_parameters=config.input_parameters,
        sampling_time_domain=time_domain_sampled,
        training_time_domain=sampler.training_time_domain,
        prediction_time_domain=sampler.prediction_time_domain,
        snapshots_compressed=snapshots_compressed,
        true_states_compressed=true_states_compressed,
        gp_means=[np.array([ms[0] for ms in pred]) for pred in gp_predictions],
        gp_stds=[np.array([ms[1] for ms in pred]) for pred in gp_predictions],
        draws_compressed=all_draws_compressed,
    )

    stateplotter = step4.StatePlotter(
        trajectory_parameters=config.input_parameters,
        sampling_time_domain=time_domain_sampled,
        training_time_domain=sampler.training_time_domain,
        prediction_time_domain=sampler.prediction_time_domain,
        spatial_domain=config.spatial_domain,
        num_variables=config.FullOrderModel.num_variables,
        snapshots=snapshots_sampled,
        true_states=true_states,
        true_states_projected=true_states_projected,
        draws=all_draws_decompressed,
        numspatialpoints=-1,
    )

    # If desired, export experimental data to HDF5 files for later.
    if exportto is not None:
        os.makedirs(os.path.dirname(exportto), exist_ok=True)
        np.save("onesnap_noisy.npy", snapshots_sampled[1][:, 10])
        reduced_dfile = f"{exportto}_data-reduced.h5"
        with opinf.utils.TimedBlock(
            f"exporting reduced data to {reduced_dfile}"
        ):
            romplotter.save(reduced_dfile, overwrite=True)
        full_dfile = f"{exportto}_data-full.h5"
        with opinf.utils.TimedBlock(f"exporting full data to {full_dfile}"):
            stateplotter.save(full_dfile, overwrite=True)

    # Plot and save results.
    with opinf.utils.TimedBlock("plotting model performance\n"):
        # Gaussian process fit in the reduced state space.
        figures = romplotter.plot_gp_training_fit(width=3)
        for i, fig in enumerate(figures):
            utils.save_figure(
                f"train_{config.DIMFMT(i)}.pdf",
                andopen=openonsave,
                fig=fig,
            )

        for k, flag in enumerate((True, False)):
            # Performance in the reduced state space.
            figures = romplotter.plot_posterior(individual=flag)
            for i, fig in enumerate(figures):
                utils.save_figure(
                    f"predict{k:d}_{config.DIMFMT(i)}.pdf",
                    andopen=openonsave,
                    fig=fig,
                )

            # Performance in original state space.
            all_figures = stateplotter.plot_posterior(
                all_draws_decompressed,
                projected=True,
                individual=flag,
            )
            for d, figures in enumerate(all_figures):
                for ell, fig in enumerate(figures):
                    utils.save_figure(
                        f"predict{k+2}-{d+1}_{config.TRJFMT(ell)}.pdf",
                        andopen=openonsave,
                        fig=fig,
                    )

    # Prediction at a new parameter value.
    if config.test_parameters is None:
        return
    test_trajectory = sampler.sample(config.test_parameters)[0]
    test_trajectory_compressed = basis.compress(test_trajectory)
    with opinf.utils.TimedBlock("sampling posterior distribution"):
        draws_compressed, draws_decompressed = draws_for_single_trajectory(
            config.test_parameters,
            test_trajectory_compressed[:, 0],
            test_trajectory_compressed,
        )

    if exportto is not None:
        dfile = f"{exportto}_newtrajectory.h5"
        with opinf.utils.TimedBlock(
            f"exporting new trajectory data to {dfile}"
        ):
            with opinf.utils.hdf5_savehandle(dfile, overwrite=True) as hf:
                hf.create_dataset(
                    "truth_reduced",
                    data=test_trajectory_compressed,
                )
                hf.create_dataset(
                    "truth_full",
                    data=test_trajectory,
                )
                hf.create_dataset(
                    "draws_reduced",
                    data=draws_compressed,
                )
                hf.create_dataset(
                    "draws_full",
                    data=draws_decompressed,
                )

    fig = romplotter.plot_posterior_newparams(
        draws_compressed,
        truth=test_trajectory_compressed,
    )
    utils.save_figure(
        "newtrajectory_reduced.pdf",
        andopen=openonsave,
        fig=fig,
    )

    for d, fig in enumerate(
        stateplotter.plot_posterior_newparams(
            draws=draws_decompressed,
            truth=test_trajectory,
            spatial_domain=config.spatial_domain,
        )
    ):
        utils.save_figure(
            f"newtrajectory_full-{d}.pdf",
            andopen=openonsave,
            fig=fig,
        )


if __name__ == "__main__":
    # Set up command line argument parsing.
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    fname = os.path.basename(__file__)
    spc = " " * len(fname)
    parser.usage = f""" python3 {fname} --help
        python3 {fname} T_MAX NUMSAMPLES NOISELEVEL NUMPTS NUMPODMODES
                {spc} [--gpreg ETA] [--opinfreg RHO] [--ndraws NDRAWS]
                {spc} [--exportto PREFIX] [--noopen]
    """

    parser.add_argument(
        "t_max",
        type=float,
        help="upper bound on the training time domain",
    )
    parser.add_argument(
        "num_samples",
        type=int,
        help="number of training snapshots to sample",
    )
    parser.add_argument(
        "noiselevel",
        type=float,
        help="percentage of noise added to training snapshots",
    )
    parser.add_argument(
        "num_regression_points",
        type=int,
        help="number of points to use in the OpInf regression0",
    )
    parser.add_argument(
        "numPODmodes",
        nargs="+",
        type=int,
        help="number of POD modes, e.g., ROM size",
    )
    parser.add_argument(
        "--gpreg",
        type=float,
        default=1e-8,
        help="regularization for GP matrices (eta)",
    )
    parser.add_argument(
        "--ndraws",
        type=int,
        default=100,
        help="number of posterior model draws",
    )
    parser.add_argument(
        "--exportto",
        help="prefix for HDF5 files to save plot data to",
    )
    parser.add_argument(
        "--noopen",
        action="store_true",
        help="do not open figures automatically",
    )

    args = parser.parse_args()
    main(
        training_span=[0, args.t_max],
        num_samples=args.num_samples,
        noiselevel=args.noiselevel,
        num_regression_points=args.num_regression_points,
        numPODmodes=args.numPODmodes,
        gp_regularizer=args.gpreg,
        ndraws=args.ndraws,
        exportto=args.exportto,
        openonsave=not args.noopen,
    )
