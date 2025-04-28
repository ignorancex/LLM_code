# main.py
"""Do a single numerical experiment from start to finish."""

import os
import h5py
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
    ddtdata: bool = False,
):
    r"""Do a single trial from start to finish.

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
    numPODmodes : list(int > 0)
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
    ddtdata : bool
        If ``True``, compute data for visualizing the estimated GP derivative
        distribution compared to the true derivatives.
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
    (
        model,
        time_domain,
        true_states,
        time_domain_sampled,
        snapshots_sampled,
    ) = step1.trajectory(training_span, num_samples, noiselevel)

    # Step 2: Fit Gaussian processes to data ----------------------------------
    # Dimensionality reduction (POD).
    with opinf.utils.TimedBlock(
        f"reducing noisy training states to {numPODmodes} dimensions"
    ):
        basis = config.Basis(num_vectors=numPODmodes)
        basis.fit(snapshots_sampled)
        ax = basis.plot_svdval_decay()
        ax.set_xlim(right=20)
        ax.set_ylim(bottom=1e-4)
        utils.save_figure("svdvals.pdf", andopen=openonsave)
        snapshots_compressed = basis.compress(snapshots_sampled)
        if exportto is not None:
            os.makedirs(os.path.dirname(exportto), exist_ok=True)
            np.save(f"{exportto}-svdvals.npy", basis.svdvals)

    # Fit Gaussian process kernels to the compressed training data.
    time_domain_training = np.linspace(
        training_span[0],
        training_span[-1],
        num_regression_points,
    )
    gps = step2.fit_gaussian_processes(
        time_domain_training=time_domain_training,
        time_domain_sampled=time_domain_sampled,
        snapshots_sampled=snapshots_compressed,
        gp_regularizer=gp_regularizer,
    )
    q0 = snapshots_compressed[:, 0]

    # Step 3: Construct the posterior hyperparameters -------------------------
    input_func = config.ReducedOrderModel.input_func
    inputs = None if input_func is None else input_func(time_domain_training)
    bayesian_model = step3.estimate_posterior(
        time_domain=time_domain,
        gps=gps,
        inputs=inputs,
    )

    # Draw samples from the posterior.
    with opinf.utils.TimedBlock("sampling posterior distribution"):
        draws_compressed = []
        qbar = snapshots_compressed.mean(axis=1).reshape((-1, 1))
        bound = 5 * np.max(np.abs(snapshots_compressed - qbar), axis=1)
        num_unstables = 0
        for _ in range(ndraws):
            draw = bayesian_model.predict(
                initial_conditions=q0,
                timepoints=time_domain,
                input_func=input_func,
            )
            if draw.shape[1] != time_domain.size:
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

    # Step 4: plot results ----------------------------------------------------
    gp_predictions = [gp.predict(time_domain_training) for gp in gps]
    true_states_compressed = basis.compress(true_states)
    true_states_projected = basis.decompress(true_states_compressed)

    romplotter = step4.ReducedPlotter(
        sampling_time_domain=time_domain_sampled,
        training_time_domain=time_domain_training,
        prediction_time_domain=time_domain,
        snapshots_compressed=snapshots_compressed,
        true_states_compressed=true_states_compressed,
        gp_means=np.array([ms[0] for ms in gp_predictions]),
        gp_stds=np.array([ms[1] for ms in gp_predictions]),
        draws_compressed=draws_compressed,
    )

    stateplotter = step4.StatePlotter(
        sampling_time_domain=time_domain_sampled,
        training_time_domain=time_domain_training,
        prediction_time_domain=time_domain,
        spatial_domain=config.spatial_domain,
        num_variables=config.FullOrderModel.num_variables,
        snapshots=snapshots_sampled,
        true_states=true_states,
        true_states_projected=true_states_projected,
        draws=draws,
        numspatialpoints=-1,
    )

    # If desired, export experimental data to HDF5 files for later.
    if exportto is not None:
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
        romplotter.plot_gp_training_fit(width=3)
        utils.save_figure("train.pdf", andopen=openonsave)

        for k, flag in enumerate((True, False)):
            # Bayesian ROM performance in the reduced state space.
            romplotter.plot_posterior(individual=flag)
            utils.save_figure(f"predict{k:d}.pdf", andopen=openonsave)

            # Bayesian ROM performance in the original state space.
            figures = stateplotter.plot_posterior(individual=flag)
            for d, fig in enumerate(figures):
                utils.save_figure(
                    f"predict{k+1:d}-{d+1}.pdf",
                    fig=fig,
                    andopen=openonsave,
                )

    # Save Gaussian process derivative data (for visualization only).
    if ddtdata:
        if exportto is None:
            raise ValueError("argument 'exportto' required when ddtdata given")
        # Get moments of the derivative of each Gaussian process.
        dqdtmeans = np.array([gp.ddt_estimate for gp in gps])
        dqdtstds = np.array(
            [
                np.std(
                    np.random.multivariate_normal(
                        gp.ddt_estimate,
                        gp.ddt_covariance,
                        size=ndraws,
                    ),
                    axis=0,
                )
                for gp in gps
            ]
        )
        # Estimate time derivatives with finite differences of the snapshots.
        dqdtFD = np.gradient(
            snapshots_compressed,
            time_domain_sampled,
            edge_order=2,
            axis=1,
        )

        # Get true derivatives from intrusive model information.
        t_fine = np.linspace(
            time_domain_training[0],
            time_domain_training[-1],
            1000,
        )
        truth_fine = model.solve(config.initial_conditions, t_fine)
        truth_fine_nonlifted = model.unlift(truth_fine)
        dQdt_nonlifted = model.derivative(t_fine, truth_fine_nonlifted)
        dQdt_fine = model.lift_ddts(truth_fine_nonlifted, dQdt_nonlifted)
        dQdt_compressed = basis.entries.T @ basis.nondimensionalize(dQdt_fine)

        # Save the data.
        with h5py.File(f"{exportto}-ddtdata.h5", "w") as hf:
            hf.create_dataset("time_domain_FD", data=time_domain_sampled)
            hf.create_dataset("ddts_finitedifferences", data=dqdtFD)
            hf.create_dataset("time_domain_GP", data=time_domain_training)
            hf.create_dataset("ddts_GPmean", data=dqdtmeans)
            hf.create_dataset("ddts_GPstd", data=dqdtstds)
            hf.create_dataset("time_domain_truth", data=t_fine)
            hf.create_dataset("ddts_truth", data=dQdt_compressed)


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
    parser.add_argument(
        "--ddtdata",
        action="store_true",
        help="save data for visualizing derivative estimates",
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
        ddtdata=args.ddtdata,
    )
