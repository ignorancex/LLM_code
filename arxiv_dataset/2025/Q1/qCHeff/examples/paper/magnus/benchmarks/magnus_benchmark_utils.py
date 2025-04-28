import cupy as cp
import marimo as mo  # type: ignore
import more_itertools
import numpy as np
import polars as pl  # type: ignore
import qutip as qt
import seaborn as sns  # type: ignore
from cupyx.profiler import benchmark

from qcheff.models.spin_chain.utils import (
    setup_magnus_chain_example,
)


def create_bench_df(bench_results, method_name: str, **kwargs):
    incl_params_dict = kwargs.get("incl_params_dict", {})

    return pl.DataFrame(
        more_itertools.zip_broadcast(
            *list(incl_params_dict.values()),
            method_name,
            bench_results.cpu_times,
            bench_results.gpu_times.squeeze(),
        ),
        schema=[*list(incl_params_dict.keys()), "Method", "cpu_time", "gpu_time"],
    )


def bench_magnus(**kwargs):
    """
    Main benchmark function for Magnus. This function will run benchmarks for the given
     configuration.

    """

    chain_size_list = kwargs.get("chain_size_list", [3])
    n_repeat = kwargs.get("n_repeat", 10)
    n_warmup = kwargs.get("n_warmup", 5)
    # Simulation parameters to include in benchmark dataframe
    incl_params = ["chain_size", "num_magnus_intervals", "num_tlist"]

    incl_params_dict = {key: kwargs.get(key) for key in incl_params}

    # Benchmark both techniques for a given number of points
    qutip_tpts = kwargs.get("qutip_tpts", kwargs.get("num_tlist", 10**3))
    magnus_tpts = kwargs.get("num_magnus_intervals", qutip_tpts)

    def qutip_bench_func(test_system, test_magnus, init_state):
        return qt.sesolve(
            H=test_system.get_qutip_tdham(test_magnus.tlist),
            psi0=init_state,
            tlist=test_magnus.tlist,
            options={"store_states": True},
        ).states

    def magnus_bench_func(test_magnus, init_state):
        return list(
            test_magnus.evolve(
                init_state=init_state,
                num_intervals=kwargs["num_magnus_intervals"],
            )
        )

    bench_list = []

    for chain_size in mo.status.progress_bar(chain_size_list):
        allzero_state = qt.basis(dimensions=[2] * chain_size, n=[0] * chain_size)
        if kwargs.get("qutip_bench", False):
            test_system, test_magnus = setup_magnus_chain_example(
                **(
                    kwargs
                    | {
                        "device": "cpu",
                        "num_tlist": qutip_tpts,
                        "chain_size": chain_size,
                    }
                )
            )
            # First, benchmark QuTiP

            qutip_bench = benchmark(
                qutip_bench_func,
                args=(test_system, test_magnus, allzero_state),
                n_repeat=n_repeat,
                n_warmup=n_warmup,
            )

            bench_list.append(
                create_bench_df(
                    qutip_bench,
                    "QuTiP",
                    incl_params_dict=incl_params_dict | {"chain_size": chain_size},
                )
            )

        test_psi0 = np.asarray((allzero_state).unit()[:])

        if kwargs.get("scipy_bench", False):
            # Then benchmark Magnus on CPU
            test_system, test_magnus = setup_magnus_chain_example(
                **(kwargs | {"device": "cpu", "chain_size": chain_size})
            )

            cpu_bench = benchmark(
                magnus_bench_func,
                args=(test_magnus, test_psi0),
                n_repeat=n_repeat,
                n_warmup=n_warmup,
            )

            bench_list.append(
                create_bench_df(
                    cpu_bench,
                    "Magnus (CPU)",
                    incl_params_dict=incl_params_dict | {"chain_size": chain_size},
                )
            )

        if kwargs.get("cupy_bench", False):
            # Then benchmark Magnus on GPU

            # Use specified GPU device
            gpu_device_id = kwargs.get("gpu_id", 0)

            with cp.cuda.Device(gpu_device_id):
                test_system, test_magnus = setup_magnus_chain_example(
                    **(kwargs | {"device": "gpu", "chain_size": chain_size})
                )

                gpu_bench = benchmark(
                    magnus_bench_func,
                    args=(test_magnus, test_psi0),
                    n_repeat=n_repeat,
                    n_warmup=n_warmup,
                )

                bench_list.append(
                    create_bench_df(
                        gpu_bench,
                        "Magnus (GPU)",
                        incl_params_dict=incl_params_dict | {"chain_size": chain_size},
                    )
                )

    bench_df = pl.concat(bench_list).drop("cpu_time").rename({"gpu_time": "time"})

    return bench_df


def magnus_bench_report(data, /, **kwargs):
    with (
        sns.plotting_context(kwargs.get("plotting_context", "paper")),
        sns.axes_style(kwargs.get("plot_style", "ticks")),
    ):
        magnus_bench_plot = sns.catplot(
            data,
            kind="bar",
            hue_order=["QuTiP", "Magnus (CPU)", "Magnus (GPU)"],
            palette={
                "QuTiP": "slategray",
                "Magnus (CPU)": (0 / 255, 104 / 255, 181 / 255),
                "Magnus (GPU)": (118 / 255, 185 / 255, 0 / 255),
            },
            x="chain_size",
            y="time",
            hue="Method",
            err_kws={"linewidth": 1},
            capsize=0.5,
            aspect=0.8,
        )
        magnus_bench_plot.ax.set_yscale("log")
        magnus_bench_plot.set_axis_labels("Subroutine", "Time (s)")

    return magnus_bench_plot


def measure_accuracy(pulse_coeffs, sample_num_list: tuple[float, ...], **kwargs):
    final_states = []
    chain_size = kwargs.get("chain_size", 3)
    method_name = kwargs.get("method_name", "qutip")

    allzero_state = qt.basis(dimensions=[2] * chain_size, n=[0] * chain_size)

    if method_name == "qutip":
        for sample_num in sample_num_list:
            test_system, test_magnus = setup_magnus_chain_example(
                pulse_coeffs=pulse_coeffs, num_tlist=sample_num, **kwargs
            )
            final_states.append(
                qt.Qobj(
                    qt.sesolve(
                        H=test_system.get_qutip_tdham(test_magnus.tlist),
                        psi0=allzero_state,
                        tlist=test_magnus.tlist,
                        options={"store_final_state": True},
                    ).final_state[:]
                )
            )

    if method_name == "magnus":
        test_psi0 = np.asarray((allzero_state).unit()[:])

        test_system, test_magnus = setup_magnus_chain_example(
            pulse_coeffs=pulse_coeffs, num_tlist=max(sample_num_list), **kwargs
        )
        final_states = [
            qt.Qobj(
                cp.asnumpy(
                    more_itertools.last(
                        list(
                            test_magnus.evolve(
                                init_state=test_psi0,
                                num_intervals=mag_intervals,
                            )
                        )
                    )
                )
            )
            for mag_intervals in sample_num_list
        ]

    baseline_state = final_states[-1]

    err_list = [
        (qt.hilbert_dist(final_state, baseline_state)) for final_state in final_states
    ]

    return err_list, baseline_state
