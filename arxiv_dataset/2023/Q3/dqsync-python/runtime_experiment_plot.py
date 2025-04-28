import pandas as pd
import matplotlib.pyplot as plt


if __name__=="__main__":
    data = pd.read_csv("experiment-runtime-2024-03-26-084528.csv")

    # Pivot data
    pdata = data.pivot_table(index=["n"], aggfunc=["min", "max", "mean"]).reset_index()

    # Process data
    n = pdata["n"].to_numpy()

    fun = lambda mtd, lbl, msr: pdata[lbl, "_".join([mtd.lower(), msr])].to_numpy()

    mat_rounding_time_mean = fun("mat", "mean", "rounding_time")
    mat_ev_time_mean = fun("mat", "mean", "ev_time")
    mat_total_time_mean = mat_ev_time_mean + mat_rounding_time_mean
    mat_rounding_time_min = fun("mat", "min", "rounding_time")
    mat_ev_time_min = fun("mat", "min", "ev_time")
    mat_total_time_min = mat_ev_time_min + mat_rounding_time_min
    mat_rounding_time_max = fun("mat", "max", "rounding_time")
    mat_ev_time_max = fun("mat", "max", "ev_time")
    mat_total_time_max = mat_ev_time_max + mat_rounding_time_max

    dq_rounding_time_mean = fun("dq", "mean", "rounding_time")
    dq_ev_time_mean = fun("dq", "mean", "ev_time")
    dq_total_time_mean = dq_rounding_time_mean + dq_ev_time_mean
    dq_rounding_time_min = fun("dq", "min", "rounding_time")
    dq_ev_time_min = fun("dq", "min", "ev_time")
    dq_total_time_min = dq_rounding_time_min + dq_ev_time_min
    dq_rounding_time_max = fun("dq", "max", "rounding_time")
    dq_ev_time_max = fun("dq", "max", "ev_time")
    dq_total_time_max = dq_rounding_time_max + dq_ev_time_max


    # Plot the data
    fig, axs = plt.subplots(1, 3, layout="constrained", sharex=True, figsize=(3.2*3, 3*1))

    def ploterr(x, mean_, min_, max_, prefix):
        plt.loglog(x, mean_, '-o', markersize=3, \
                   label = prefix.upper() + ",mean", \
                   lw = 0.5)
        plt.fill_between(x, \
                         min_, \
                         max_, \
                         alpha = 0.3, \
                         label = prefix.upper() + ",[min,max]")
        plt.grid(which="both", alpha=0.3)

    # EV time
    plt.sca(axs[0])
    ploterr(n, dq_ev_time_mean, dq_ev_time_min, dq_ev_time_max, "dq")
    ploterr(n, mat_ev_time_mean, mat_ev_time_min, mat_ev_time_max, "mat")
    plt.title("Eigenproblem Solver Runtime")
    plt.xlabel("n")
    plt.ylabel("Runtime (sec)")
    plt.legend(fontsize = 7)

    # Rounding time
    plt.sca(axs[1])
    ploterr(n, dq_rounding_time_mean, dq_rounding_time_min, dq_rounding_time_max, "dq")
    ploterr(n, mat_rounding_time_mean, mat_rounding_time_min, mat_rounding_time_max, "mat")
    plt.title("Rounding Runtime")
    plt.xlabel("n")
    plt.ylabel("Runtime (sec)")
    plt.legend(fontsize = 7)

    # Total time
    plt.sca(axs[2])
    ploterr(n, dq_total_time_mean, dq_total_time_min, dq_total_time_max, "dq")
    ploterr(n, mat_total_time_mean, mat_total_time_min, mat_total_time_max, "mat")
    plt.title("Total Runtime")
    plt.xlabel("n")
    plt.ylabel("Runtime (sec)")
    plt.legend(fontsize = 7)

    plt.savefig("runtime_experiment.pdf")

