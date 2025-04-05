import os
import json
import itertools
import numpy as np
import pandas as pd
from scipy.linalg import lstsq
from phasefinder.utils import build_path

import matplotlib
#matplotlib.use("agg")
import matplotlib.pyplot as plt
matplotlib.rc("xtick", labelsize=8)
matplotlib.rc("ytick", labelsize=8)

plt.style.reload_library()
plt.style.use('./include/sf.mplstyle')

figsize = plt.rcParams['figure.figsize']

### plot magnetization, order parameter curves, and U_4 Binder cumulant curves

def subplot_stat(results_dir, J, observable_name, L, N=None, fold=None, seed=None, colors=None, what="distribution", xlabel=True, ylabel=True, title=None):
    dir = build_path(results_dir, J, observable_name, L, N=N, fold=fold, seed=seed, subdir="processed")
    with np.load(os.path.join(dir, "stats.npz")) as fp:
        stats = dict(fp)
    with np.load(os.path.join(dir, "tc.npz")) as fp:
        tc_estimate = fp["mean"]
    tc_exact = 2/np.log(1+np.sqrt(2))
    if what == "distribution":
        stats["distribution_range"] = stats["distribution_range"].tolist() if "distribution_range" in stats else [-1, 1]
        plt.imshow(np.flip(stats["distributions"].T, 0), cmap="gray_r", vmin=0, vmax=1, extent=(stats["temperatures"].min(), stats["temperatures"].max(), *stats["distribution_range"]), aspect="auto")
        plt.axvline(x=tc_exact, linestyle="dashed", color=colors["tc"])
        if xlabel:
            plt.xlabel(r"Temperature ($T$)", fontsize=8)
        else:
            ax = plt.gca()
            ax.set_xticklabels([])
        if ylabel:
            plt.ylabel(J.capitalize()+"\n"+r"Observable ($\mathcal{O}$)", fontsize=8)
        if title is not None:
            plt.title(title, fontsize=8)
    if what == "order":
        onsager = np.where(stats["temperatures"]<2/np.log(1+np.sqrt(2)), np.clip(1-1/np.sinh(2/stats["temperatures"])**4, 0, None)**(1/8), np.zeros_like(stats["temperatures"]))
        stats["order_means"] /= stats["order_means"][0]
        #onsager = stats["order_means"][0]/onsager[0]*onsager
        plt.plot(stats["temperatures"], stats["order_means"], color="black")
        plt.plot(stats["temperatures"], stats["order_means"]-stats["order_stds"], color="black", linestyle="dashed")
        plt.plot(stats["temperatures"], stats["order_means"]+stats["order_stds"], color="black", linestyle="dashed")
        plt.plot(stats["temperatures"], onsager, linestyle="dashed", color=blue)
        plt.axvline(x=tc_exact, linestyle="dashed", color=colors["tc"])
        plt.ylim(-0.05,1.05)
        if xlabel:
            plt.xlabel(r"Temperature ($T$)", fontsize=8)
        else:
            ax = plt.gca()
            ax.set_xticklabels([])
        if ylabel:
            plt.ylabel(J.capitalize()+"\n"+r"Mean Abs Obs ($\langle|\mathcal{O}|\rangle$)", fontsize=8)
        if title is not None:
            plt.title(title, fontsize=8)
    if what == "binder":
        mask = stats["temperatures"] < tc_estimate
        not_mask = np.logical_not(mask)
        step_fit = np.array([stats["u4_means"][mask].mean()]*mask.sum() + [stats["u4_means"][not_mask].mean()]*not_mask.sum())
        plt.plot(stats["temperatures"], stats["u4_means"], color="black")
        plt.plot(stats["temperatures"], stats["u4_means"]-stats["u4_stds"], color="black", linestyle="dashed")
        plt.plot(stats["temperatures"], stats["u4_means"]+stats["u4_stds"], color="black", linestyle="dashed")
        plt.plot(stats["temperatures"], step_fit, linestyle="dashed", color=colors["fit"])
        plt.axvline(x=tc_exact, linestyle="dashed", color=colors["tc"])
        plt.ylim(-0.09,0.7)
        if xlabel:
            plt.xlabel(r"Temperature ($T$)", fontsize=8)
        else:
            ax = plt.gca()
            ax.set_xticklabels([])
        if ylabel:
            plt.ylabel(J.capitalize()+"\n"+r"Binder ($U_4$)", fontsize=8)
        if title is not None:
            plt.title(title, fontsize=8)


def plot_stat(results_dir, Js, observable_names, L, N=None, fold=None, seed=None, colors=None, titles=None, what="distribution"): 
    plt.figure(figsize=(2*figsize[0],2*figsize[1])) 
    nrows,ncols = len(Js), len(observable_names)
    for (index, (J, name)) in enumerate(itertools.product(Js, observable_names)):
        plt.subplot(nrows, ncols, index+1)
        xlabel = index//ncols == nrows-1
        ylabel = index%ncols == 0
        title = titles[name] if index//ncols==0 else None
        subplot_stat(results_dir, J, name, L, N=N, fold=fold, seed=seed, colors=colors, what=what, xlabel=xlabel, ylabel=ylabel, title=title)

    plt.tight_layout()
    output_dir = os.path.join(results_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "{}.pdf".format(what)))
    plt.close()


### plot critical temperature estimates

def y_minmax(means, stds, padding=0.05):
    y_min = (means-stds*np.less(means, 0).astype(np.int)).min()
    y_max = (means+stds*np.greater(means, 0).astype(np.int)).max()
    y_range = y_max-y_min
    y_min = y_min - padding*y_range
    y_max = y_max + padding*y_range
    return y_min, y_max


def bar_width_shifts(n_bars):
    total_width = 0.7
    width = total_width/n_bars
    shifts = np.array([-total_width/2 + total_width/(2*n_bars)*(2*m+1) for m in range(n_bars)])
    return width, shifts


def bar_yerrs(ys, errs):
    yerrs = []
    for (y, err) in zip(ys, errs):
        if y >= 0:
            yerrs.append([0, err])
        else:
            yerrs.append([err, 0])
    yerrs = np.array(yerrs).T
    return yerrs


def get_unique_legend_handles_labels(fig):
    tuples = [(h, l) for ax in fig.get_axes() for (h, l) in zip(*ax.get_legend_handles_labels())]
    handles, labels = zip(*tuples)
    unique = [(h, l) for (i, (h, l)) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    handles, labels = zip(*unique)
    return list(handles), list(labels)


def subplot_tc(results_dir, J, L, Ns, encoder_names, labels, colors, xlabel=True, ylabel=True, title=None):
    data = pd.read_csv(os.path.join(results_dir, "processed", "gathered.csv"))
    data = data[data.J.eq(J) & data.L.eq(str(L))]
    y_min, y_max = y_minmax(data.tc_mean.values, data.tc_std.values)
    x = np.arange(len(Ns))
    width, shifts = bar_width_shifts(len(encoder_names))
    for (name, shift) in zip(encoder_names, shifts):
        plt.bar(x+shift, data[data.observable.eq(name)].tc_mean.values, width, yerr=bar_yerrs(data[data.observable.eq(name)].tc_mean.values, data[data.observable.eq(name)].tc_std.values), capsize=2, ecolor=colors[name], color=colors[name], label=labels[name])
    plt.axhline(y=data[data.observable.eq("magnetization")].tc_mean.values[0], linestyle="dashed", color=colors["magnetization"], label=labels["magnetization"])
    plt.xticks(x, Ns)
    if L == None:
        plt.ylim(-4.2, 2.3)
    else:
        plt.ylim(y_min, y_max)
    if xlabel:
        plt.xlabel(r"Samples per temp. $(N)$", fontsize=8)
    else:
        ax = plt.gca()
        ax.set_xticklabels([])
    if ylabel:
        plt.ylabel(r"Error $(\%)$", fontsize=8)
    else:
        if L == None:
            ax = plt.gca()
            ax.set_yticklabels([])
    if title is not None:
        plt.title(title, fontsize=8)


def plot_tc(results_dir, J, Ls, Ns, encoder_names, labels, colors, grid_dims=None):
    plt.figure(figsize=(figsize[0],figsize[0]))
    nrows, ncols = grid_dims
    for (index, L) in enumerate(Ls):
        plt.subplot(nrows, ncols, index+1)
        xlabel = index//ncols == nrows-1
        ylabel = index%ncols == 0
        title = r"$L = {:d}$".format(L)
        subplot_tc(results_dir, J, L, Ns, encoder_names, labels, colors, xlabel=xlabel, ylabel=ylabel, title=title)
    handles, labels = get_unique_legend_handles_labels(plt.gcf())
    plt.figlegend(handles, labels, ncol=2, loc=(0.1,0.9), fancybox=True, fontsize=8)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85,wspace=0.18)
    output_dir = os.path.join(results_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "tc_{}.pdf".format(J)))
    plt.close()


def plot_tc_extrapolate(results_dir, Js, Ns, encoder_names, labels, colors):
    plt.figure(figsize=(figsize[0],figsize[0]))
    nrows, ncols = 1, len(Js)
    for (index, J) in enumerate(Js):
        plt.subplot(nrows, ncols, index+1)
        xlabel = index//ncols == nrows-1
        ylabel = index%ncols == 0
        title = J.capitalize()
        subplot_tc(results_dir, J, None, Ns, encoder_names, labels, colors, xlabel=xlabel, ylabel=ylabel, title=title)
    handles, labels = get_unique_legend_handles_labels(plt.gcf())
    plt.figlegend(handles, labels, ncol=2, loc=(0.1,0.9), fancybox=True, fontsize=8)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85,wspace=0.075)
    output_dir = os.path.join(results_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "tc_extrapolate.pdf"))
    plt.close()


### plot anomaly detection for ferromagnetic

def subplot_ad(results_dir,T,encoder_names, labels, colors, xlabel=True, ylabel=True, title=None):
    data = pd.read_csv(os.path.join(results_dir, "processed", "anomaly_detection_FM.csv"))
    
    d = data[data.temperature == T]

    x = [1,2,3]
    width, shifts = bar_width_shifts(len(encoder_names))

    for (name, shift) in zip(encoder_names, shifts):
        y = d[f'{name}_mean'].values[1:]
        Δy = d[f'{name}_std'].values[1:]
        plt.bar(x+shift, y, width, yerr=bar_yerrs(y,Δy), capsize=2,ecolor=colors[name], color=colors[name], label=labels[name])

    ax = plt.gca()
    ax.set_xticks(x)
    ax.set_xticklabels([r'$10^{-3}$',r'$10^{-2}$',r'$10^{-1}$'])
    ax.set_ylim(0,13.5)
    plt.xlabel(r'Sym. breaking field $(h)$')
    if ylabel:
        plt.ylabel(r'Confidence score  $(\xi)$')
    else:
        ax.set_yticklabels([])

    if title is not None:
        plt.title(title, fontsize=8)


def plot_ad(results_dir, Ts, encoder_names, labels, colors):
    plt.figure(figsize=(figsize[0],figsize[0]))
    nrows, ncols = 1, len(Ts)
    Tc = 2.0/np.log(1.0+np.sqrt(2))
   
    for (index, T) in enumerate(Ts):
         
        plt.subplot(nrows, ncols, index+1)
        xlabel = index//ncols == nrows-1
        ylabel = index%ncols == 0
        if T < Tc:
            title = f'$T = {T:.1f} < T_c$'
        else:
            title = f'$T = {T:.1f} > T_c$'
        
        subplot_ad(results_dir, T, encoder_names, labels, colors, xlabel=xlabel, ylabel=ylabel, title=title)
        
    handles, labels = get_unique_legend_handles_labels(plt.gcf())
    plt.figlegend(handles, labels, ncol=2, loc=(0.22,0.93),fancybox=True, fontsize=8)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85,wspace=0.1)
    
    output_dir = os.path.join(results_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "anomaly_detection.pdf"))
    plt.close()


### plot error vs lattice size data

def subplot_tc_vs_lattice(results_dir, J, Ls, observable_names, labels, colors, N=None, xlabel=True, ylabel=True, title=None):
    x = 1/np.array(Ls)
    x_pnts = np.linspace(0, x.max(), 100, endpoint=True)
    data = pd.read_csv(os.path.join(results_dir, "processed", "gathered.csv"))
    data = data[(data.J==J) & (data.N==N)]
    for name in observable_names:
        y = data[(data.observable==name) & (data.L!="None")].tc_mean.values
        slope = float( data[(data.observable==name) & (data.L=="None")].tc_slope.values )
        intercept = float( data[(data.observable==name) & (data.L=="None")].tc_yintercept.values )
        yhat = slope*x_pnts + intercept
        plt.scatter(x, y, alpha=0.7, s=12, lw=0.25,color=colors[name], label=labels[name])
        plt.plot(x_pnts, yhat, alpha=0.7, color=colors[name])
        plt.xlim(0,0.069)
        plt.ylim(-2.6,17.4)
    if xlabel:
        plt.xlabel(r"Inverse lattice size ($L^{-1}$)", fontsize=8)
    if ylabel:
        plt.ylabel(r"Error $(\%)$", fontsize=8)
    else:
        ax = plt.gca()
        ax.set_yticklabels([])
    if title is not None:
        plt.title(title, fontsize=8)


def plot_tc_vs_lattice(results_dir, Js, Ls, observable_names, labels, colors, N=None):
    plt.figure(figsize=(figsize[0],figsize[0]))
    nrows, ncols = 1, len(Js)
    for (index, J) in enumerate(Js):
        plt.subplot(nrows, ncols, index+1)
        xlabel = index//ncols == nrows-1
        ylabel = index%ncols == 0
        title = J.capitalize()
        subplot_tc_vs_lattice(results_dir, J, Ls, observable_names, labels, colors, N=N, xlabel=xlabel, ylabel=ylabel, title=title)
    handles, labels = get_unique_legend_handles_labels(plt.gcf())
    plt.figlegend(handles, labels, ncol=2, loc=(0.1,0.9), fancybox=True, fontsize=8)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85,wspace=0.075)
    output_dir = os.path.join(results_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "tc_vs_L.pdf"))
    plt.close()


### plot execution times

def subplot_time(results_dir, J, Ls, N, encoder_names, labels, colors, xlabel=True, ylabel=True, title=None):
    encoder_names_singlescale = [name for name in encoder_names if "multiscale" not in name]
    data = pd.read_csv(os.path.join(results_dir, "processed", "gathered.csv"))
    data = data[data.N==N]
    total_times = (data.generation_time+data.preprocessing_time+data.training_time).values
    y_min, y_max = y_minmax(total_times, np.zeros_like(total_times))
    data_Ls = data[(data.J==J) & (data.L!="None")]
    data_inf = data[(data.J==J) & (data.L=="None")]
    x = np.arange(len(Ls))
    width, shifts = bar_width_shifts(len(encoder_names_singlescale))
    for (name, shift) in zip(encoder_names_singlescale, shifts):
        subdata = data_Ls[data_Ls.observable.eq(name)]
        plt.bar(x+shift, subdata.generation_time.values, width, color=colors["magnetization"],alpha=1.0)
        plt.bar(x+shift, subdata.preprocessing_time.values+subdata.training_time.values, width, bottom=subdata.generation_time.values, color=colors[name],alpha=1.0)
    width, shifts = bar_width_shifts(len(encoder_names))
    for (name, shift) in zip(encoder_names, shifts):
        subdata = data_inf[data_inf.observable.eq(name)]
        plt.bar([len(x)+shift], subdata.generation_time.values, width, color=colors["magnetization"], label=labels["magnetization"],alpha=1.0)
        plt.bar([len(x)+shift], subdata.preprocessing_time.values+subdata.training_time.values, width, bottom=subdata.generation_time.values, color=colors[name], label=labels[name],alpha=1.0)
    plt.xticks(list(x)+[len(x)], list(Ls)+[r"$\infty$"])
    plt.ylim(y_min, y_max)
#   plt.yscale("log")
    if xlabel:
        plt.xlabel(r"Lattice size ($L$)", fontsize=8)
    if ylabel:
        plt.ylabel("Time (min)", fontsize=8)
    else:
        ax = plt.gca()
        ax.set_yticklabels([])
    if title is not None:
        plt.title(title, fontsize=8)


def plot_time(results_dir, Js, Ls, N, encoder_names, labels, colors):
    plt.figure(figsize=(figsize[0],figsize[0]))
    nrows, ncols = 1, len(Js)
    for (index, J) in enumerate(Js):
        plt.subplot(nrows, ncols, index+1)
        xlabel = index//ncols == nrows-1
        ylabel = index%ncols == 0
        title = J.capitalize()
        subplot_time(results_dir, J, Ls, N, encoder_names, labels, colors, xlabel=xlabel, ylabel=ylabel, title=title)
    handles, labels = get_unique_legend_handles_labels(plt.gcf())
    plt.figlegend(handles, labels, ncol=2, loc=(0.1,0.9), fancybox=True, fontsize=8)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85,wspace=0.075)
    output_dir = os.path.join(results_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "time.pdf"))
    plt.close()


### plot correlation to magnetization

def subplot_cor(results_dir, J, Ls, Ns, encoder_name, colors, xlabel=True, ylabel=True, title=None):
    data = pd.read_csv(os.path.join(results_dir, "processed", "gathered.csv"))
    data = data[(data.J==J) & (data.observable==encoder_name) & (data.L!="None")]
    y_min, y_max = y_minmax(data.cor_magnetization_mean.values, data.cor_magnetization_std.values)
    x = np.arange(len(Ls))
    width, shifts = bar_width_shifts(len(Ns))
    for (N, shift) in zip(Ns, shifts):
        plt.bar(x+shift, data[data.N.eq(N)].cor_magnetization_mean.values, width, yerr=bar_yerrs(data[data.N.eq(N)].cor_magnetization_mean.values, data[data.N.eq(N)].cor_magnetization_std.values), capsize=2, ecolor=colors[str(N)], color=colors[str(N)], label=str(int(N)))
    plt.xticks(x, Ls)
    plt.ylim(0, 11.5)
    if xlabel:
        plt.xlabel(r"Lattice size ($L$)", fontsize=8)
    if ylabel:
        plt.ylabel(r"Error $\nu (\%)$", fontsize=8)
    if title is not None:
        plt.title(title, fontsize=8)
    if title == "Antiferromagnetic":
        ax = plt.gca()
        ax.set_yticklabels([])



def plot_cor(results_dir, J, Ls, Ns, encoder_name, colors):
    plt.figure(figsize=(figsize[0],figsize[0]))
    nrows, ncols = 1, len(Js)
    for (index, J) in enumerate(Js):
        plt.subplot(nrows, ncols, index+1)
        xlabel = index//ncols == nrows-1
        ylabel = index%ncols == 0
        title = J.capitalize()
        subplot_cor(results_dir, J, Ls, Ns, encoder_name, colors, xlabel=xlabel, ylabel=ylabel, title=title)
    handles, labels = get_unique_legend_handles_labels(plt.gcf())
    plt.figlegend(handles, labels, ncol=3, loc="upper center", fancybox=True, fontsize=8, title=r"Samples per temperature ($N$)")
    plt.tight_layout()
    plt.subplots_adjust(top=0.78)
    output_dir = os.path.join(results_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "cor_magnetization.pdf"))
    plt.close()


### plot correlation to onsager

def subplot_onsager(results_dir, J, Ls, observable_names, labels, colors, N=None, xlabel=True, ylabel=True, title=None):
    x = 1/np.array(Ls)
    x_pnts = np.linspace(0, x.max(), 100, endpoint=True)
    data = pd.read_csv(os.path.join(results_dir, "processed", "gathered.csv"))
    data = data[(data.J==J) & (data.N==N)]
    for name in observable_names:
        y = data[(data.observable==name) & (data.L!="None")].cor_onsager_mean.values
        slope = float( data[(data.observable==name) & (data.L=="None")].cor_onsager_slope.values )
        intercept = float( data[(data.observable==name) & (data.L=="None")].cor_onsager_yintercept.values )
        yhat = slope*x_pnts + intercept
        plt.scatter(x, y, alpha=0.7, s=12, linewidth=0.25, color=colors[name], label=labels[name])
        plt.plot(x_pnts, yhat, alpha=0.7, color=colors[name])
        plt.ylim(0,24)
        plt.xlim(0,0.069)
    if xlabel:
        plt.xlabel(r"Inverse lattice size ($L^{-1}$)", fontsize=8)
    if ylabel:
        plt.ylabel(r"Distance $(\%)$", fontsize=8)
    else:
        ax = plt.gca()
        ax.set_yticklabels([])
    if title is not None:
        plt.title(title, fontsize=8)


def plot_onsager(results_dir, Js, Ls, observable_names, labels, colors, N=None):
    plt.figure(figsize=(figsize[0],figsize[0]))
    nrows, ncols = 1, len(Js)
    for (index, J) in enumerate(Js):
        plt.subplot(nrows, ncols, index+1)
        xlabel = index//ncols == nrows-1
        ylabel = index%ncols == 0
        title = J.capitalize()
        subplot_onsager(results_dir, J, Ls, observable_names, labels, colors, N=N, xlabel=xlabel, ylabel=ylabel, title=title)
    handles, labels = get_unique_legend_handles_labels(plt.gcf())
    plt.figlegend(handles, labels, ncol=2, loc=(0.1,0.9), fancybox=True, fontsize=8)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85,wspace=0.075)
    output_dir = os.path.join(results_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "cor_onsager.pdf"))
    plt.close()


### tabulate symmetry generators

def tabulate_generators(results_dir, Js, encoder_name):
    output_dir = os.path.join(results_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(results_dir, "processed", "generators.json"), "r") as fp:
        gens = json.load(fp)
    for J in Js:
        gens[J] = gens[J][encoder_name]
    generator_types = ["spatial", "internal"]
    stds = [gens[J][gen_type]["std"] for J in Js for gen_type in generator_types]
    max_precision = 1 + max([1-int(np.log10(s)) for s in stds   ])
    S_columns = 2*"S[table-format=-1.{:d}(2),table-align-uncertainty=true]".format(max_precision)
    with open(os.path.join(output_dir, "generators.tex"), "w") as fp:
        fp.write("\\begin{{tabular}}{{c{}}}\n".format(S_columns))
        fp.write("\\toprule\n")
        fp.write("\\quad & {Spatial} & {Internal} \\\\\n")
        fp.write("\\midrule\n")
        for J in Js:
            fp.write(J.capitalize())
            for gen_type in generator_types:
                precision = 2-int(np.log10(gens[J][gen_type]["std"]))
                fp.write(" & {{:.{:d}f}}\\pm {{:.{:d}f}}".format(precision, precision).format(gens[J][gen_type]["mean"], gens[J][gen_type]["std"]))
            fp.write(" \\\\\n")
        fp.write("\\bottomrule\n")
        fp.write("\\end{tabular}")


if __name__ == "__main__":
    results_dir = "results"
    Js = ["ferromagnetic", "antiferromagnetic"]
    Ls = [16, 32, 64, 128]
    Ns = [8, 16, 32, 64, 128, 256]
    Ts = [2.0, 2.5]

    # old colors
    # red,orange,yellow,purple,blue,green = "red", "orange", "magenta", "purple", "blue", "green"

    # new colors
    red,orange,yellow,purple,blue,green = "#e41a1c", "#ff7f00", "#fee08b", "#984ea3", "#377eb8", "#4daf4a"

    print("Plotting statistics . . . ")
    observable_names = ["magnetization", "latent", "latent_equivariant"]
    titles = {"magnetization": "Magnetization", "latent": "Baseline-AE", "latent_equivariant": "GE-AE"}
    colors = {"tc": red, "fit": blue}
    for what in ["distribution", "order", "binder"]:
        plot_stat(results_dir, Js, observable_names, 128, N=256, fold=0, seed=0, colors=colors, titles=titles, what=what)

    print("Plotting error . . . ")
    encoder_names = ["latent", "latent_equivariant", "latent_multiscale_4"]
    observable_names = ["magnetization", "latent", "latent_equivariant", "latent_multiscale_4"]
    labels = {"magnetization": "Magnetization", "latent": "Baseline-AE", "latent_equivariant": "GE-AE", "latent_multiscale_4": "GE-AE (multiscale)"}
    colors = {"magnetization": red, "latent": green, "latent_equivariant": blue, "latent_multiscale_4": purple}
    for J in Js:
        plot_tc(results_dir, J, Ls, Ns, encoder_names, labels, colors, grid_dims=(2, 2))
    plot_tc_extrapolate(results_dir, Js, Ns, encoder_names, labels, colors)
    plot_tc_vs_lattice(results_dir, Js, Ls, observable_names, labels, colors, N=256)

    print("Plotting time . . . ")
    plot_time(results_dir, Js, Ls, 256, encoder_names, labels, colors)

    print("Plotting correlations . . . ")
    plot_onsager(results_dir, Js, Ls, observable_names, labels, colors, N=256)
    colors = {str(N): color for (N, color) in zip(Ns, [red, orange, yellow, purple, blue, green])}
    plot_cor(results_dir, Js, Ls, Ns, "latent", colors)

    print("Plotting anomaly detection . . . ")
    ad_labels = {"baseline": "Baseline-AE", "ge": "GE-AE"}
    ad_encoder_names = ['baseline','ge']
    ad_colors = {'baseline':green,'ge':blue}
    plot_ad(results_dir,Ts,ad_encoder_names,ad_labels,ad_colors)

    print("Tabulating generators . . . ")
    tabulate_generators(results_dir, Js, "latent_equivariant")

    print("Done!")
