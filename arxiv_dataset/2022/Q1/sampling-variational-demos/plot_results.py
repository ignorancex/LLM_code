import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


def plot_samples2d(file, var1, var2, ax=None):
    ax = ax or plt.gca()
    data = pd.read_csv(file, comment='#')
    ax.scatter(data[var1][::10], data[var2][::10], marker='.', color='b', label='built-in NUTS')


def plot_samples1d(file, var, ax=None):
    ax = ax or plt.gca()
    data = pd.read_csv(file, comment='#')
    ax.hist(data[var], color='b', density=True, bins=50, label='built-in NUTS')


def plot_advi2d(file, var1, var2, meanfield=True, nsig=(1,), ax=None):
    ax = ax or plt.gca()
    # Note on format: first row of the CSV is the estimated mean from ADVI, and
    # the rest of the rows are samples from q.
    data = pd.read_csv(file, comment='#')
    mean = data.loc[0]
    ax.plot(0,0,'-',color='r',label='built-in ADVI')
    ax.plot(mean[var1], mean[var2], marker='+', color='r')
    samples = data.loc[1:]
    t = np.linspace(0, 2*np.pi)
    c, s = np.cos(t), np.sin(t)
    if meanfield:
        std1 = np.sqrt(np.sum((samples[var1] - mean[var1])**2) / len(samples))
        std2 = np.sqrt(np.sum((samples[var2] - mean[var2])**2) / len(samples))
        for sig in nsig:
            ax.plot(mean[var1]+c*std1*sig, mean[var2]+s*std2*sig, color='r')
    else:
        raise NotImplementedError() # TODO


def plot_advi1d(file, var, ax=None):
    ax = ax or plt.gca()
    data = pd.read_csv(file, comment='#')
    mean = data.loc[0][var]
    samples = data.loc[1:]
    std = np.sqrt(np.sum((samples[var] - mean)**2) / len(samples))
    t = np.linspace(mean-3.5*std, mean+3.5*std)
    p = np.exp(-0.5*(t-mean)**2/std**2)/std/np.sqrt(2*np.pi)
    ax.plot(t, p, color='r', label='built-in ADVI')



def plot_isvi2d(file, var1, var2, n_plot=50, nsig=(1,), min_sig=.01, ax=None):
    ax = ax or plt.gca()
    data = pd.read_csv(file, comment='#')
    mu1, mu2 = data["mu_" + var1], data["mu_" + var2]
    log_sig1, log_sig2 = data["omega_" + var1], data["omega_" + var2]
    sig1, sig2 = np.exp(log_sig1), np.exp(log_sig2)
    sig1[sig1 < min_sig] = min_sig  # clip minimum variance for the purposes of plotting
    sig2[sig2 < min_sig] = min_sig  # clip minimum variance for the purposes of plotting
    idx = list(map(int, np.linspace(0, len(data)-1, n_plot)))
    t = np.linspace(0, 2*np.pi)
    c, s = np.cos(t), np.sin(t)
    for m1, s1, m2, s2 in zip(mu1.loc[idx], sig1.loc[idx], mu2.loc[idx], sig2.loc[idx]):
        for sig in nsig:
            ax.plot(m1+c*s1*sig, m2+s*s2*sig, color='g')
    ax.plot(0,0,'-',color='g',label="ours (λ=2.0)")


def plot_isvi1d(file, var, n_plot=50, height_scale=0.1, min_sig=.01, ax=None):
    ax = ax or plt.gca()
    data = pd.read_csv(file, comment='#')
    mu = data["mu_" + var]
    log_sig = data["omega_" + var]
    sig = np.exp(log_sig)
    sig[sig < min_sig] = min_sig  # clip minimum variance for the purposes of plotting
    idx = list(map(int, np.linspace(0, len(data)-1, n_plot)))
    for m, s in zip(mu.loc[idx], sig.loc[idx]):
        t = np.linspace(m-3.5*s, m+3.5*s)
        p = np.exp(-0.5*(t-m)**2/s**2)/s/np.sqrt(2*np.pi)
        ax.plot(t, p*height_scale, color='g')
    ax.plot(0,0,'-',color='g',label="ours (λ=2.0)")


def plot_grid(sample_file, isvi_file, advi_file):
    data = pd.read_csv(sample_file, comment='#')
    params = [c for c in data.columns if '__' not in c]
    fig, axs = plt.subplots(len(params), len(params))
    for i in range(len(params)):
        for j in range(len(params)):
            if i==j:
                plot_samples1d(sample_file, params[i], ax=axs[i,j])
                plot_isvi1d(isvi_file, params[i], ax=axs[i,j])
                plot_advi1d(advi_file, params[i], ax=axs[i,j])
            else:
                plot_samples2d(sample_file, params[j], params[i], ax=axs[i,j])
                plot_isvi2d(isvi_file, params[j], params[i], ax=axs[i,j])
                plot_advi2d(advi_file, params[j], params[i], ax=axs[i,j])
            if j==0:
                axs[i,j].set_ylabel(params[i])
            if i==len(params)-1:
                axs[i,j].set_xlabel(params[j])
    fig.tight_layout()
    return fig

def plot_traces(file):
    data = pd.read_csv(file, comment='#')
    params = [c for c in data.columns if '__' not in c]
    data[params].plot()


def main(args):
    if args.type == 'grid':
        plot_grid(Path(args.problem) / f"nuts_{args.chain}.csv",
            Path(args.problem) / f"isvi_{args.lam}_{args.chain}.csv",
            Path(args.problem) / "advi.csv")
    elif args.type == 'traces':
        plot_traces(Path(args.problem) / f"nuts_{args.chain}.csv")
        plot_traces(Path(args.problem) / f"isvi_{args.lam}_{args.chain}.csv")
    plt.suptitle(f"{args.problem}  λ={args.lam}  chain={args.chain}")
    plt.gcf().tight_layout()
    plt.savefig(fig_path / f"{args.problem}_{args.lam}_{args.chain}_{args.type}.pdf")
    if args.display:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", default="cigar", type=str)
    parser.add_argument("--lam", default="2.0", type=str)
    parser.add_argument("--chain", default=1, type=int, choices=[1,2,3,4])
    parser.add_argument("--display", action='store_true')
    parser.add_argument("--type", default='grid', choices=['grid', 'traces'])
    args = parser.parse_args()

    # throw an error if args.lam does not parse as a float greater than or equal to 1.0
    l = float(args.lam)
    assert 1.0 <= l

    fig_path = Path("figures")
    fig_path.mkdir(exist_ok=True)

    main(args)