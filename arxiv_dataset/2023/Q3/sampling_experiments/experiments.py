import sys

import matplotlib

import utils
from metrics import energy_dist
from utils import cached_function

#matplotlib.use('Agg')
matplotlib.use('pdf')
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'font.size': 10.95,
    'text.usetex': True,
    'pgf.rcfonts': False,
    # 'legend.framealpha': 0.5,
    'text.latex.preamble': r'\usepackage{times} \usepackage{amsmath} \usepackage{amsfonts} \usepackage{amssymb}'
})

from algorithms import *
from pathlib import Path
import matplotlib.pyplot as plt


def plot_mc_logpartition(d: int, n_reps: int = 11):
    colors = [u'#a06010', u'#d62728', u'#e377c2', u'#2ca02c', u'#ff7f0e', u'#9467bd', u'#17becf', u'#7f7f7f',
              u'#bcbd22', u'#1f77b4', u'#44FF44', u'#000000']
    # colors = [(0, 0, 0), (215, 0, 0), (140, 60, 255), (2, 136, 0), (0, 172, 199), (152, 255, 0), (255, 127, 209),
    #           (108, 0, 79), (255, 165, 48),
    #           (0,0,157), (134,112,104), (0,73,66), (79,42,0), (0,253,207), (188,183,255)]

    # todo: put the legend to the right of the plot
    # todo: use more distinct colors or less lines

    plt.figure(figsize=(6, 4))

    torch.manual_seed(0)
    np.random.seed(0)

    lip_consts = [4**k for k in range(12)]
    ns = [2**k for k in range(20)]
    # plot upper bounds first, such that they are in the background of the plot
    for i, lip_const in enumerate(reversed(lip_consts)):
        beta = lip_const / np.sqrt(d)
        true_dist = LinearGibbsDistribution(params=torch.as_tensor([beta], dtype=torch.float32))
        upper_bounds = []
        for n in ns:
            upper_bounds.append(mc_logpartition_upper_bound(n, d, lip_const, delta=0.5))

        color = colors[i % len(colors)]
        plt.loglog(ns, upper_bounds, linestyle='--', color=color)

    for i, lip_const in enumerate(reversed(lip_consts)):
        beta = lip_const / np.sqrt(d)
        true_dist = LinearGibbsDistribution(params=torch.as_tensor([beta], dtype=torch.float32))
        dist_str = f'linear_1d_{beta:g}'
        # print(f'{lip_const=}')
        mc_errors = []
        upper_bounds = []
        for n in ns:
            apx_values = cached_function(f'{dist_str}_pc', 0,
                                         lambda n, n_batch: MCLogPartition(
                                            UniformCubeDistribution(d=d)).eval(true_dist, n_reps=n_batch, n=n).cpu(),
                                         n=n, n_batch=n_reps)
            ref_value = true_dist.log_partition()
            errors = torch.abs(apx_values - ref_value)
            mc_errors.append(torch.median(errors).item())
            # upper_bounds.append(mc_logpartition_upper_bound(n, d, lip_const, delta=0.5))

        color = colors[i % len(colors)]
        plt.loglog(ns, mc_errors, linestyle='-', color=color, label=f'$|f|_1 = {lip_const}$')
        # plt.loglog(ns, upper_bounds, linestyle='--', color=color)

    plt.xlabel('$n$')
    plt.ylabel(r'Median error $|L_f - \tilde L_f|$')
    plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))
    plt.tight_layout(pad=0.4)
    plt.savefig(f'plots/mc_logpartition_{d}d_new.pdf')
    plt.show()
    plt.close()


def plot_logpartition_errors_ax(ax: plt.Axes, d: int = 3, beta: float = 5, n_batch: int = 1001, rerun: bool = False):
    # d = 3
    # true_dist = LinearGibbsDistribution(params=torch.as_tensor([20.0, 20.0, 20.0]))
    # dist_str = f'linear_3d_20_20_20'

    # plt.figure(figsize=(6, 4))
    params = torch.as_tensor([beta] * d, dtype=torch.float64)
    true_dist = LinearGibbsDistribution(params=params)
    params_str = '_'.join([str(beta)] * d)
    dist_str = f'linear_{d}d_{params_str}'
    true_value = true_dist.log_partition()
    n_per_dims = [1, 2, 3, 4, 6, 8, 10, 13, 16, 20, 26, 32]

    n_arr = np.asarray([1, 2*n_per_dims[-1]**d])
    factor = beta
    ax.loglog(n_arr, factor * n_arr ** (-1 / 3), 'k--', label=r'$\beta n^{-1/3}$')
    ax.loglog(n_arr, factor * n_arr ** (-1 / 2), '--', color='tab:orange', label=r'$\beta n^{-1/2}$')
    ax.loglog(n_arr, factor * n_arr ** (-2 / 3), '--', color='tab:blue', label=r'$\beta n^{-2/3}$')
    ax.loglog(n_arr, factor * n_arr ** (-5 / 6), '--', color='tab:green', label=r'$\beta n^{-5/6}$')

    ns = []
    errors = []
    for n_per_dim in n_per_dims:
        apx_value = cached_function(f'{dist_str}_pc', 0,
                                    lambda n_per_dim: PiecewiseConstantGibbsDistribution(true_dist, n_per_dim).log_partition(),
                                    rerun=rerun, n_per_dim=n_per_dim)
        ns.append(n_per_dim**d)
        errors.append(np.abs(apx_value - true_value))
    ax.loglog(ns, errors, '.-', color='tab:blue', label='PC')

    ns = []
    errors = []
    for n_per_dim in n_per_dims:
        apx_value = cached_function(f'{dist_str}_mc', 0,
                                    lambda n, n_batch: MCLogPartition(UniformCubeDistribution(d=d)).eval(
                                        true_dist,
                                        n_reps=n_batch,
                                        n=n),
                                    rerun=rerun, n=n_per_dim**d, n_batch=n_batch)
        ns.append(n_per_dim ** d)
        errors.append((apx_value - true_value).abs().median().item())
    # print(errors)
    ax.loglog(ns, errors, '.-', color='tab:orange', label='MC')

    ns = []
    errors = []
    for n_per_dim in n_per_dims:
        apx_value = cached_function(f'{dist_str}_pc_mc', 0,
                                    lambda n_per_dim, n, n_batch: MCLogPartition(
                                        PiecewiseConstantGibbsDistribution(true_dist, n_per_dim)).eval(true_dist,
                                                                                                       n_reps=n_batch,
                                                                                                       n=n),
                                    rerun=rerun, n_per_dim=n_per_dim, n=n_per_dim**d, n_batch=n_batch)
        ns.append(2 * n_per_dim ** d)
        errors.append((apx_value - true_value).abs().median().item())
    ax.loglog(ns, errors, '.-', color='tab:green', label='PC+MC')


def plot_logpartition_experiments(n_reps: int = 1001):
    # fig = plt.figure(figsize=(6, 6))
    # axs = fig.subplots(2, 2)
    fig, axs = plt.subplots(2, 2, figsize=(6, 6))
    plot_logpartition_errors_ax(ax=axs[0, 0], d=3, beta=10000, n_batch=n_reps)
    plot_logpartition_errors_ax(ax=axs[0, 1], d=3, beta=40, n_batch=n_reps)
    plot_logpartition_errors_ax(ax=axs[1, 0], d=3, beta=0.1, n_batch=n_reps)
    for ax in [axs[0, 0], axs[0, 1], axs[1, 0]]:
        ax.set_xlabel('Number of function evaluations $n$')
        ax.set_ylabel(r'Median error $|L_f - \tilde L_f|$')
    axs[0, 0].set_title(r'$\beta = 10^4$')
    axs[0, 1].set_title(r'$\beta = 40$')
    axs[1, 0].set_title(r'$\beta = 0.1$')

    axs[-1, -1].axis('off')
    fig.legend(*axs[0, 0].get_legend_handles_labels(), loc='center', bbox_to_anchor=(0.75, 0.3), ncol=1)
    # plt.xlabel('Number of function evaluations $n$')
    # plt.ylabel('Median log-partition estimation error')
    plt.tight_layout(pad=0.4)
    file_path = Path('plots') / 'logpartition_plot_large.pdf'
    utils.ensureDir(file_path)
    plt.savefig(file_path)
    plt.show()
    plt.close(fig)


def plot_sampling_errors_ax(ax: plt.Axes, beta: float, n_samples: int = 10000, d: int = 3, rerun: bool = False):
    params = torch.as_tensor([beta] * d, dtype=torch.float64)
    true_dist = LinearGibbsDistribution(params=params)
    params_str = '_'.join([str(beta)] * d)
    dist_str = f'linear_{d}d_{params_str}'
    true_max = d * beta
    slope_sum = true_max
    # d = 4
    # true_dist = LinearGibbsDistribution(params=torch.as_tensor([5.0, 5.0, 5.0, 5.0]))
    # dist_str = f'linear_4d_5_5_5_5'

    true_samples = cached_function(f'{dist_str}_true_samples', 1234,
                                   lambda n_samples: true_dist.sample(n_samples).cpu(), rerun=rerun, n_samples=n_samples)
    true_samples_ref = [cached_function(f'{dist_str}_true_samples', i,
                                     lambda n_samples: true_dist.sample(n_samples).cpu(), rerun=rerun, n_samples=n_samples)
                        for i in range(3)]
    # loss = geomloss.SamplesLoss(loss='sinkhorn', p=1, blur=1e-3, backend='multiscale')
    # loss_name = 'error'
    loss = lambda s1, s2: energy_dist(s1, s2, device=DefaultDevice.get())
    loss_name = 'energy-dist'
    # n_per_dims = [1, 2, 4, 8, 16]
    n_per_dims = [1, 2, 3, 4, 6, 8, 10, 13, 16, 20, 26, 32]
    n_arr = np.asarray([1, 2*n_per_dims[-1]**d])
    colors = [u'#a06010', u'#d62728', u'#e377c2', u'#2ca02c', u'#ff7f0e', u'#9467bd', u'#17becf', u'#7f7f7f',
              u'#bcbd22', u'#1f77b4', u'#44FF44', u'#000000']
    ax.loglog(n_arr, n_arr ** (-1 / 3), '--', color='tab:green', label=r'$n^{-1/3}$')
    ax.loglog(n_arr, n_arr ** (-1 / 2), '--', color='mediumpurple', label=r'$n^{-1/2}$')
    ax.loglog(n_arr, n_arr ** (-2 / 3), '--', color='tab:pink', label=r'$n^{-2/3}$')

    numerical_errors = [cached_function(f'{dist_str}_true_samples_{loss_name}', 0,
                                      lambda idx, **kwargs: loss(true_samples, true_samples_ref[idx]).item(),
                                rerun=rerun, n_samples=n_samples, idx=i) for i in range(len(true_samples_ref))]
    ax.loglog([1, 2*n_per_dims[-1]**d], [np.max(numerical_errors)] * 2, '--', color='#666666',
               label='Numerical accuracy')

    ns = []
    errors = []
    for n_per_dim in n_per_dims:
        samples = cached_function(f'{dist_str}_pc_samples', 0,
                                  lambda n_per_dim, n_samples: PiecewiseConstantGibbsDistribution(true_dist, n_per_dim).sample(n_samples).cpu(),
                                  rerun=rerun, n_per_dim=n_per_dim, n_samples=n_samples)
        error = cached_function(f'{dist_str}_pc_samples_{loss_name}', 0, lambda **kwargs: loss(true_samples, samples).item(),
                                rerun=rerun, n_per_dim=n_per_dim, n_samples=n_samples)
        ns.append(n_per_dim**d)
        errors.append(error)
    ax.loglog(ns, errors, '.-', color='k', label='PC')

    ns = []
    errors = []
    for n_per_dim in n_per_dims:
        samples = cached_function(f'{dist_str}_mc_samples', 0,
                                  lambda n, n_samples: MCSampler(UniformCubeDistribution(d=d)).sample(
                                        true_dist,
                                        n_samples=n_samples,
                                        n=n).cpu(),
                                  rerun=rerun, n=n_per_dim**d, n_samples=n_samples)
        error = cached_function(f'{dist_str}_mc_samples_{loss_name}', 0, lambda **kwargs: loss(true_samples, samples).item(),
                                rerun=rerun, n=n_per_dim**d, n_samples=n_samples)
        ns.append(n_per_dim ** d)
        errors.append(error)

    # print(errors)
    ax.loglog(ns, errors, '.-', color='tab:blue', label='MC')

    ns = []
    errors = []
    for n_per_dim in n_per_dims:
        samples = cached_function(f'{dist_str}_rs_samples', 0,
                                  lambda n, n_samples: RejectionSampler(UniformCubeDistribution(d=d).shift_by(true_max)).sample(
                                      true_dist,
                                      n_samples=n_samples,
                                      n=n).cpu(),
                                  rerun=rerun, n=n_per_dim ** d, n_samples=n_samples)
        error = cached_function(f'{dist_str}_rs_samples_{loss_name}', 0, lambda **kwargs: loss(true_samples, samples).item(),
                                rerun=rerun, n=n_per_dim ** d, n_samples=n_samples)
        ns.append(n_per_dim ** d)
        errors.append(error)

    # print(errors)
    ax.loglog(ns, errors, '.-', color='tab:red', label='RS')

    ns = []
    errors = []
    for n_per_dim in n_per_dims:
        samples = cached_function(f'{dist_str}_pc_mc_samples', 0,
                                  lambda n_per_dim, n, n_samples: MCSampler(
                                        PiecewiseConstantGibbsDistribution(true_dist, n_per_dim)).sample(true_dist,
                                                                                                       n_samples=n_samples,
                                                                                                       n=n).cpu(),
                                  rerun=rerun, n_per_dim=n_per_dim, n=n_per_dim**d, n_samples=n_samples)
        ns.append(2 * n_per_dim ** d)
        error = cached_function(f'{dist_str}_pc_mc_samples_{loss_name}', 0, lambda **kwargs: loss(true_samples, samples).item(),
                                rerun=rerun, n_per_dim=n_per_dim, n=n_per_dim**d, n_samples=n_samples)
        errors.append(error)
    ax.loglog(ns, errors, '.-', color='tab:cyan', label='PC+MC')

    ns = []
    errors = []
    for n_per_dim in n_per_dims:
        samples = cached_function(f'{dist_str}_pc_rs_samples', 0,
                                  lambda n_per_dim, n, n_samples: RejectionSampler(
                                      PiecewiseConstantGibbsDistribution(true_dist, n_per_dim).shift_by(slope_sum/(2*n_per_dim))).sample(true_dist,
                                                                                                       n_samples=n_samples,
                                                                                                       n=n).cpu(),
                                  rerun=rerun, n_per_dim=n_per_dim, n=n_per_dim ** d, n_samples=n_samples)
        ns.append(2 * n_per_dim ** d)
        error = cached_function(f'{dist_str}_pc_rs_samples_{loss_name}', 0,
                                lambda **kwargs: loss(true_samples, samples).item(),
                                rerun=rerun, n_per_dim=n_per_dim, n=n_per_dim ** d, n_samples=n_samples)
        errors.append(error)
    ax.loglog(ns, errors, '.-', color='tab:orange', label='PC+RS')



def plot_sampling_errors(beta: float, n_samples: int = 10000, d: int = 3, rerun: bool = False):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    plot_sampling_errors_ax(ax, beta=beta, n_samples=n_samples, d=d, rerun=rerun)
    plt.legend()  # todo: better legend placement
    ax.set_xlabel('$n$')
    ax.set_ylabel('Empirical energy distance')
    plt.tight_layout(pad=0.4)
    params_str = '_'.join([f'{beta:g}'] * d)
    dist_str = f'linear_{d}d_{params_str}'
    plt.savefig(f'plots/sampling_plot_{dist_str}_energy-dist.pdf')
    plt.show()
    plt.close(fig)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        device = sys.argv[1]
    else:
        device = 'cpu'
    DefaultDevice.set(device)
    plot_logpartition_experiments(n_reps=10001)
    # plot_logpartition_errors(d=3, param=30)
    # plot_logpartition_errors(d=3, param=20)
    # plot_logpartition_errors(d=3, param=1)
    # plot_logpartition_errors(d=3, param=0.1)
    # plot_logpartition_errors(d=3, param=0.01)
    # plot_logpartition_errors(d=3, param=5)
    # plot_logpartition_errors(d=3, param=10000)
    # plot_logpartition_errors(d=3, param=40)
    # plot_mc_logpartition(d=1, n_reps=1001)
    plot_mc_logpartition(d=1, n_reps=10001)
    plot_sampling_errors(beta=15, n_samples=1000000)
