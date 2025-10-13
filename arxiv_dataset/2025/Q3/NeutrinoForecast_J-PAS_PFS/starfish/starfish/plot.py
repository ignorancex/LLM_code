from .set import *
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["text.usetex"] = True

__all__ = ["check_wiggle", "get_figure"]


def check_wiggle(PK, window_array, z_index=0):
    if np.isscalar(window_array):
        window_array = np.array([window_array])
    else:
        window_array = np.array(window_array)

    fig, axes = plt.subplots(2, 1)
    fig.set_dpi(250)

    from .galaxy.power import P_smooth

    pks = P_smooth(settings.k_fid, settings.z[z_index], PK, False).reshape(-1)
    axes[0].loglog(settings.k_fid, pks, "k")

    for win in window_array:
        pks_smooth = P_smooth(
            settings.k_fid, settings.z[z_index], PK, True, window_length=win
        ).reshape(-1)
        axes[0].loglog(settings.k_fid, pks_smooth, label=f"window = {win}")
        axes[1].plot(settings.k_fid, pks / pks_smooth)

    fig.legend(loc="upper left")
    axes[0].grid()
    axes[1].grid()

    axes[1].set_xlim(0, np.max(settings.k_max))
    axes[1].set_ylim(0.9, 1.1)
    axes[1].set_xlabel(r"$k\;[h\;\mathrm{Mpc^{-1}}]$")


def plot_ellipse(mean, cov, ax):
    """
    ellipse plotting
    """

    mean = np.array(mean)
    cov = np.array(cov)

    if cov.shape != (2, 2):
        raise ValueError(
            f"the covariance matrix must be of shape (2,2), but {cov.shape} is given!"
        )
    elif not np.allclose(cov[0, 1], cov[1, 0]):
        raise ValueError(
            f"the input matrix is not symmetric, with (0,1)-element being {cov[0,1]} and (1,0)-element being {cov[1,0]}!"
        )
    elif not np.all(np.linalg.eigvals(cov) > 0):
        raise ValueError(
            f"the input matrix is not positive definite! i.e. all of its eigen values must be positive."
        )

    l1 = (cov[0, 0] + cov[1, 1]) / 2 + np.sqrt(
        ((cov[0, 0] - cov[1, 1]) / 2) ** 2 + cov[0, 1] * cov[1, 0]
    )
    l2 = (cov[0, 0] + cov[1, 1]) / 2 - np.sqrt(
        ((cov[0, 0] - cov[1, 1]) / 2) ** 2 + cov[0, 1] * cov[1, 0]
    )

    if np.allclose(cov[0, 1], 0):  # actually no rotation
        if cov[0, 0] >= cov[1, 1]:
            angle = 0
        else:
            angle = np.pi / 2
    else:
        angle = np.arctan2(l1 - cov[0, 0], cov[0, 1])

    t = np.linspace(0, 2 * np.pi, num=1000)

    for A in np.sqrt([2.3, 6.18]):
        x = A * (
            np.sqrt(l1) * np.cos(angle) * np.cos(t)
            - np.sqrt(l2) * np.sin(angle) * np.sin(t)
        )
        y = A * (
            np.sqrt(l1) * np.sin(angle) * np.cos(t)
            + np.sqrt(l2) * np.cos(angle) * np.sin(t)
        )
        ax.plot(x + mean[0], y + mean[1])
    ax.plot(mean[0], mean[1], "r+", markersize=10)


def get_axes(var_order1, var_order2):
    """
    axes settings
    """

    index1 = []
    index2 = []
    for index, var_order in ((index1, var_order1), (index2, var_order2)):
        for v in var_order:
            try:
                (i,) = np.where(np.array(settings.var_name) == v)
                index.append(i[0])
            except:
                raise ValueError(f"cannot find {v} in var_name!")

    x_num = len(var_order1)
    y_num = len(var_order2)

    fig, axes = plt.subplots(y_num, x_num, sharex="col")
    fig.set_dpi(250)
    fig.set_size_inches(len(var_order1) / 7 * 8, len(var_order2) / 7 * 8)
    fig.subplots_adjust(hspace=0, wspace=0)

    if index1 == index2:
        for i in range(y_num):  # y
            for j in range(x_num):  # x
                if j > i:  # upper triangle
                    axes[i, j].axis("off")
                else:
                    if j == 0:
                        axes[i, j].set_ylabel(
                            settings.var_exp[index2[i]], fontsize="large"
                        )
                    if i == y_num - 1:
                        axes[i, j].set_xlabel(
                            settings.var_exp[index1[j]], fontsize="large"
                        )
                        axes[i, j].tick_params(axis="x", labelrotation=45)

                    if i == j:  # diagonal
                        axes[i, j].tick_params(
                            bottom=True,
                            top=False,
                            left=False,
                            right=False,
                            direction="in",
                        )
                        axes[i, j].set_yticks([])
                        axes[i, j].set_ylabel("")
                    else:  # lower triangle
                        axes[i, j].tick_params(
                            length=5,
                            bottom=True,
                            top=True,
                            left=True,
                            right=True,
                            direction="in",
                        )
                        if j > 0:
                            axes[i, j].sharey(axes[i, j - 1])
                            axes[i, j].tick_params(labelleft=False)
    else:
        for i in range(y_num):  # y
            for j in range(x_num):  # x
                if j == 0:
                    axes[i, j].set_ylabel(settings.var_exp[index2[i]], fontsize="large")
                if i == y_num - 1:
                    axes[i, j].set_xlabel(settings.var_exp[index1[j]], fontsize="large")
                    axes[i, j].tick_params(axis="x", labelrotation=45)

                axes[i, j].tick_params(
                    length=5,
                    bottom=True,
                    top=True,
                    left=True,
                    right=True,
                    direction="in",
                )

                if j > 0:
                    axes[i, j].sharey(axes[i, j - 1])
                    axes[i, j].tick_params(labelleft=False)

    return fig, axes, index1, index2


def get_xlim(ax, i):
    if i == np.where(np.array(settings.var_name) == "m_nu")[0][0]:
        ax.set_xlim(0)
    # if i==np.where(np.array(var_name)=='h')[0][0]:
    #     ax.set_xlim(0.61,0.74)
    #     ax.set_xticks([0.64,0.70])
    # elif i==np.where(np.array(var_name)=='omega_m0')[0][0]:
    #     ax.set_xlim(0.29,0.34)
    #     ax.set_xticks([0.30,0.33])
    # elif i==np.where(np.array(var_name)=='omega_b0')[0][0]:
    #     ax.set_xlim(0.044,0.054)
    #     ax.set_xticks([0.046,0.052])
    # elif i==np.where(np.array(var_name)=='sigma_8')[0][0]:
    #     ax.set_xlim(0.78,0.84)
    #     ax.set_xticks([0.79,0.82])
    # elif i==np.where(np.array(var_name)=='n_s')[0][0]:
    #     ax.set_xlim(0.91,1.03)
    #     ax.set_xticks([0.94,1.00])
    # elif i==np.where(np.array(var_name)=='m_nu')[0][0]:
    #     ax.set_xlim(0,0.6)
    #     ax.set_xticks([0.1,0.4])
    # elif i==np.where(np.array(var_name)=='w0')[0][0]:
    #     ax.set_xlim(-1.15,-0.85)
    #     ax.set_xticks([-1.1,-0.9])


def get_ylim(ax, i):
    if i == np.where(np.array(settings.var_name) == "m_nu")[0][0]:
        ax.set_ylim(0)
    # if i==np.where(np.array(var_name)=='h')[0][0]:
    #     ax.set_ylim(0.61,0.74)
    #     ax.set_yticks([0.64,0.70])
    # elif i==np.where(np.array(var_name)=='omega_m0')[0][0]:
    #     ax.set_ylim(0.29,0.34)
    #     ax.set_yticks([0.30,0.33])
    # elif i==np.where(np.array(var_name)=='omega_b0')[0][0]:
    #     ax.set_ylim(0.044,0.054)
    #     ax.set_yticks([0.046,0.052])
    # elif i==np.where(np.array(var_name)=='sigma_8')[0][0]:
    #     ax.set_ylim(0.78,0.84)
    #     ax.set_yticks([0.79,0.82])
    # elif i==np.where(np.array(var_name)=='n_s')[0][0]:
    #     ax.set_ylim(0.91,1.03)
    #     ax.set_yticks([0.94,1.00])
    # elif i==np.where(np.array(var_name)=='m_nu')[0][0]:
    #     ax.set_ylim(0,0.6)
    #     ax.set_yticks([0.1,0.4])
    # elif i==np.where(np.array(var_name)=='w0')[0][0]:
    #     ax.set_ylim(-1.15,-0.85)
    #     ax.set_yticks([-1.1,-0.9])


def get_figure(cov_all, var_order1, var_order2=None):
    """
    figure plotting
    """

    if var_order2 == None:
        var_order2 = var_order1

    fig, axes, index1, index2 = get_axes(var_order1, var_order2)
    cov_all_sub = np.empty((2, 2))
    try:
        lnbs_value = np.loadtxt("lnbs.csv")
        var_value = np.concatenate(
            (settings.cosmo_value, np.zeros(len(settings.z)), lnbs_value)
        )
    except FileNotFoundError:
        var_value = settings.cosmo_value

    if index1 == index2:
        for i in range(len(index2)):
            for j in range(len(index1)):
                y = index2[i]
                x = index1[j]
                mean = np.array(var_value)[[x, y]]

                if i > j:
                    cov_all_sub[0, 0] = cov_all[x, x]
                    cov_all_sub[1, 1] = cov_all[y, y]
                    cov_all_sub[0, 1] = cov_all[x, y]
                    cov_all_sub[1, 0] = cov_all[y, x]

                    plot_ellipse(mean=mean, cov=cov_all_sub, ax=axes[i, j])

                    get_ylim(axes[i, j], y)
                    get_xlim(axes[i, j], x)

                elif i == j:
                    scale = np.sqrt(cov_all[x, x])
                    array = np.linspace(mean[0] - 3 * scale, mean[0] + 3 * scale, 100)
                    prob = norm.pdf(array, loc=mean[0], scale=scale)

                    axes[i, j].plot(array, prob / np.max(prob))
    else:
        for i in range(len(index2)):
            for j in range(len(index1)):
                y = index2[i]
                x = index1[j]
                mean = np.array(var_value)[[x, y]]

                cov_all_sub[0, 0] = cov_all[x, x]
                cov_all_sub[1, 1] = cov_all[y, y]
                cov_all_sub[0, 1] = cov_all[x, y]
                cov_all_sub[1, 0] = cov_all[y, x]

                plot_ellipse(mean=mean, cov=cov_all_sub, ax=axes[i, j])

                get_ylim(axes[i, j], y)
                get_xlim(axes[i, j], x)

    return fig
