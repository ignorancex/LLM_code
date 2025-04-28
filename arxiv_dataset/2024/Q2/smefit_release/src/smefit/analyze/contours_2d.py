# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import seaborn as sns
from matplotlib import patches, transforms
from matplotlib.patches import Ellipse


def split_solution(full_solution):
    """Split the posterior solution"""

    min_val = min(full_solution)
    max_val = max(full_solution)
    mid = np.mean([max_val, min_val])

    # solution 1 should be closer to 0
    solution1 = full_solution[full_solution < mid]
    solution2 = full_solution[full_solution > mid]
    if min(abs(solution2)) < min(abs(solution1)):
        solution1, solution2 = solution2, solution1

    return solution1, solution2


def confidence_ellipse(
    coeff1, coeff2, ax, facecolor="none", confidence_level=95, **kwargs
):
    """
    Draws 95% C.L. ellipse for data points `x` and `y`

    Parameters
    ----------
    coeff1: array_like
        ``(N,) ndarray`` containing ``N`` posterior samples for the first coefficient
    coeff2: array_like
        ``(N,) ndarray`` containing ``N`` posterior samples for the first coefficient
    ax: matplotlib.axes
        Axes object to plot on
    facecolor: str, optional
        Color of the ellipse
    **kwargs
        Additional plotting settings passed to matplotlib.patches.Ellipse

    Returns
    -------
    matplotlib.patches.Ellipse
        Ellipse object

    """

    if coeff1.size != coeff2.size:
        raise ValueError("coeff1 and coeff2 must be the same size")

    # construct covariance matrix of coefficients
    cov = np.cov(coeff1, coeff2)

    # diagonalise
    eig_val, eig_vec = np.linalg.eig(cov)

    # eigenvector with largest eigenvalue
    eig_vec_max = eig_vec[:, np.argmax(eig_val)]

    # angle of eigenvector with largest eigenvalue with the horizontal axis
    cos_th = eig_vec[0, np.argmax(eig_val)] / np.linalg.norm(eig_vec_max)
    if eig_vec_max[1] > 0:
        inclination = np.arccos(cos_th)
    else:
        # pay attention to range of arccos (extend to [0, -\pi] domain)
        inclination = -np.arccos(cos_th)

    eigval_sort = np.sort(eig_val)

    chi2_qnt = scipy.stats.chi2.ppf(confidence_level / 100.0, 2)

    ell_radius_x = np.sqrt(chi2_qnt * eigval_sort[-1])
    ell_radius_y = np.sqrt(chi2_qnt * eigval_sort[-2])

    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs
    )

    mean_coeff1 = np.median(coeff1)
    mean_coeff2 = np.median(coeff2)

    transf = (
        transforms.Affine2D().rotate(inclination).translate(mean_coeff1, mean_coeff2)
    )

    ellipse.set_transform(transf + ax.transData)

    return ax.add_patch(ellipse)


def plot_contours(
    ax,
    posterior,
    ax_labels,
    coeff1,
    coeff2,
    kde,
    clr_idx,
    confidence_level=95,
    double_solution=None,
):
    """

    Parameters
    ----------
    ax: matplotlib.axes
        Axes object to plot on
    posterior: pandas.DataFrame
        Posterior samples per coefficient
    ax_labels: list
        Latex names
    coeff1: str
        Name of first coefficient
    coeff2: str
        Name of second coefficient
    kde: bool
        Performs kernel density estimate (kde) when quadratics are turned on
    clr_idx: int
        Color index that makes sure each fit gets associated a different color
    confidence_level: int, optional
        Confidence level interval, set to 95% by default
    double_solution: dict, optional
        Dictionary of operators with double (disjoint) solution per fit

    Returns
    -------
    hndls: list
        List of Patch objects
    """

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    x_values = posterior[coeff2].values
    y_values = posterior[coeff1].values

    # perform kde for quadratic EFT fit
    if kde:
        solution1x = solution2x = x_values
        if coeff2 in double_solution:
            solution1x, solution2x = split_solution(posterior[coeff2])

        solution1y = solution2y = y_values
        if coeff1 in double_solution:
            solution1y, solution2y = split_solution(posterior[coeff1])

        sns.kdeplot(
            x=x_values,
            y=y_values,
            levels=[1 - confidence_level / 100.0, 1.0],
            bw_adjust=1.2,
            ax=ax,
            fill=True,
            alpha=0.3,
            color=colors[clr_idx],
        )
        sns.kdeplot(
            x=x_values,
            y=y_values,
            levels=[1 - confidence_level / 100.0],
            bw_adjust=1.2,
            ax=ax,
            alpha=1,
            color=colors[clr_idx],
            label=confidence_level,
        )
        ax.scatter(
            np.mean(solution1x),
            np.mean(solution1y),
            c=colors[clr_idx],
            s=50,
            marker="o",
        )
        ax.scatter(
            np.mean(solution2x),
            np.mean(solution2y),
            c=colors[clr_idx],
            s=50,
            marker="o",
        )

        hndls = (
            patches.Patch(ec=colors[clr_idx], fc=colors[clr_idx], fill=True, alpha=0.3),
            patches.Patch(
                ec=colors[clr_idx], fc=colors[clr_idx], fill=False, alpha=1.0
            ),
        )

    else:  # draw ellipses for linear EFT fit
        p1 = confidence_ellipse(
            x_values,
            y_values,
            ax,
            alpha=1,
            edgecolor=colors[clr_idx],
            confidence_level=confidence_level,
        )

        p2 = confidence_ellipse(
            x_values,
            y_values,
            ax,
            alpha=0.3,
            facecolor=colors[clr_idx],
            edgecolor=None,
            confidence_level=confidence_level,
        )

        ax.scatter(
            np.mean(x_values, axis=0),
            np.mean(y_values, axis=0),
            c=colors[clr_idx],
            s=50,
            marker="o",
        )

        hndls = (p1, p2)

    plt.xlabel(ax_labels[1], fontsize=26)
    plt.ylabel(ax_labels[0], fontsize=26)

    plt.tick_params(which="both", direction="in", labelsize=22)

    return hndls
