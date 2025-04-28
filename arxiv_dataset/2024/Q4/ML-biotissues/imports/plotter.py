import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns


def create_pd_df(arr_x: np.ndarray, arr_y: np.ndarray,
                 xheader: str = 'X', yheader: str = 'Y',
                 xcol_ids: list[str] | None = None,
                 ycol_ids: list[str] | None = None) -> tuple[pd.DataFrame, list, list]:
    assert arr_x.shape == arr_y.shape, 'X, Y data shape mismatch'
    if xcol_ids is None:
        xcol_ids = [xheader + str(i) for i in range(arr_x.shape[1])]
    if ycol_ids is None:
        ycol_ids = [yheader + str(i) for i in range(arr_y.shape[1])]
    df = pd.DataFrame(np.c_[arr_x, arr_y], columns=xcol_ids + ycol_ids)
    return df, xcol_ids, ycol_ids


def arr_pairplot(arr_x: np.ndarray, arr_y: np.ndarray,
                 xheader: str = 'X', yheader: str = 'Y',
                 xcol_ids: list[str] | None = None, ycol_ids: list[str] | None = None):
    df, xcol_ids, ycol_ids = create_pd_df(arr_x, arr_y, xheader, yheader, xcol_ids, ycol_ids)
    g = sns.PairGrid(df, x_vars=xcol_ids, y_vars=ycol_ids)
    g.map(sns.scatterplot, size=1)


def truth_plot(arr_t: np.ndarray, arr_p: np.ndarray, nrows: int, ncols: int,
               subtitle: str = 'Subplot', scatter_pts: bool = False, figsize: list = [6, 4]):
    sns.set(style='ticks')
    if nrows > 1 or ncols > 1:
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        axes = axes.flatten()
        for i in range(nrows * ncols):
            sns.regplot(x=arr_t[:, i], y=arr_p[:, i], ax=axes[i], scatter=scatter_pts,
                        line_kws=dict(color='k'), ci=None,
                        scatter_kws=dict(s=3, alpha=0.5))
            axes[i].set_title(f'{subtitle} {i+1}{i+1}')
            axes[i].axline([arr_t[:, i].mean(), arr_t[:, i].mean()],
                           slope=1, color='r', linestyle=':')

    else:
        fig, axes = plt.subplots()
        sns.regplot(x=arr_t, y=arr_p, ax=axes, scatter=scatter_pts,
                    line_kws=dict(color='k'), ci=None,
                    scatter_kws=dict(s=3, alpha=0.5))

        axes.set_title(f'{subtitle}')
        axes.axline([arr_t.mean(), arr_t.mean()],
                    slope=1, color='r', linestyle=':')

    sns.despine(fig)
    fig.tight_layout()


def extract_symm(arr: np.ndarray) -> np.ndarray:
    assert arr.ndim == 3, 'Expecting 3D NxKxK array only'
    assert arr.shape[1] == arr.shape[2], 'Expecting 3D NxKxK array only'
    ncols = arr.shape[1]
    arr_list = list()
    for i in range(ncols):
        for j in range(i, ncols):
            arr_list.append(arr[:, i, j])
    return np.c_[*arr_list]


def create_invars(F: np.ndarray) -> np.ndarray:
    B = np.einsum('ijk, ikl -> ijl', F, np.transpose(F, axes=(0, 2, 1)))
    B_sq = np.einsum('ijk, ikl -> ijl', B, B)
    J = np.linalg.det(F)
    I1 = np.trace(B, axis1=1, axis2=2)
    I2 = 0.5 * (I1**2 - np.trace(B_sq, axis1=1, axis2=2))
    I3 = J
    invars = np.c_[I1, I2, I3]
    return invars
