# --------------------------------------------------------------------------------------------------------------------
# File to store all functions for visualization
# --------------------------------------------------------------------------------------------------------------------

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os
from types import SimpleNamespace
from utils_plot import update_layout, plotly_colors



def visualize_results(name_id, result_path, threshold=0.3):
    """
    Visualize results
    Parameters
    ----------
    name_id: str
    result_path: str
    threshold:float

    Returns
    -------

    """

    # -----------------------------------------------------------------
    # Read in optimization results
    # -----------------------------------------------------------------
    res = np.load(os.path.join(result_path, "results.npy"), allow_pickle='TRUE').item()
    results = SimpleNamespace(**res)
    beta_0 = np.hstack([0.0, results.beta])

    # -----------------------------------------------------------------
    # Analyze results
    # -----------------------------------------------------------------

    plot_name_list = ["SIS", "Random", "IdealEx"]
    idx = list(np.arange(len(beta_0)))
    idx_nonzero = np.array(idx)[beta_0 != 0]

    ## analyze plot optimization
    plot_beta_list = [results.diff_beta, results.diff_singlex_beta, results.diff_benchmark_beta]
    plot_beta_list_0 = [results.diff_beta + beta_0, results.diff_benchmark_beta +
                        beta_0, results.diff_singlex_beta + beta_0]
    plot_beta_cover_list = [results.cover_trajectory, results.benchmark_cover_trajectory, results.singlex_cover]

    plot_optimization(plot_beta_list, plot_name_list,
                      beta_0, idx_nonzero, name_id + "_nonzero", result_path, type_beta="error",
                      cover_list=plot_beta_cover_list)
    plot_optimization(plot_beta_list, plot_name_list,
                      beta_0, idx, name_id + "_all", result_path, type_beta="error",
                      cover_list=plot_beta_cover_list)
    fig1 = plot_optimization(plot_beta_list_0, plot_name_list,
                      beta_0, idx_nonzero, name_id + "_nonzero", result_path, type_beta="absolut",
                      cover_list=plot_beta_cover_list)

    ## analyze plot optimization round
    plot_beta_seq_list = [results.diff_beta_trajectory, results.diff_benchmark_beta_trajectory]
    plot_name_seq_list = ["SIS", "Random"]
    plot_cover_seq_list = [results.cover_trajectory, results.benchmark_cover_trajectory]
    # plot for all components
    fig2 = plot_optimization_round(plot_beta_seq_list, plot_name_seq_list, idx, name_id, result_path,
                            cover_list=plot_cover_seq_list)
    # plot for nonzero components
    fig3 = plot_optimization_round(plot_beta_seq_list, plot_name_seq_list, idx_nonzero, name_id, result_path,
                                   cover_list=plot_cover_seq_list, nonzero=True, threshold_cover=threshold)


    ## analyze plot optimization round with causal regularization norm
    mnorm_hat = results.max_norm_hat
    beta_trajectory = results.diff_beta_trajectory + beta_0
    fig4 = plot_optimization_round_norm(beta_trajectory, mnorm_hat, beta_0, idx, name_id, result_path)

    return fig1, fig2, fig3, fig4



def plot_optimization(beta_list, name_list, beta_0, idx, name_id, fig_path, type_beta="absolute", cover_list=None,
                      boxmean=True):
    """
    plot optimization
    Parameters
    ----------
    beta_list: list of ndarrays
        list of beta to be plotted
    name_list: list of str
        corresponding reference names to the betas
    beta_0: ndarray
        true beta value
    idx: index
        indices to be plotted
    name_id: str
    fig_path: str
    type_beta: str
        default *absolute* (other value *error*) depending on whether the absolute beta values or beta - beta_0 is to be plotted
    cover_list: list of ndarrays
        list of coverage for ndarrays
    boxmean: bool
        whether to additionally plot the mean in the boxplots along with the median

    Returns
    -------

    """
    jitter = 0.5
    n_runs = len(beta_list[0])
    if cover_list is not None:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    else:
        fig = make_subplots(rows=1, cols=1)

    x = [[comp for comp in np.arange(1, len(idx) + 1)] for _ in np.arange(n_runs)]
    x = list(np.concatenate(x))
    x = np.array(x)

    jter = 0
    for beta_error in beta_list:
        res = [ar[idx] for ar in beta_error]
        beta_true_chopped = beta_0[idx]

        fig.add_trace(
            go.Box(y=np.concatenate(res),
                      x=x,
                   boxmean=boxmean,
                      name=name_list[jter],
                      jitter=jitter,
                      marker=dict(color=plotly_colors[jter])),
        row=1, col=1)
        jter += 1

    jter=0
    if cover_list is not None:
        for cover in cover_list:
            res = [ar[-1][idx] for ar in cover]
            fig.add_trace(
                go.Box(y=np.concatenate(res),
                          x=x,
                          name=name_list[jter],
                          showlegend=False,
                          marker=dict(color=plotly_colors[jter]),
                          opacity=0.7),
                row=2, col=1,
                )
            jter += 1

    fig.update_layout(showlegend=True, boxmode='group', violinmode="group")

    chop = 1 / len(idx)
    for iter in np.arange(len(idx)):
        fig.add_shape(type="line",
                      y0=0 if type_beta == "error" else beta_true_chopped[iter],
                      y1=0 if type_beta == "error" else beta_true_chopped[iter],
                      xref="paper",
                      x0=iter / len(idx),
                      x1=iter / len(idx) + chop,
                      line_dash="dot",
                      )

    ## Update Layout here
    fig.update_yaxes(title_text=r"$\Large{\hat{\beta} - \beta}$" if type_beta == "error" else r"$\Large{\hat{\beta}}$",)
    if cover_list is not None:
        fig.update_yaxes(title_text=r"$\Large{\angle{(P_{\alpha[T]}, e_i})}$", row=2, col=1)
    fig = update_layout(fig)
    fig.update_layout(legend=dict(
        yanchor="top",
        orientation="h",
        y=1.0,
        xanchor="left",
        x=0.01
    ))
    fig.update_layout(width=2400, height=500)

    fig.write_image(os.path.join(fig_path, "Beta_" + type_beta + "_" + name_id + ".pdf"), format="pdf")

    return fig



def plot_optimization_round(beta_list, name_list, idx, name_id, fig_path, cover_list=None, threshold_cover=0.3,
                            boxmean=True, nonzero=False):
    """

    Parameters
    ----------
    beta_list: list of ndarrays
        list of beta to be plotted
    name_list: list of str
        corresponding reference names to the betas
    idx: index
        indices to be plotted
    name_id: str
    fig_path: str
    cover_list: list of ndarrays
        list of coverage for ndarrays
    threshold_cover: float
    boxmean: bool
        whether to additionally plot the mean in the boxplots along with the median
    nonzero: bool
        whether the nonzero values are to be plotted
    Returns
    -------

    """
    if cover_list is not None:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    else:
        fig = make_subplots(rows=1, cols=1)
    n_runs = len(beta_list[0])
    x = [["Round " + str(comp + 1) for comp in np.arange(len(beta_list[0][i]))] for i in np.arange(n_runs)]
    x = list(np.concatenate(x))

    jter = 0
    for beta_traj in beta_list:
        beta_traj_concat = np.array([b[idx] for b in np.concatenate(beta_traj)])
        beta_traj_norm = np.sqrt((beta_traj_concat**2).sum(axis=1))
        fig.add_trace(
            go.Box(y=beta_traj_norm, x=x, name=name_list[jter],
                   boxmean=boxmean,
                   marker=dict(color=plotly_colors[jter])),
        row=1, col=1)
        jter =+1

    jter = 0
    if cover_list is not None:
        for cover in cover_list:
            if nonzero:
                cover_traj_concat = np.array([b[idx] for b in np.concatenate(cover)])
                cover_traj_norm = 1.0 - ((cover_traj_concat < threshold_cover).sum(axis=1) / len(idx))

            else:
                cover_traj_concat = np.array([b for b in np.concatenate(cover)])
                cover_traj_norm = 1.0 - ((cover_traj_concat < threshold_cover).sum(axis=1) / len(cover[0][0]))

            fig.add_trace(
                go.Box(y=cover_traj_norm,
                          x=x,
                          name=name_list[jter],
                          showlegend=False,
                          marker=dict(color=plotly_colors[jter]),
                          opacity=0.7),
                row=2, col=1)
            jter += 1

    fig = update_layout(fig)
    fig.update_layout(legend=dict(
        yanchor="top",
        orientation="h",
        y=0.15,
        xanchor="right",
        x=0.25
    ))

    fig.update_yaxes(title_text="MSE")
    if cover_list is not None:
        if nonzero:
            fig.update_yaxes(title_text=r"$\Large{1 - \frac{\text{# Comp.Covered}}{\text{# NonZeroComp.}}}$", row=2,
                         col=1)
        else:
            fig.update_yaxes(title_text=r"$\Large{1 - \frac{\text{# Comp.Covered}}{\text{# AllComp.}}}$", row=2,
                             col=1)

    fig.update_layout(boxmode="group", violinmode="group")
    fig.update_layout(width=1200, height=500)
    if nonzero:
        name_id = name_id + str("_nonzero")
    fig.write_image(os.path.join(fig_path, "BetaSequential_" + name_id + ".pdf"), format="pdf")

    return fig


def plot_optimization_round_norm(beta_traj, mnorm_hat, beta_0, idx, name_id, fig_path, boxmean=True):
    """

    Parameters
    ----------
    beta_traj: list of ndarrays
        list of beta trajectories
    mnorm_hat: ndarray
    beta_0: ndarray
    idx: index
    name_id: str
    fig_path: str
    boxmean: bool

    Returns
    -------

    """

    fig = go.Figure()
    n_runs = len(beta_traj)
    x = [["Round " + str(comp + 1) for comp in np.arange(len(beta_traj[i]))] for i in np.arange(n_runs)]
    x = list(np.concatenate(x))

    beta_traj_concat = np.array([b[idx] for b in np.concatenate(beta_traj)])
    beta_traj_norm = np.sqrt((beta_traj_concat ** 2).sum(axis=1))

    # true beta
    beta_norm_true = np.sqrt(beta_0 @ beta_0)

    fig.add_shape(type="line",
                  y0=beta_norm_true,
                  y1=beta_norm_true,
                  xref="paper",
                  x0=0,
                  x1=len(x),
                  line_dash="dot")


    fig.add_trace(
        go.Box(y=beta_traj_norm,
               x=x,
               boxmean=boxmean,
               name=r"$||\Large{\widehat{P_{\alpha[T]}} \beta}||$",
               marker=dict(color=plotly_colors[0])))

    fig.add_trace(
        go.Scatter(
            x=[x[-1]],
            y=[beta_norm_true - 1.0],
            name=r"$\Large{||\beta||}$",
            line=dict(color='black', dash='dot'),
            showlegend=True
        )
    )

    sqrt_norm_hat = np.sqrt(mnorm_hat)
    fig.add_trace(
        go.Scatter(
            x=[x[-1]],
            y=[beta_norm_true - 1.0],
            name=r"$\Large{\widehat{||\beta||}}$",
            line=dict(color=plotly_colors[1], dash='dot'),
            showlegend=True
        )
    )
    fig.add_shape(type="rect",
                  y0=sqrt_norm_hat.min(),
                  y1=sqrt_norm_hat.max(),
                  xref="paper",
                  x0=0,
                  x1=len(x),
                  line=dict(color=plotly_colors[1], dash='dot'),
                  fillcolor=plotly_colors[1],
                  opacity=0.5)
    fig = update_layout(fig)
    fig.update_yaxes(title_text=r"Norm")
    fig.update_layout(legend=dict(
        y=0.15,
        x=0.75,
    ))
    fig.update_layout(width=1200, height=500)
    fig.write_image(os.path.join(fig_path, "BetaSequentialNorm_" + name_id + ".pdf"), format="pdf")

    return fig


def plot_three_sets(res_beta123, res_beta12_3, res_beta1_2_3, beta_0, name_id, fig_path, showlegend=True, boxmean=True):
    """

    Parameters
    ----------
    res_beta123: list of ndarrays
    res_beta12_3: list of ndarrays
    res_beta1_2_3: list of ndarrays
    beta_0: ndarray
    name_id: str
    fig_path: str
    showlegend: bool
    boxmean: bool

    Returns
    -------

    """

    boxpoints = None
    jitter = 0.4
    n_runs = len(res_beta123)
    p = len(res_beta123[0]) - 1
    fig = go.Figure()

    x = [[comp for comp in np.arange(len(beta_0))] for _ in np.arange(n_runs)]
    x = list(np.concatenate(x))
    x = np.array(x)

    fig.add_trace(
        go.Box(y=np.concatenate(res_beta123), x=x, name="IdealEx", boxpoints=boxpoints, jitter=jitter, boxmean=boxmean,
               marker=dict(color=plotly_colors[0])))
    fig.add_trace(
        go.Box(y=np.concatenate(res_beta12_3), x=x, name="TwoEx", boxpoints=boxpoints, jitter=jitter, boxmean=boxmean,
               marker=dict(color=plotly_colors[1])))
    fig.add_trace(
        go.Box(y=np.concatenate(res_beta1_2_3), x=x, name="ThreeEx", boxpoints=boxpoints, jitter=jitter, boxmean=boxmean,
               marker=dict(color=plotly_colors[2])))
    fig.update_layout(showlegend=True, boxmode='group')

    chop = 1 / (p + 1)
    for iter in np.arange(p + 1):
        fig.add_shape(type="line",
                      y0=beta_0[iter],
                      y1=beta_0[iter],
                      xref="paper",
                      x0=iter / (p + 1),
                      x1=iter / (p + 1) + chop,
                      line_dash="dot",
                      )


    fig = update_layout(fig)
    fig.update_layout(width=1000, height=350, showlegend=True, boxmode='group')
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="top",
        y=1.2,
        xanchor="left",
        x=0.01
    ))
    fig.update_layout(showlegend=showlegend)
    fig.write_image(os.path.join(fig_path, "FiniteSample" + name_id + ".pdf"),
                    format="pdf")
    return fig




