import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import figure, ticker


def plot_cost(
    results: list[tuple[str, pd.DataFrame]],
    fig: figure.Figure | None = None,
    transition: float | None = None,
    x_tick_spacing: int = 200,
):
    fig, ax = plot_transition(
        results=results,
        col_name="phi",
        fig=fig,
        transition=transition,
        x_tick_spacing=x_tick_spacing,
    )
    fig.suptitle(r"cost $\phi$")
    ax.legend(loc="lower right")
    return fig, ax


def plot_y_violation(
    results: list[tuple[str, pd.DataFrame]],
    fig: figure.Figure | None = None,
    transition: float | None = None,
    max_violation: float | None = None,
    x_tick_spacing: int = 200,
):
    fig, ax = plot_transition(
        results=results,
        col_name="y_violation",
        fig=fig,
        transition=transition,
        x_tick_spacing=x_tick_spacing,
    )
    fig.suptitle(r"y-constraint violation $\|\cdot\|_2$")
    if max_violation is not None:
        ax.set_ylim(bottom=-max_violation / 20, top=max_violation)
    ax.legend(loc="upper right")
    return fig, ax


def plot_dist_to_optimal(
    results: list[tuple[str, pd.DataFrame]],
    fig: figure.Figure | None = None,
    transition: float | None = None,
    x_tick_spacing: int = 200,
):
    fig, ax = plot_transition(
        results=results,
        col_name="d",
        fig=fig,
        transition=transition,
        x_tick_spacing=x_tick_spacing,
        log_y=True,
    )
    fig.suptitle(r"suboptimality $\|u-u^*\|_2$")
    ax.legend(loc="upper right")
    return fig, ax


def plot_cost_and_violation(
    results: list[tuple[str, pd.DataFrame]],
    transition: float | None = None,
    max_violation: float | None = None,
    x_tick_spacing: int = 200,
):
    fig = plt.figure()
    (fig1, fig2) = fig.subfigures(nrows=2, ncols=1)
    # 1
    _, ax = plot_cost(
        results=results, fig=fig1, transition=transition, x_tick_spacing=x_tick_spacing
    )
    ax.get_legend().remove()
    # 2
    plot_y_violation(
        results=results,
        fig=fig2,
        transition=transition,
        max_violation=max_violation,
        x_tick_spacing=x_tick_spacing,
    )

    fig1.subplots_adjust(left=10 / 100, wspace=5 / 100, right=95 / 100, top=0.85, bottom=0.15)
    fig2.subplots_adjust(left=10 / 100, wspace=5 / 100, right=95 / 100, top=0.85, bottom=0.15)


def plot_transition(
    results: list[tuple[str, pd.DataFrame]],
    col_name: str,
    fig: figure.Figure | None = None,
    transition: float | None = None,
    x_tick_spacing: int = 200,
    log_y: bool = False,
):
    # ensure figure exists
    if fig is None:
        fig = plt.figure(constrained_layout=True)

    # aggreagte df
    df = pd.DataFrame()
    for label, result_df in results:
        df[label] = result_df[col_name]

    # subplots
    if transition is None:
        ax_right = fig.subplots(nrows=1, ncols=1)
        if log_y:
            ax_right.semilogy(range(df.shape[0]), df, label=df.columns)
        else:
            ax_right.plot(range(df.shape[0]), df, label=df.columns)
    else:
        (ax_left, ax_right) = fig.subplots(
            nrows=1, ncols=2, gridspec_kw={"width_ratios": [1, 2]}, sharey="row"
        )
        if log_y:
            ax_left.semilogy(range(transition + 1), df[: transition + 1], label=df.columns)
            ax_right.semilogy(range(transition, df.shape[0]), df[transition:], label=df.columns)
        else:
            ax_left.plot(range(transition + 1), df[: transition + 1], label=df.columns)
            ax_right.plot(range(transition, df.shape[0]), df[transition:], label=df.columns)

    # y-ticks
    if not log_y:
        max_y = np.max(np.abs(ax_right.get_ylim()))
        if max_y < 1:
            formatter = ticker.FuncFormatter(lambda y, _: f"{y:.2f}")
        elif max_y < 10:
            formatter = ticker.FuncFormatter(lambda y, _: f"{y:.1f}")
        else:
            formatter = ticker.FuncFormatter(lambda y, _: f"{y:.0f}")
        ax_right.yaxis.set_major_formatter(formatter)
        ax_right.yaxis.set_major_locator(ticker.MaxNLocator(4))

    # x-ticks
    _, upper = ax_right.get_xlim()
    right_ticks = np.arange(0, upper, x_tick_spacing)
    if transition is None:
        ax_right.set_xticks(right_ticks)
    else:
        right_ticks[0] = transition
        ax_left.set_xticks(np.arange(stop=transition + 5, step=5))
        ax_right.set_xticks(right_ticks)

    return fig, ax_right


def plot_carthesian(results: list[tuple[str, pd.DataFrame]], x_col, y_col):
    plt.figure(figsize=(8, 8))
    for label, df in results:
        if x_col not in df.columns:
            continue
        if y_col not in df.columns:
            continue
        plt.plot(df[x_col], df[y_col], marker=".", linewidth=0.9, label=label)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid()
    plt.legend()
