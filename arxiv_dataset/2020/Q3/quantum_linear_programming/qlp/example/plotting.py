"""Plotting setup for example problem
"""
from typing import List
import numpy as np

import plotly.graph_objects as go

from sympy import solve
from sympy.core import relational
from sympy.utilities.lambdify import lambdify


from qlp.example import f, DEPENDENTS

LAYOUT = go.Layout(
    paper_bgcolor="white",
    plot_bgcolor="white",
    height=600,
    width=600,
    showlegend=False,
    xaxis=dict(
        gridwidth=1,
        gridcolor="LightGray",
        range=[0.0, 4.1],
        tickmode="linear",
        tick0=-1.0,
        dtick=1.0,
        showline=True,
        linecolor="LightGray",
        title="$x_0$",
    ),
    yaxis=dict(
        gridwidth=1,
        gridcolor="LightGray",
        range=[0.0, 4.1],
        tickmode="linear",
        tick0=-1.0,
        dtick=1.0,
        showline=True,
        linecolor="LightGray",
        title="$x_1$",
    ),
)

_RANGE = np.linspace(0, 4, 100)
_XX, _YY = np.meshgrid(*[_RANGE] * 2)
_ZZ = f(_XX, _YY)


def _get_f_data(zmin: float = 6.0) -> go.Heatmap:
    """Plots heatmap for f
    """
    zz = np.where(_ZZ < zmin, np.nan, _ZZ)
    hist = go.Heatmap(
        x=_RANGE,
        y=_RANGE,
        z=zz,
        colorscale="Blues",
        zmin=zmin,
        zsmooth="best",
        opacity=0.9,
    )
    return hist


F_DATA = _get_f_data()


def get_constrained_data(inequalities: List[relational.Relational]):
    """Plots shaded area for excluded regions.
    """
    x0, x1 = DEPENDENTS

    fill_kwargs = dict(
        mode="lines", fillcolor="rgba(255, 0, 0, 0.2)", line={"width": 0},
    )

    data = []
    for ineq in inequalities:
        y = solve(ineq.lhs, x1)[0]
        func = lambdify(x0, y, "numpy")
        if isinstance(ineq, relational.GreaterThan):
            data.append(go.Scatter(x=_RANGE, y=func(_RANGE), **fill_kwargs))
            data.append(
                go.Scatter(
                    x=_RANGE, y=[4] * len(_RANGE), fill="tonexty", **fill_kwargs
                ),
            )
        elif isinstance(ineq, relational.LessThan):
            data.append(
                go.Scatter(x=_RANGE, y=func(_RANGE), fill="tozeroy", **fill_kwargs),
            )
        else:
            raise TypeError(str(ineq))

    return data
