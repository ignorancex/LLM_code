"""
File for plotting functionalities and styling constants
"""

import plotly.graph_objects as go


# different color palettes

colours=["#e41a1c",
"#377eb8",
"#4daf4a",
"#984ea3",
"#ff7f00",
"#ffff33",
"#a65628",
"#f781bf"]

plotly_colors=['#636EFA',
'#FFA15A',
'#00CC96',
'#AB63FA',
     '#EF553B',
     '#19D3F3',
     '#FF6692',
     '#B6E880',
     '#FF97FF',
     '#FECB52',
               '#636EFA',
               '#EF553B',
               '#00CC96',
               '#AB63FA',
               '#FFA15A',
               '#19D3F3',
               '#FF6692',
               '#B6E880',
               '#FF97FF',
               '#FECB52'
               ]

col_rgba = ['rgba(31, 119, 180, 0.2)',
       'rgba(255, 127, 14, 0.2)',
        'rgba(44, 160, 44, 0.2)',
       'rgba(214, 39, 40, 0.2)',
    'rgba(148, 103, 189, 0.2)',
       'rgba(140, 86, 75, 0.2)',
        'rgba(227, 119, 194, 0.2)',
       'rgba(127, 127, 127, 0.2)',
        'rgba(188, 189, 34, 0.2)',
       'rgba(23, 190, 207, 0.2)']


def update_layout(fig, do_markers=True):
    """update layout for the paper

    Parameters
    ----------
    fig : plotly figure

    Returns
    -------
    fig : plotly figure
        input figure with updated layout

    """

    layout = go.Layout(
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="serif", size=32),
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(showline=True, linewidth=2, linecolor="black"),  # gridcolor="grey"),
        yaxis=dict(showline=True, linewidth=2, linecolor="black")  # , gridcolor="grey")
    )
    fig.update_xaxes(showline=True, linewidth=2, linecolor="black")
    fig.update_yaxes(showline=True, linewidth=2, linecolor="black")
    if do_markers:
        fig.update_traces(marker_line_width=2, marker_size=5, line_width=4)
    fig.update_layout(layout)

    return fig



