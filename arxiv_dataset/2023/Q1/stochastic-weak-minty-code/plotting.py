import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})
plt.rcParams['figure.facecolor'] = 'white'

import itertools
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

PROJECT_PATH = "./output"

def get_history(key, name, x_axis="epoch", base_filter={}, take_every=1):
    filters = {}
    filters.update(base_filter)
    filters.update({"displayName": name})
    runs = glob.glob(os.path.join(PROJECT_PATH, name + '(*'))

    dfs = []
    for run in runs:
        npy_path = os.path.join(run, key + '.npy')
        df = pd.DataFrame({key: np.load(npy_path)}).reset_index()
        dfs.append(df)
    histories = pd.concat(dfs)
    histories = histories[histories['index']%take_every == 0]
    histories = histories.groupby('index').aggregate(['mean', 'std'])
    return histories



def create_plot(
    experiments={},
    base_filter={},
    key=None,
    x_axis="epoch",
    take_every=1,
    transform=lambda x: x,
    plot_kwargs={},
):
    histories = {acronym: get_history(key, name, base_filter=base_filter, x_axis=x_axis, take_every=take_every) for acronym, name in experiments.items()}
    
    markers = itertools.cycle(('*', '+', '.', 's', '^', ',', 'o')) # see https://matplotlib.org/stable/api/markers_api.html ('^', 's')
    colors = itertools.cycle((["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"])) # from https://github.com/matplotlib/matplotlib/issues/9460
    fig, ax = plt.subplots(1, 1)
    for label, data in histories.items():
        ax.plot(transform(data[(key, 'mean')]), 
            marker=next(markers),
            color=next(colors),
            label=label,
            **plot_kwargs)
    ax.legend(loc='lower left')
    return fig, ax
