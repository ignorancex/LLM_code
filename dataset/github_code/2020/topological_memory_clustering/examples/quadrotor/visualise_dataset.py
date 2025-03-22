from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import os
import sys

ENVIRONMENT = 'cross'
# ENVIRONMENT = 'sphere'

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Expects dataset to be given as first argument.")
        exit(1)

    if not os.path.exists(sys.argv[1]):
        print("File '{0}' does not exist!".format(sys.argv[1]))
        exit(1)
    data = np.load(sys.argv[1])
    print(data.keys())
    samples_X = data['samples_X'][:,:3,:]
    samples_U = data['samples_U']
    samples_final_cost = data['samples_final_cost']

    sample_dim = samples_X.shape[0]
    state_dim = samples_X.shape[1]
    time_dim = samples_X.shape[2]
    print(sample_dim, state_dim, time_dim)

    fig = plt.figure()
    fig.set_size_inches(12,12)
    ax = plt.axes(projection='3d')
    X_to_plot = np.asarray(samples_X)[:,:3,:]
    print(X_to_plot.shape)
    for i in range(X_to_plot.shape[0]):
        ax.plot3D(X_to_plot[i,0,::2],X_to_plot[i,1,::2],X_to_plot[i,2,::2], alpha=1)

    obstacle_color = 'black'
    if ENVIRONMENT == 'cross':
        # Bars
        tmp_values = np.arange(-5,5.25,.25)
        tmp_zeros = np.zeros_like(tmp_values)
        ax.plot3D(tmp_zeros, tmp_values, tmp_zeros, color=obstacle_color, linewidth=5, alpha=0.5)
        ax.plot3D(tmp_values, tmp_zeros, tmp_zeros, color=obstacle_color, linewidth=5, alpha=0.5)
        ax.plot3D(tmp_zeros, tmp_zeros, tmp_values, color=obstacle_color, linewidth=5, alpha=0.5)
    elif ENVIRONMENT == 'sphere':
        # draw sphere
        radius = 1.0
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = 2. * radius * np.cos(u)*np.sin(v)
        y = 2. * radius * np.sin(u)*np.sin(v)
        z = 2. * radius * np.cos(v)
        ax.plot_wireframe(x, y, z, color=obstacle_color) # throws np.asarray deprecation warning

    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)
    ax.set_zlim(-5,5)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.grid(False)

    plt.show()
