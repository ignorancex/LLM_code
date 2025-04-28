#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# FIXME: unocomment, when new matplotlib (3.3.0) is released
# ax.set_box_aspect((1, 1, 1))
try:
  ax.set_aspect('equal')
except NotImplementedError:
  pass

radius = -1
points = []
xx, yy, zz = [], [], []
colors = []
trinities = set()
f = open(sys.argv[1])
for line in f:
    values = line.strip().split('\t')
    if values[0] == 'to_plot':
        points.append([float(v) for v in values[1:]])
        xx.append(float(values[1]))
        yy.append(float(values[2]))
        zz.append(float(values[3]))
    elif values[0] == 'point':
        points.append([float(v) for v in values[1:]])
        xx.append(float(values[2]))
        yy.append(float(values[3]))
        zz.append(float(values[4]))
    elif values[0] == 'radius':
        radius = float(values[1])
    elif values[0] == 'color':
        colors.append(int(values[1]))
    elif values[0] == 'trinity':
        trinities.add(tuple([int(values[1]), int(values[2]), int(values[3])]))
    elif values[0] == 'opposite' or values[0] == 'trinity' or values[0] == 'flow_values':
        pass
    else:
        print("WTF: problems with parsing input:", values[0], line.strip(), file=sys.stderr)

u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)

radius *= 0.9
x = radius * np.outer(np.cos(u), np.sin(v))
y = radius * np.outer(np.sin(u), np.sin(v))
z = radius * np.outer(np.ones(np.size(u)), np.cos(v))

elev = 10.0
rot = 80.0 / 180 * np.pi
ax.plot_surface(x, y, z, rstride=4, cstride=4, color='w', linewidth=0, alpha=0.5)

'''
a = np.array([-np.sin(elev / 180 * np.pi), 0, np.cos(elev / 180 * np.pi)])
b = np.array([0, 1, 0])
b = b * np.cos(rot) + np.cross(a, b) * np.sin(rot) + a * np.dot(a, b) * (1 - np.cos(rot))
ax.plot(np.sin(u), np.cos(u), 0, color='k', linestyle = 'dashed')
horiz_front = np.linspace(0, np.pi, 100)
ax.plot(np.sin(horiz_front), np.cos(horiz_front), 0, color='k')
vert_front = np.linspace(np.pi / 2, 3 * np.pi / 2, 100)
ax.plot(a[0] * np.sin(u) + b[0] * np.cos(u), b[1] * np.cos(u), a[2] * np.sin(u) + b[2] * np.cos(u), color='k', linestyle = 'dashed')
ax.plot(a[0] * np.sin(vert_front) + b[0] * np.cos(vert_front), b[1] * np.cos(vert_front), a[2] * np.sin(vert_front) + b[2] * np.cos(vert_front), color='k')
'''

if not colors:
    ax.scatter(xx[:1], yy[:1], zz[:1], color="r", s=45)
    ax.scatter(xx[1:], yy[1:], zz[1:], color="k", s=18)
else:
    color_palette = ['k', 'r', 'g', 'c']
    color_full_name = ['black', 'red', 'green', 'cyan']
    positive_idx = [[], [], [], []]
    negative_idx = [[], [], [], []]
    for i in range(len(colors)):
        if colors[i] > 0:
            positive_idx[colors[i] - 1].append(i)
        else:
            negative_idx[-colors[i] - 1].append(i)
    for i in range(len(color_palette)):
        ax.scatter([xx[ix] for ix in positive_idx[i]], [yy[iy] for iy in positive_idx[i]], [zz[iz] for iz in positive_idx[i]], color=color_palette[i], s=20)
        ax.scatter([xx[ix] for ix in negative_idx[i]], [yy[iy] for iy in negative_idx[i]], [zz[iz] for iz in negative_idx[i]], color=color_palette[i], s=20, marker='>')
    if trinities:
        t = trinities.pop()
        txx = [xx[t[0]], xx[t[1]], xx[t[2]]]
        tyy = [yy[t[0]], yy[t[1]], yy[t[2]]]
        tzz = [zz[t[0]], zz[t[1]], zz[t[2]]]

        ax.scatter(txx, tyy, tzz, color="y", s=90, marker='x')

ax.view_init(elev = elev, azim = 0)
plt.show()

