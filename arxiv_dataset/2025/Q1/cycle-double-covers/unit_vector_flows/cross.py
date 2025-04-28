#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math

def cross(points, edge1, edge2):
    # Get normal to planes containing great circles
    # np.cross product of vector to each point from the origin
    N1 = np.cross(points[edge1[0]], points[edge1[1]])
    N2 = np.cross(points[edge2[0]], points[edge2[1]])

    # Find line of intersection between two planes
    L = np.cross(N1, N2)

    # Find two intersection points
    X1 = L / np.sqrt(L[0] ** 2 + L[1] ** 2 + L[2] ** 2) # TODO catch division by zero or negative numbers?
    X2 = -X1
    i_lat1 = math.asin(X1[2])
    i_long1 = math.atan2(X1[1], X1[0])
    i_lat2 = math.asin(X2[2])
    i_long2 = math.atan2(X2[1], X2[0])

    p = points[edge1[0]]
    R = (p[0] ** 2 + p[1] ** 2 + p[2] ** 2) ** 0.5
    i_x1 = R * math.cos(i_long1) * math.cos(i_lat1)
    i_y1 = R * math.sin(i_long1) * math.cos(i_lat1)
    i_z1 = R * math.sin(i_lat1)

    i_x2, i_y2, i_z2 = -i_x1, -i_y1, -i_z1

    return [i_x1, i_y1, i_z1], [i_x2, i_y2, i_z2]
