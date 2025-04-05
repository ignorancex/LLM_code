# encoding: utf-8
'''
Point generation 3D
===================
'''

from __future__ import division

import numpy as np

def sphere_arc(bbox, theta, n):
    '''
    Generates a longitude arc on a sphere inscribed in a bounding box.

    :param float bbox: a boundary box, [:math:`x_{min}, y_{min}, x_{max}, y_{max}`].
    :param float tetha: an angle between the projection of the longitude on xy-plane and x-axis.
    :param int n: number of points to generate on the longitude.
    :returns: points
    '''
    r = np.abs(bbox[3] - bbox[0])*1.0/2  
    center = [(bbox[0]+bbox[3])*1.0/2, (bbox[1] + bbox[4])*1.0/2, (bbox[2]+bbox[5])*1.0/2]
    verticle_angle = np.linspace(0, np.pi, n)
    x = r*np.cos(theta)*np.sin(verticle_angle) + center[0]
    y = r*np.sin(theta)*np.sin(verticle_angle) + center[1]
    z = r*np.cos(verticle_angle) + center[2]
    return np.hstack((x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)))

def horizontal_circle(r, n=10, center=[0,0,0]):
    '''
    Generates a horizontal circle parallel to xy-plane.

    :param float r: radius
    :param int n: number of points to generate on the longitude.
    :param float center: coordinates of a center of a circle.
    :returns: points
    '''
    x = center[0] + r*np.cos(np.linspace(0, 2*np.pi, n)).reshape(-1,1)
    y = center[1] + r*np.sin(np.linspace(0, 2*np.pi, n)).reshape(-1,1)
    z = center[2] + np.zeros(n).reshape(-1,1)
    return np.concatenate((x,y,z), axis=1)

def vertical_circle_xz(r, n=5, center=[0,0,0], theta=0):
    '''
    Generates a circle parallel to xz-plane.

    :param float r: radius
    :param int n: number of points to generate on the longitude.
    :param float center: coordinates of a center of a circle.
    :returns: points
    '''
    x = center[0] + r*np.cos(np.linspace(0, 2*np.pi, n)).reshape(-1,1)
    y = center[1] + np.zeros(n).reshape(-1,1)
    z = center[2] + r*np.sin(np.linspace(0, 2*np.pi, n)).reshape(-1,1)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    if theta != 0:
        x1 = x*cos_theta - y*sin_theta
        y1 = y*cos_theta + x*sin_theta
        x = x1
        y = y1
    return np.concatenate((x,y,z), axis=1)

if __name__ == "__main__":
     print sphere_arc([0,0,0,10,10,10], 0, 10)
