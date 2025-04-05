# encoding: utf-8
'''
Utils
=====
'''

from __future__ import division

import sys
import importlib

import numpy as np
import math

def get_bbox_diagonal(mesh_points=None, bbox=None):
    '''
    Returns the diagonal of a bounding box in 2D or 3D.
    '''
    if mesh_points.shape[1]==2:
        if bbox is None and mesh_points is not None:
            bbox = list()
            bbox.append(np.amin(mesh_points[:,0]))
            bbox.append(np.amin(mesh_points[:,1]))
            bbox.append(np.amax(mesh_points[:,0]))
            bbox.append(np.amax(mesh_points[:,1]))
        if bbox is not None: 
            return np.sqrt((bbox[2]-bbox[0])**2 + (bbox[3]-bbox[1])**2)
        else:
            return None
    elif mesh_points.shape[1]==3:
        if bbox is None and mesh_points is not None:
            bbox = list()
            bbox.append(np.amin(mesh_points[:,0]))
            bbox.append(np.amin(mesh_points[:,1]))
            bbox.append(np.amin(mesh_points[:,2]))
            bbox.append(np.amax(mesh_points[:,0]))
            bbox.append(np.amax(mesh_points[:,1]))
            bbox.append(np.amax(mesh_points[:,2]))
        if bbox is not None: 
            return np.sqrt((bbox[3]-bbox[0])**2 + (bbox[4]-bbox[1])**2 + (bbox[5]-bbox[2])**2)
        else:
            return None

def boundary_points(bbox):
    '''
    Returns corner points of a boundary box in which
    4 corners of the bottom rectangle are the first 4 points,
    oriented CCW starting from (xmin, ymin, zmin).
    Similarly, 4 corners of the top rectangle are the last 4 points,
    oriented CCW starting from (xmin, ymin, zmax).
    '''
    if bbox is not None and len(bbox) == 6:
        return np.array([[bbox[0], bbox[1], bbox[2]],\
                    [bbox[3], bbox[1], bbox[2]],\
                    [bbox[3], bbox[4], bbox[2]],\
                    [bbox[0], bbox[3], bbox[2]],\
                    [bbox[0], bbox[1], bbox[5]],\
                    [bbox[3], bbox[1], bbox[5]],\
                    [bbox[3], bbox[4], bbox[5]],\
                    [bbox[0], bbox[4], bbox[5]]])

def mid_corners(bbox):
    if bbox is not None and len(bbox) == 6:
        xmin = bbox[0]
        xmax  = bbox[3]
        ymin = bbox[1]
        ymax = bbox[4]
        zmin = bbox[2]
        zmax = bbox[5]
        zmid = (zmax+ zmin)/2.0
        return np.array([[xmin, ymin, zmid],\
                    [xmax, ymin, zmid],\
                    [xmax, ymax, zmid],\
                    [xmin, ymax, zmid]])

def get_combination(n, vals=[1,-1]):
    '''
    Returns all combinations of given values for n positions.
    '''
    if n > 1:
        comb = np.array([])
        successors = get_combination(n-1, vals)
        for val in vals:
            for s in successors:
                row = np.append(val, s)
                if comb.size == 0:
                    comb = row
                else:
                    comb = np.vstack((comb,row))
        return comb
    elif n==1:
        return vals

def vectorize(func_str, X):
    '''
    Computes function values on X domain.

    :param str func_str: a name of a function in math module in python or \ a name of a function within this module (medianshape.simplicial.utils).
    :returns: func_values -- function values on the given domain.
    '''
    func_points = []
    if func_str.find(".") != -1:
        mod_name, func_name = func_str.rsplit('.', 1)
        mod = importlib.import_module(mod_name)
        func = getattr(mod, func_name, "")
        if func:
            vec_func = np.vectorize(func)    
    else:
        this_module = sys.modules[__name__]
        func = getattr(this_module, func_str, "")
        vec_func = np.vectorize(func)    
    func_values = vec_func(X)
    return func_values

def x(x):
    '''
    :math:`f(x) = x`
    '''
    return x
def x2(x):
    '''
    :math:`f(x) = x^2`
    '''
    return x**2
def x5(x):
    '''
    :math:`f(x) = x^5`
    '''
    return x**5
def sin2pi(x):
    '''
    :math:`f(x) = |\sin(2\pi x)|`
    '''
    return np.abs(math.sin(2*math.pi*x))
def sin1pi(x):
    '''
    :math:`f(x) = |\sin(\pi x)|`
    '''
    return np.abs(math.sin(math.pi*x))
def half_sin1pi(x):
    '''
    :math:`f(x) = \\frac{1}{2}|\sin(\pi x)|`
    '''
    return 1.0/2*np.abs(math.sin(math.pi*x)) 
def small_sin1pi(x):
    '''
    :math:`f(x) = \\frac{1}{4}|\sin(\pi x)|`
    '''
    return np.abs(math.sin(math.pi*x))/4

def func1(x):
    '''
    :math:`f(x) = \\frac{2}{\pi \cos^{-1}(x)}`
    '''
    return 2/math.pi*math.acos(x)

def func2(x):
    '''
    :math:`f(x) = \\frac{1}{2}(1 + \sin2\pi x)`
    '''
    return 1/2*(1+math.sin(2*math.pi*x))

def curve1(x):
    '''
    :math:`f(x) = 10e^{\\frac{x}{180}}\sin(\\frac{1}{100}\pi x) +25`
    '''
    return 10*math.sin(1/100* math.pi * x) * np.exp(1/180 * x)  + 25
def curve2(x):
    '''
    :math:`f(x) = -10e^{\\frac{x}{200}}\sin(\\frac{1}{50}\pi x) +25`
    '''
    return -10*math.sin(1/50* math.pi * x) * np.exp(1/200 * x) + 25
def curve3(x):
    '''
    :math:`f(x) = 30e^{\\frac{-x}{150}}\sin(\\frac{1}{50}\pi x) +20`
    '''
    return 30*math.sin(1/50* math.pi * x) * np.exp(-1/150 * x) + 20
def curve4(x):
    '''
    :math:`f(x) = 10e^{\\frac{-x}{200}}\sin(\\frac{1}{30}\pi x) +40`
    '''
    return 10*math.sin(1/30* math.pi * x) * np.exp(-1/200 * x) + 40
def curve5(x):
    '''
    :math:`f(x) = 20\sin(\\frac{1}{200}\pi x) +25`
    '''
    return 20*math.sin(1/200* math.pi * x) + 25
def curve6(x):
    '''
    :math:`f(x) = 10e^{\\frac{-1}{200}x}\sin(\\frac{1}{55}\pi x) +10`
    '''
    return 10*math.sin(1/55* math.pi * x) * np.exp(-1/200 * x) + 10

def deformcurve1(x):
    '''
    :math:`f(x) = 20\sin(\\frac{1}{190}\pi x) +25`
    '''
    return 20*math.sin(1/190* math.pi * x) + 25

def deformcurve2(x):
    '''
    :math:`f(x) = -20\sin(\\frac{1}{190}\pi x) +25`
    '''
    return -20*math.sin(1/190* math.pi * x) + 25
    
if __name__ == "__main__":
    Xs = np.linspace(0, 200, 20)
    Ys = list()
    Ys2 = list()
    Ys3 = list()
    Ys4 = list()
    Ys5 = list()
    Ys6 = list()
    for x in Xs:
        Ys.append(curve1(x))
        Ys2.append(curve2(x))
        Ys3.append(curve3(x))
        Ys4.append(curve4(x))
        Ys5.append(curve5(x))
        Ys6.append(curve6(x))
    #plt.ylim(0, 60)
    #plt.xlim(0, 210)
    plt.figure()
    #plt.plot(Xs, Ys, '-', label="1")
    #plt.plot(Xs, Ys2, '-', label="2")
    #plt.plot(Xs, Ys3, '-', label="3")
    plt.plot(Xs, Ys4, '-', color='r', label="4")
    plt.plot(Xs, Ys5, '-', color='b', label="5")
    #plt.plot(Xs, Ys6, '-', label="6")
    plt.show()
