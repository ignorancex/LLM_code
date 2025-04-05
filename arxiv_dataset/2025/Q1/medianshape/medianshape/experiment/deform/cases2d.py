# encoding: utf-8
'''
Deformation experiment cases in 2D
++++++++++++++++++++++++++++++++++
'''
from __future__ import absolute_import

import numpy as np
from medianshape.simplicial import pointgen2d
from medianshape.simplicial.meshgen import meshgen2d
from medianshape.simplicial.mesh import Mesh2D

def curvedeform2d():
    '''
    Two curves with shared boundaries.
    '''
    boundary_box = (0,0,200,50)
    l = 3
    mesh = meshgen2d(boundary_box, l)
    functions= ['deformcurve1', 'deformcurve2']
    points = list()
    for f in functions:
        points.append(pointgen2d.sample_function_mesh(mesh, f))
    alpha1 = np.array([0])
    alpha1 = np.append(alpha1, np.linspace(0.4999, 0.5, 10))
    alpha1 = np.append(alpha1[:-1], np.linspace(0.5, 0.5001, 10))
    alpha1 = np.append(alpha1, np.array([1]))
    alpha1 = alpha1.reshape(alpha1.size, 1) 
    alpha2 = (1-alpha1).reshape(alpha1.size, 1)
    alphas = np.hstack((alpha1, alpha2))
    lambdas = [0.01]
    mus = [0.0001]
    is_closed = False
    return mesh, mesh.simplices, mesh.edges, np.array(points), lambdas, mus, alphas, is_closed

def ellipsesdeform2d():
    '''
    Two overlapping ellipses.
    '''
    boundary_box = (0,0,1,1)
    l=0.07
    mesh = meshgen2d(boundary_box, l)
    ellipse1 = pointgen2d.sample_ellipse(0.4, 0.2, 10)
    ellipse2 = pointgen2d.sample_ellipse(0.2, 0.4, 10)
    shapes = [ellipse1, ellipse2]
    lambdas = [1]
    mus = [0.01]
    alpha1 = np.array([0])
    alpha1 = np.append(alpha1, np.linspace(0.4, 0.5, 10))
    alpha1 = np.append(alpha1[:-1], np.linspace(0.5, 0.6, 10))
    alpha1 = np.append(alpha1, np.array([1]))
    alpha1 = alpha1.reshape(alpha1.size, 1) 
    alpha2 = (1-alpha1).reshape(alpha1.size, 1)
    alphas = np.hstack((alpha1, alpha2))
    is_closed = True
    return mesh, mesh.simplices, mesh.edges, np.array(shapes), lambdas, mus, alphas, is_closed
