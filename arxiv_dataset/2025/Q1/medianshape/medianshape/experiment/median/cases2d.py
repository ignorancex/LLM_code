# encoding: utf-8
'''
Median shape experiment cases in 2D
-----------------------------------

'''
from __future__ import absolute_import

import numpy as np
from medianshape.simplicial import pointgen2d
from medianshape.simplicial.meshgen import meshgen2d
from medianshape.simplicial.mesh import Mesh2D

def triangles2d():
    '''
    A simple experiment in 2D.
    A square with unit sides in the first quarter triangulated
    by 8 triangles.
    There are 2 curves represented by interploting points.
    '''
    mesh = Mesh2D()
    mesh.bbox = [0, 0, 1, 1]
    mesh.set_boundary_points()
    mesh.set_diagonal()
    mesh.set_boundary_values()
    mesh.points = np.array([[0,0], [0,0.5], [0,1], [0.5,1], [1,1], \
                            [1,0.5],[1,0],[0.5,0], [0.5, 0.5]])
    mesh.simplices = np.array([[0,8,1],
                                [1,8,2],
                                [2,8,3],
                                [3,8,4],
                                [4,8,5],
                                [5,8,6],
                                [6,8,7],
                                [7,8,0]])

    mesh.edges = np.array([[0,1],
                            [0,7],
                            [0,8],
                            [1,2],
                            [1,8],
                            [2,3],
                            [2,8],
                            [3,4],
                            [3,8],
                            [4,5],
                            [4,8],
                            [5,6],
                            [5,8],
                            [6,7],
                            [6,8],
                            [7,8]])
    curves = np.zeros((2, 4, 2))
    curves[0] = np.array([[0,0], [0,0.5], [0,1], [1,1]])
    curves[1] = np.array([[0,0], [0.5,0], [1,0], [1,1]])
    is_closed = False
    lambdas = [1]
    mus = [0.01]
    return mesh, mesh.simplices, mesh.edges, curves, lambdas, mus, is_closed

def ellipses2d():
    '''
    Two ellipses in a simplicial complex, K of dimension 2.
    '''
    boundary_box = (0,0,1,1)
    l=0.07
    mesh = meshgen2d(boundary_box, l)
    ellipse1 = pointgen2d.sample_ellipse(0.4, 0.2, 10)
    ellipse2 = pointgen2d.sample_ellipse(0.2, 0.4, 10)
    #ellipse1 = pointgen2d.sample_ellipse(0.4, 0.2, 3)
    #ellipse2 = pointgen2d.sample_ellipse(0.2, 0.4, 3)
    shapes = [ellipse1, ellipse2]
    lambdas = [1]
    #mus = [0.0001] #for cvxopt solver
    mus = [0.01]
    is_closed = True
    return mesh, mesh.simplices, mesh.edges, np.array(shapes), lambdas, mus, is_closed

def twisted_curves2d():
    '''
    Two ellipses in a simplicial complex, K of dimension 2.
    '''
    boundary_box = (0,0,40,40)
    l = 2
    mesh = meshgen2d(boundary_box, l)
    curves= pointgen2d.twisted_curves()
    lambdas = [0.01]
    mus = [0.00001]
    is_closed = False
    return mesh, mesh.simplices, mesh.edges, np.array(curves), lambdas, mus, is_closed

def x_x2_x5_2d():
    '''
    Three curves in 2D given as :math:`x, x^2, x^5`.
    '''
    boundary_box = (0,0,1,1)
    l = 0.04
    mesh = meshgen2d(boundary_box, l)
    functions= ['x', 'x2', 'x5']
    points = list()
    for f in functions:
        points.append(pointgen2d.sample_function(f, boundary_box, 10))
    lambdas = [0.001]
    mus = [0.00001]
    is_closed = False
    return mesh, mesh.simplices, mesh.edges, points, lambdas, mus, is_closed

def sinuses2d():
    '''
    Two curves described by sin1pi and half_sin1pi functions defined in 'medianshape.simplicial.utils'
    '''
    boundary_box = (0,0,1,1)
    l = 0.02
    mesh = meshgen2d(boundary_box, l)
    functions= ['sin1pi', 'half_sin1pi']
    points = list()
    for f in functions:
        points.append(pointgen2d.sample_function(f, boundary_box, 15))
    lambdas = [0.001]
    mus = [0]
    is_closed = False
    return mesh, mesh.simplices, mesh.edges, points, lambdas, mus, is_closed

def multicurves2d():
    '''
    3 curves described by curve1, curve2, curve3 functions defined in 'medianshape.simplicial.utils'.
    '''
    boundary_box = (0,0,200,50)
    l=3
    mesh = meshgen2d(boundary_box, l)
    functions= ['curve1', 'curve2', 'curve3']
    points = list()
    for f in functions:
        points.append(pointgen2d.sample_function(f, boundary_box, 25))
    lambdas = [0.001]
    mus = [0.00001]
    is_closed = False
    return mesh, mesh.simplices, mesh.edges, points, lambdas, mus, is_closed

def two_curves2d():
    '''
    2 curves described by deformcurve1, deformcurve2 functions defined in 'medianshape.simplicial.utils'.
    '''
    boundary_box = (0,0,200,50)
    #l = 3
    l = 30
    mesh = meshgen2d(boundary_box, l)
    functions= ['deformcurve1', 'deformcurve2']
    points = list()
    for f in functions:
        #points.append(pointgen2d.sample_function_mesh(mesh, f))
        points.append(pointgen2d.sample_function_mesh(mesh, f, sample_size=6))
    lambdas = [0.01]
    mus = [0.0001]
    is_closed = False
    return mesh, mesh.simplices, mesh.edges, np.array(points), lambdas, mus, is_closed
