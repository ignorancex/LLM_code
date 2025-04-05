# encoding: utf-8
'''
Deformation experiment cases in 3D
----------------------------------
'''
from __future__ import absolute_import
import importlib
import os

import numpy as np

from medianshape.simplicial import pointgen3d
from medianshape.simplicial.mesh import Mesh3D
from medianshape.simplicial.meshgen import meshgen3d, distmesh3d, get_mesh_surface
from medianshape.utils import get_subsimplices
from medianshape import inout

def longitudes3ds():
    '''
    Two longitude curves on a solid sphere.
    '''
    boundary_box = (0,0,200,50)
    fixed_points = [(0,0),(200,0),(0,50),(200,50)]
    l=6
    boundary_box = (0,0,40,40)
    fixed_points = [(0,0),(40,0),(0,40),(40,40)]
    boundary_box = (0,0,1,1)
    fixed_points = [(0,0),(1,0),(0,1),(1,1)]
    l=0.07
    boundary_box = [0,0,0,20,20,20]
    l=2
    mesh = Mesh3D()
    # l - initial length of triangle sides. Change it to vary traingle size
    mesh.bbox = boundary_box
    mesh.set_boundary_points()
    mesh.set_diagonal()
    mesh.set_boundary_values()
    mesh.points, mesh.simplices= distmesh3d(mesh.bbox, l, mesh.fixed_points, shape="sphere")
    #mesh.points, mesh.simplices = scipy_mesh3d(mesh.bbox, mesh.fixed_points, l)
    #mesh.points, mesh.simplices= meshpy_cuboid3d(mesh.bbox, mesh.fixed_points.tolist(), max_volume=(1**3)*1.0/(6*np.sqrt(2)))
    #plot3d.plotmesh3d(mesh)
    mesh.triangles = get_subsimplices(mesh.simplices)
    mesh.edges = get_subsimplices(mesh.triangles)
    lambdas = [0.001]
    mus = [0.00001]
    #lambdas = [0.0000001]
    #mus = [0.000000001]
    is_closed = False
    alpha2 = np.array([0])
    alpha2 = np.append(alpha2, np.linspace(0.4999, 0.5, 3))
    alpha2 = np.append(alpha2, np.linspace(0.500001, 0.50001, 50))
    alpha2 = np.append(alpha2, np.array([1]))
    alpha2 = alpha2.reshape(alpha2.size, 1) 
    alpha1 = (1-alpha2).reshape(alpha2.size, 1)
    alphas = np.hstack((alpha1, alpha2))
    
    curve1 = pointgen3d.sphere_arc(mesh.bbox, np.pi + np.pi/6, 10)
    curve2 = pointgen3d.sphere_arc(mesh.bbox, np.pi/18, 10)
    shapes = [curve1, curve2]
    input_points  = np.array(shapes)
    return mesh, mesh.triangles, mesh.edges, input_points, lambdas, mus, alphas, is_closed
    
def longitudes3ds_finer(): 
    '''
    Two longitude curves on the surface of a sphere.
    '''

    '''
    # l - initial length of triangle sides. Change it to vary traingle size
    boundary_box = [0,0,0,20,20,20]
    l = 0.6
    mesh = meshgen3d(boundary_box, l, include_corners=False)
    smesh = get_mesh_surface(mesh)
    inout.save_data(mesh, dirname=os.path.abspath("data/mesh"))
    inout.save_data(smesh, dirname=os.path.abspath("data/smesh"))
    print "Saved"
    exit()
    '''
    smesh = inout.load_mesh3d(dirname='data/smesh')

    curve1 = pointgen3d.sphere_arc(smesh.bbox, np.pi + np.pi/6, 10)
    curve2 = pointgen3d.sphere_arc(smesh.bbox, np.pi/18, 10)

    shapes = [curve1, curve2]
    input_points  = np.array(shapes)
    lambdas = [0.001]
    mus = [0.00001]
    #lambdas = [0.0000001]
    #mus = [0.000000001]
    is_closed = False
    alpha2 = np.array([0])
    alpha2 = np.append(alpha2, np.linspace(0.4999, 0.5, 3))
    alpha2 = np.append(alpha2, np.linspace(0.500001, 0.50001, 50))
    alpha2 = np.append(alpha2, np.array([1]))
    alpha2 = alpha2.reshape(alpha2.size, 1) 
    alpha1 = (1-alpha2).reshape(alpha2.size, 1)
    alphas = np.hstack((alpha1, alpha2))
    return smesh, smesh.triangles, smesh.edges, input_points, lambdas, mus, alphas, is_closed
