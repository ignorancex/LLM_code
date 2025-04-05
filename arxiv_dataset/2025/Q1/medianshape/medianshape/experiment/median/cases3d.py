# encoding: utf-8
'''
Median shape experiment cases in 3D
-----------------------------------
'''
from __future__ import absolute_import
import importlib
import os

import numpy as np

from medianshape.simplicial import pointgen3d
from medianshape.simplicial.meshgen import meshgen3d, get_mesh_surface
from medianshape import inout

import matplotlib.pyplot as plt
import medianshape.viz.plot3d as plot3d

def torus3d():
    '''
    Horizontal and vertical cycles on a solid torus.
    '''
    # l - initial length of triangle sides. Change it to vary traingle size
    bbox = [-10,-10,-10,10,10,10]
    l = 2
    #radius of a sphere inscribed in the boundary box
    radius = np.abs(bbox[3] - bbox[0])*1.0/2  
    r = radius/4
    R = radius - r
    mesh = meshgen3d(bbox, l, include_corners=False, shape="torus", R=R, r=r)
    '''
    bbox = [-1,-1,-0.2,1,1,0.2]
    l = 0.06
    radius = np.abs(bbox[3] - bbox[0])*1.0/2  
    r = 0.2
    R = 0.8
    mesh  = inout.load_mesh3d(dirname='data/torus')
    '''
    # generating horizontal cycle
    n = 10
    x = radius*np.cos(np.linspace(0, 2*np.pi, n)).reshape(-1,1)
    y = radius*np.sin(np.linspace(0, 2*np.pi, n)).reshape(-1,1)
    z = np.zeros(n).reshape(-1,1)
    curve1 = np.concatenate((x,y,z), axis=1)
    # generating vertical cycle
    n = 5
    x = r*np.cos(np.linspace(0, 2*np.pi, n)).reshape(-1,1) - R
    y = np.zeros(n).reshape(-1,1)
    z = r*np.sin(np.linspace(0, 2*np.pi, n)).reshape(-1,1)
    curve2 = np.concatenate((x,y,z), axis=1)
    shapes = [curve1, curve2]
    input_points  = np.array(shapes)
    #lambdas = [0.001]
    #mus = [0.00001]
    lambdas = [1, 0.0001]
    mus = [0.00001]
    is_closed = True
    inout.save_data(mesh=mesh, dirname="data/dumps/torus")
    return mesh, mesh.triangles, mesh.edges, input_points, lambdas, mus, is_closed

def torus_surface3d():
    '''
    Horizontal and vertical cycles on the surface on a torus.
    '''
    # l - initial length of triangle sides. Change it to vary traingle size
    bbox = [-10,-10,-10,10,10,10]
    bbox = [-1,-1,-0.2,1,1,0.2]
    l = 0.06
    radius = np.abs(bbox[3] - bbox[0])*1.0/2  
    r = 0.2
    R = 0.8
    '''
    mesh = meshgen3d(bbox, l, include_corners=False, shape="torus")
    print "mesh Saved"
    inout.save_data(mesh, dirname=os.path.abspath("data/torus"))
    smesh = get_mesh_surface(mesh)
    inout.save_data(smesh, dirname=os.path.abspath("data/storus"))
    print "Saved"
    '''
    #load mesh from data/storus which is a surface of a torus.
    smesh = inout.load_mesh3d(dirname='data/storus')
    #plot3d.plot_simplices3d(smesh, np.ones(len(smesh.triangles)))
    #plt.show()

    # generating horizontal cycle
    n = 10
    x = radius*np.cos(np.linspace(0, 2*np.pi, n)).reshape(-1,1)
    y = radius*np.sin(np.linspace(0, 2*np.pi, n)).reshape(-1,1)
    z = np.zeros(n).reshape(-1,1)
    curve1 = np.concatenate((x,y,z), axis=1)
    # generating vertical cycle
    n = 5
    x = r*np.cos(np.linspace(0, 2*np.pi, n)).reshape(-1,1) - R
    y = np.zeros(n).reshape(-1,1)
    z = r*np.sin(np.linspace(0, 2*np.pi, n)).reshape(-1,1)
    curve2 = np.concatenate((x,y,z), axis=1)
    shapes = [curve1, curve2]
    input_points  = np.array(shapes)
    #lambdas = [0.001]
    #mus = [0.00001]
    lambdas = [0.001, 0.0001]
    mus = [0.00001, 0.000001]
    is_closed = True
    return smesh, smesh.triangles, smesh.edges, input_points, lambdas, mus, is_closed

def handle_loops_on_torus_surface3d():
    '''
    Two handle loops on a surface of a torus which is hollow.
    '''
    smesh = inout.load_mesh3d(dirname='data/storus')
    bbox = [-1,-1,-0.2,1,1,0.2]
    l = 0.06
    radius = np.abs(bbox[3] - bbox[0])*1.0/2  
    r = 0.2
    R = 0.8
    loop1 = pointgen3d.vertical_circle_xz(r, n=5, center=(R,0,0), theta = -np.pi/6)
    loop2 = pointgen3d.vertical_circle_xz(r, n=5, center=(R,0,0), theta=-np.pi/3)
    shapes = [loop1, loop2]
    input_points  = np.array(shapes)
    lambdas = [0.001]
    mus = [0.00001]
    is_closed = True
    return smesh, smesh.triangles, smesh.edges, input_points, lambdas, mus, is_closed

def tunnel_loops_on_torus_surface3d():
    '''
    Two tunnel loops on a surface of a torus which is not filled.
    '''
    mesh = inout.load_mesh3d(dirname='data/storus')
    bbox = [-1,-1,-0.2,1,1,0.2]
    l = 0.06
    radius = np.abs(bbox[3] - bbox[0])*1.0/2  
    r = 0.2
    R = 0.8
    #mesh = meshgen3d(bbox, l, include_corners=False, shape="torus", R=R, r=r)
    loop1 = pointgen3d.horizontal_circle(R + r*np.cos(np.pi/6), n=10, center=(0,0, r*np.sin(np.pi/6)))
    loop2 = pointgen3d.horizontal_circle(R + r*np.cos(np.pi/3), n=10, center=(0,0,-r*np.sin(np.pi/3)))
    loop1 = pointgen3d.horizontal_circle(R-r, n=10, center=(0,0,0))
    loop2 = pointgen3d.horizontal_circle(radius, n=20, center=(0,0,0))
    shapes = [loop1, loop2]
    input_points  = np.array(shapes)
    lambdas = [0.001]
    mus = [0.00001]
    is_closed = True
    return mesh, mesh.triangles, mesh.edges, input_points, lambdas, mus, is_closed
    
    
def equally_spaced_longitudes3ds(): 
    '''
    Three equally-spaced longitude curves on the surface of a sphere.
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
    '''
    curve1 = pointgen3d.sphere_arc(mesh.bbox, 0, 10)
    curve2 = pointgen3d.sphere_arc(mesh.bbox, 2*np.pi/3, 10)
    curve3 = pointgen3d.sphere_arc(mesh.bbox, 4*np.pi/3, 10)
    '''
    curve1 = pointgen3d.sphere_arc(smesh.bbox, 0, 10)
    curve2 = pointgen3d.sphere_arc(smesh.bbox, np.pi/4, 10)
    curve3 = pointgen3d.sphere_arc(smesh.bbox, np.pi/2, 10)

    shapes = [curve1, curve2, curve3]
    input_points  = np.array(shapes)
    lambdas = [0.001]
    mus = [0.00001]
    #lambdas = [0.0000001]
    #mus = [0.000000001]
    is_closed = False
    return smesh, smesh.triangles, smesh.edges, input_points, lambdas, mus, is_closed

def equally_spaced_longitudes3d(): 
    '''
    Three equally-spaced longitude curves on the surface of a ball.
    '''
    # l - initial length of triangle sides. Change it to vary traingle size
    #boundary_box = [-10,-10,-10,10,10,10]
    boundary_box = [-10,-10,-10,10,10,10]
    l = 2
    mesh = meshgen3d(boundary_box, l, include_corners=False, shape='ball')
    #inout.save_data(mesh, dirname=os.path.abspath("data/mesh_2"))
    #mesh  = inout.load_mesh3d(dirname='data/mesh_2')
    curve1 = pointgen3d.sphere_arc(mesh.bbox, 0, 10)
    curve2 = pointgen3d.sphere_arc(mesh.bbox, 2*np.pi/3, 10)
    curve3 = pointgen3d.sphere_arc(mesh.bbox, 4*np.pi/3, 10)
    shapes = [curve1, curve2, curve3]
    input_points  = np.array(shapes)
    lambdas = [0.001]
    mus = [0.00001]
    is_closed = False
    return mesh, mesh.triangles, mesh.edges, input_points, lambdas, mus, is_closed

def differently_spaced_longitudes3d(): 
    '''
    Three longitude curves on the surface of a ball.
    '''
    # l - initial length of triangle sides. Change it to vary traingle size
    boundary_box = [0,0,0,20,20,20]
    l = 4
    mesh = meshgen3d(boundary_box, l, include_corners=False, shape="ball")
    curve1 = pointgen3d.sphere_arc(mesh.bbox, 0, 10)
    curve2 = pointgen3d.sphere_arc(mesh.bbox, np.pi/4, 10)
    curve3 = pointgen3d.sphere_arc(mesh.bbox, 9*np.pi/8, 10)
    shapes = [curve1, curve2, curve3]
    input_points  = np.array(shapes)
    lambdas = [0.001]
    mus = [0.00001]
    is_closed = False
    return mesh, mesh.triangles, mesh.edges, input_points, lambdas, mus, is_closed

def get_saved_case(dirname='data/curves_on_sphere'): 
    '''
    Loads a saved experiment in the given directory.
    '''
    #Three longitude curves on a ball made of a finer mesh.
    mesh = inout.load_mesh3d(dirname=dirname)
    input_currents = inout.load_input_currents(len(mesh.edges), 3, dirname=dirname)
    t, q, r = inout.load_solutions(len(mesh.triangles), len(mesh.edges), 3, dirname=dirname)
    return mesh, input_currents, t, q, r
