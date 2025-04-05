# encoding: utf-8
'''
2D Median surface embedded in 3D
--------------------------------
'''
from __future__ import absolute_import
import importlib
import os

import numpy as np
from scipy.spatial import Delaunay

from medianshape.simplicial import pointgen3d, mesh
from medianshape.simplicial.meshgen import meshgen2d, meshgen3d, get_mesh_surface
from medianshape import inout

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm

from medianshape.viz import plot2d, plot3d
from distmesh.plotting import axes_simpplot3d
from meshpy.tet import MeshInfo, Options, build
from medianshape.simplicial.utils import boundary_points

def func(x, y, sign=1): 
    return np.sin(np.pi*x)*np.cos(np.pi*y)

def sample_surf(scale, step=0.2):
    '''
        Return a tuple X, Y, Z with a test surface.
    '''
    x = y = np.arange(-4.0, 4.0, step)
    X, Y = np.meshgrid(x, y)
    from matplotlib.mlab import  bivariate_normal
    '''
    Z1 = bivariate_normal(X, Y, 1.0, 1.0, 0.0, 0.0)
    Z2 = bivariate_normal(X, Y, 1.5, 0.5, 1, 1)
    #Z3 = bivariate_normal(X, Y, 1, 1, -2, -2)
    Z = Z2 - Z1
    '''
    # Ups
    ZU1 = bivariate_normal(X,Y, 1.5, 1, 0,-2)
    ZU2 = bivariate_normal(X, Y, 1.5, 1.5, 4, 1)
    ZU3 = bivariate_normal(X, Y, 1, 1, -4, 1)
    #ZU4 = bivariate_normal(X, Y, 1.5, 1.5, -4, -4)
    #ZU5 = bivariate_normal(X, Y, 1, 1, 4, -4)
    ZU4 = bivariate_normal(X, Y, 4, 0.5, 0, -4)
    # Downs
    ZD1 = bivariate_normal(X, Y, 1.5, 1, 0, 1)
    ZD2 = bivariate_normal(X, Y, 1.5, 1.5, -4, -2)
    ZD3 = bivariate_normal(X, Y, 1, 1, 4, -2)
    ZD4 = bivariate_normal(X, Y, 4, 1, 0, 4)
    Z = ZU1 + ZU2 + ZU3 - ZD1 - ZD2 - ZD3 - ZD4
    Zmax = np.amax(Z)
    X = X * scale[0]/4.0
    Y = Y * scale[1]/4.0
    Z = Z/Zmax * scale[2]
    return X, Y, Z

def interpolate_surf(points, values, ipoints, method = "cubic"):
    from scipy.interpolate import griddata
    return griddata(points, values, ipoints, method= method)

def surfaces(bbox=[-10,-10,-10, 10,10,10], l=0.5, overlaps =[0.4, 0.7]):
    '''
    '''
    # Generating point grids for two surfaces
    xmin = bbox[0]
    xmax  = bbox[3]
    ymin = bbox[1]
    ymax = bbox[4]
    zmin = bbox[2]
    zmax = bbox[5]
    xlen = xmax - xmin 
    y = np.arange(ymin, ymax, l)
    y = np.append(y, ymax)
    xmin_points = np.ndarray((len(y), 2))
    xmin_points[:,0] = xmin
    xmin_points[:, 1] = y
    xmax_points = np.ndarray((len(y), 2))
    xmax_points[:,0] = xmax
    xmax_points[:, 1] = y
    xoverlaps = [xmin + xlen*o for o in overlaps]
    xo_points = None
    for i, o in enumerate(xoverlaps):
        xo_tmp = np.ndarray((len(y), 2))
        xo_tmp[:,0] = o
        xo_tmp[:, 1] = y
        if i == 0:
            xo_points = xo_tmp
        else:
            xo_points = np.vstack((xo_points, xo_tmp)) 
    fixed_points = np.concatenate((xmin_points, xmax_points, xo_points), axis=0)
    print fixed_points 
    mesh = meshgen2d([xmin, ymin, xmax, ymax], l, fixed_points, include_corners=False)
    #plot2d.plotmesh2d(mesh)
    X, Y, Z1 = sample_surf([xmax*0.8, ymax*0.8, zmax*0.2])
    Z2 = -Z1 - zmax*0.3
    Z1 = Z1 + zmax*0.3
    #z2 = elevate_surf(mesh.points[:,0], mesh.points[:,1])
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    surf = ax.plot_surface(X, Y, Z1, rstride=1, cstride=1, cmap=cm.winter,
                           linewidth=0, antialiased=False)
    #surf = ax.plot_surface(X, Y, Z2, rstride=1, cstride=1, cmap=cm.autumn,
    #                       linewidth=0, antialiased=False)
    plt.show()
    sample_points = np.hstack((X.reshape(-1,1), Y.reshape(-1,1)))
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    surf = ax.scatter(X, Y, Z1.reshape(-1,1), color='b')
    surf = ax.scatter(X, Y, Z2.reshape(-1,1), color='r')
    plt.show()
    Z1 = interpolate_surf(sample_points, Z1.reshape(-1,1), mesh.points)
    Z2 = interpolate_surf(sample_points, Z2.reshape(-1,1), mesh.points)
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    surf = ax.scatter(mesh.points[:,0], mesh.points[:,1], Z1, color='b')
    surf = ax.scatter(mesh.points[:,0],mesh.points[:,1], Z2, color='r')
    plt.show()
    Z1[np.argwhere(mesh.points[:,1]==ymin)] = 0
    Z1[np.argwhere(mesh.points[:,1]==ymax)] = 0
    '''
    for xo in xoverlaps:
        Z1[np.argwhere(mesh.points[:,0]==xo)] = 0
    '''
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    surf = ax.scatter(mesh.points[:,0], mesh.points[:,1], Z1, color='b')
    surf = ax.scatter(mesh.points[:,0],mesh.points[:,1], Z2, color='r')
    plt.show()

    exit()
    #surf = ax.scatter(mesh.points[:,0], mesh.points[:,1], z2, color="r")
    ax.set_zlim(-1, 1)
    plt.show()
    X, Y = np.meshgrid(x, y, sparse=False)
    # Z coordinate of surface1
    Z1 = 7*np.sin(X)
    # Z coordinate of surface2
    Z2 = -7*np.sin(X)
    # Illustrating the surfaces

    # Triangulating the surfaces
    s1 = np.concatenate((X.reshape(-1,1), Y.reshape(-1,1), Z1.reshape(-1,1)), axis=1).reshape(-1,3)
    s2 = np.concatenate((X.reshape(-1,1), Y.reshape(-1,1), Z2.reshape(-1,1)), axis=1).reshape(-1,3)
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    surf = ax.scatter(s1[:,0], s1[:,1], s1[:,2])
    surf = ax.scatter(s2[:,0], s2[:,1], s2[:,2])
    ax.set_zlim(-10, 10)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    plt.show()
    leftline1 = np.where(s1[:,0]==xmin)[0]
    rightline1 = np.where(s1[:, 0] == xmax)[0]
    backline1 = np.where(s1[:,1]==xmin)[0]
    frontline1 = np.where(s1[:, 1] == xmax)[0]
    b1 = np.unique(np.concatenate((leftline1, rightline1, backline1, frontline1), axis=0))
    print b1

    leftline2 = np.where(s2[:,0]==xmin)[0]
    rightline2 = np.where(s2[:, 0] == xmax)[0]
    backline2 = np.where(s2[:,1]==xmin)[0]
    frontline2 = np.where(s2[:, 1] == xmax)[0]
    b2 = np.unique(np.concatenate((leftline2, rightline2, backline2, frontline2), axis=0))
    intersection = np.where(s1[:,0]== intersect)[0]
    closed_boundary = np.concatenate((leftline1, rightline2), axis=0)
    print b2
    print leftline1
    print rightline1
    print leftline2
    print leftline2

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    surf = ax.scatter(backline1[:,0], backline1[:,1], backline1[:,2])
    surf = ax.scatter(frontline1[:,0],frontline1[:,1],frontline1[:,2])
    surf = ax.scatter(leftline1[:,0], leftline1[:,1], leftline1[:,2])
    surf = ax.scatter(rightline1[:,0], rightline1[:,1], rightline1[:,2])
    surf = ax.scatter(backline2[:,0], backline2[:,1], backline2[:,2])
    surf = ax.scatter(frontline2[:,0],frontline2[:,1],frontline2[:,2])
    surf = ax.scatter(leftline2[:,0], leftline2[:,1], leftline2[:,2])
    surf = ax.scatter(rightline2[:,0], rightline2[:,1], rightline2[:,2])
    ax.set_zlim(-10, 10)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    plt.show()


    s1_complex = Delaunay(s1[:,:-1])
    s2_complex = Delaunay(s2[:,:-1])
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.set_zlim(-10, 10)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.plot_trisurf(s1[:,0], s1[:,1], s1[:,2], triangles = s1_complex.simplices, cmap=cm.autumn)
    ax.plot_trisurf(s2[:,0], s2[:,1], s2[:,2], triangles = s2_complex.simplices, cmap=cm.winter)
    plt.show()
    exit()

    s_points = np.vstack((s1, s2))
    s1_triangles = s1_complex.simplices
    s2_triangles = s2_complex.simplices + len(s1_complex.points)
    surfaces = np.vstack((s1_triangles, s2_triangles))

    # Plotting the surfaces
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.set_zlim(-10, 10)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.plot_trisurf(s_points[:,0], s_points[:,1], s_points[:,2], triangles = s1_triangles, cmap=cm.autumn)
    ax.plot_trisurf(s_points[:,0], s_points[:,1], s_points[:,2], triangles = s2_triangles, cmap=cm.winter)
    plt.show()

    '''
    # Plotting the surfaces
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.set_zlim(-10, 10)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax.plot_trisurf(points[:,0], points[:,1], points[:,2], triangles = surfaces, cmap=cm.autumn)
    plt.show()
    '''
    # Grid Points sampled from boundary box
    corners = boundary_points(bbox) 
    box = np.array(bbox).reshape(2, -1)
    points = np.mgrid[tuple(slice(min, max+1, 1) for min, max in box.T)]
    points = points.reshape(3, -1).T

    # Points of PLC.
    # Uncomment the line below to include sample poinds in bbox along with the surface points  
    #points  = np.concatenate((s_points, points), axis=0).reshape(-1,3)
    points  = np.concatenate((s_points, corners), axis=0).reshape(-1,3)
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    surf = ax.scatter(points[:,0], points[:,1], points[:,2])
    ax.set_zlim(-12, 12)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    plt.show()

    # Saving the current PLC to a file in .smesh format
    with open('surf.smesh', 'w') as f:
        f.write('%d %d %d %d\n'%(len(points), 3, 0,0))
        for i, p in enumerate(points):
            f.write("%d %f %f %f\n"%(i, p[0], p[1], p[2]))
        f.write('%d %d\n'%(len(surfaces), 0))
        for t in surfaces:
            f.write("%d %d %d %d\n"%(3, t[0], t[1], t[2]))
        f.write("%d\n"%0)
        f.write("%d\n"%0)
    np.savetxt("surface.face", surfaces)
    np.savetxt("points.node", points)

    # Build the mesh using Tetgen
    mesh_info = MeshInfo()
    mesh_info.set_points(points.tolist())
    mesh_info.set_facets((surfaces.reshape(-1, 3)).tolist())
    print len(mesh_info.facets)
    opts = Options("YVfeq", verbose=True, nobisect=True, facesout=True, edgesout=True, docheck=True) # Overriding 'pq' with no options or flags
    #opts = Options("", verbose=True, nobisect=True, facesout=True, edgesout=True, docheck=True, insertaddpoints=True) # Overriding 'pq' with no options or flags
    mesh = build(mesh_info, options=opts, volume_constraints=True, max_volume=1)

    # Plot the mesh
    ax = plt.gca(projection='3d')
    fig = ax.figure
    m1 = np.amin(bbox[0:3])
    m2 = np.amax(bbox[3:])
    ax.set_xlim([m1, m2])
    ax.set_ylim([m1, m2])
    ax.set_zlim([m1, m2])
    ax.set_aspect('equal')
    axes_simpplot3d(ax, np.array(list(mesh.points)), np.array(list(mesh.elements)))
    plt.show()
    # Write it as a file in vtk format, so you can use Paraview to see it.
    mesh.write_vtk("test.vtk")
    input_surfaces = np.zeros((2,len(mesh.faces)))
    inputs = list (s1_triangles, s2_triangles)
    for i, s in enumerate(inputs):
        for j, t in enumerate(np.array(mesh.faces)):
            if np.all(s==t):
                input_surfaces[i,j] = 1

    lambdas = [0.001]
    mus = [0.00001]
    mesh1 = Mesh3D()
    mesh1.simplices = np.array(mesh.elements)
    mesh1.triangles = np.array(mesh.faces)
    mesh1.edges = np.array(mesh.edges)
    mesh1.points = np.array(mesh.points)
    return mesh1, mesh1.simplices, mesh1.triangles, input_surfaces, lambdas, mus

    
if __name__ == "__main__":
    surfaces()
