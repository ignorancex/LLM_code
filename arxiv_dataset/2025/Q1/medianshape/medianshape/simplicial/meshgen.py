# encoding: utf-8
'''
Mesh generation
===============

'''

from __future__ import absolute_import

import importlib
import copy

import numpy as np
from scipy.spatial import Delaunay
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import medianshape.viz.plot3d as plt3d

import distmesh as dm

from medianshape.simplicial.mesh import Mesh2D, Mesh3D
import medianshape.utils as utils 

def get_mesh_surface(mesh):
    '''
    Gets a surface mesh out of a tetrahedralized simplicial complex K.

    '''
    smesh = copy.copy(mesh)
    b_matrix = np.abs(utils.boundary_matrix(mesh.triangles, mesh.edges, format='dok'))
    s_edges_idx = np.where(b_matrix.sum(axis=1)==3)[0]
    s_edges_idx = np.hstack((s_edges_idx, np.where(b_matrix.sum(axis=1)==4)[0]))

    interior_edges_idx = np.array([i for i in np.arange(mesh.edges.shape[0]) if i not in s_edges_idx])

    s_edges = mesh.edges[s_edges_idx.tolist()]
    surface_points= np.unique(s_edges.flatten())

    interior_edges = mesh.edges[interior_edges_idx.tolist()]
    tmp = list()
    for i, p_idx in enumerate(surface_points):
        k = len(np.argwhere(interior_edges[:,0]==p_idx))
        k += len(np.argwhere(interior_edges[:,1]==p_idx))
        s = len(np.argwhere(s_edges[:,0]==p_idx))
        s += len(np.argwhere(s_edges[:,1]==p_idx))
        if k > s:
            pass
            #print len(s_edges)
            #s_edges = np.array([e for e in s_edges if p_idx not in e]).reshape(-1,2)
            #print len(s_edges)
        else:
            tmp.append(p_idx)
    surface_points = tmp

    s_triangles = np.array([0,0,0]).reshape(1,3)
    t_idx = list()
    for i, t in enumerate(mesh.triangles):
        counter = 0
        for p in surface_points:
            if np.any(t==p):
                counter += 1
        if counter == 3:
            s_triangles = np.vstack((s_triangles, t.reshape(1,3)))
            t_idx.append(i)
    smesh.triangles = s_triangles[1:,:]
    smesh.edges = utils.get_subsimplices(smesh.triangles)
    smesh.surf_points = surface_points
    '''
    plt3d.plot_simplices3d(smesh, np.array(t_idx))
    plt.show()
    plt3d.plot_curve3d(smesh, np.ones(len(smesh.edges)))
    plt.show()
    '''
    return smesh

def meshgen2d(bbox, l, fixed_points=None, include_corners=True):
    '''
    Generates a simplical complex K of dimension 2, a triangulated mesh.

    :param float bbox: a bounding box, [:math:`x_{min}, y_{min}, x_{max}, y_{max}`].
    :param float l: initial edge length.
    :param float fixed_points: fixed points that should be in K.
    :param bool include_corners: If True, the corner points of the given boundary box must be in K.
    :returns: mesh -- an object of Mesh2D class.
    :rtype: object.
    '''
    mesh = Mesh2D()
    #l - initial length of triangle sides. Change it to vary traingle size
    mesh.bbox = bbox
    mesh.set_boundary_points()
    mesh.set_diagonal() 
    mesh.set_boundary_values()
    mesh.fixed_points = fixed_points
    if include_corners:
        if mesh.fixed_points is not None:
            mesh.fixed_points = np.vstack((mesh.fixed_points, mesh.boundary_points))
        else:
            mesh.fixed_points = mesh.boundary_points
    mesh.points, mesh.simplices = distmesh2d(bbox=mesh.bbox, l=l, fixed_points=mesh.fixed_points)
    mesh.edges = utils.get_subsimplices(mesh.simplices)
    mesh.orient_simplices_2D()
    return mesh

def meshgen3d(bbox, l, fixed_points=None, include_corners=True, load_data=False, shape="ball", **kwargs):
    '''
    Generates a simplical complex K of dimension 3, a tetrahedralized mesh.

    :param float bbox: a bounding box, [:math:`x_{min}, y_{min}, z_{min}, x_{max}, y_{max}, z_{max}`].
    :param float l: initial edge length.
    :param float fixed_points: fixed points that should be in K.
    :param bool include_corners: If True, the corner points of the given boundary box must be in K.
    :returns: mesh -- an object of Mesh3D class.
    :rtype: object.
    '''
    mesh = Mesh3D()
    mesh.bbox = bbox
    mesh.set_boundary_points()
    mesh.set_diagonal()
    mesh.set_boundary_values()
    mesh.fixed_points = fixed_points
    if include_corners:
        if mesh.fixed_points is not None:
            mesh.fixed_points = np.vstack((mesh.fixed_points, mesh.boundary_points))
        else:
            mesh.fixed_points = mesh.boundary_points
    mesh.points, mesh.simplices= distmesh3d(mesh.bbox, l, mesh.fixed_points, shape, **kwargs)
    #mesh.points, mesh.simplices = scipy_mesh3d(mesh.bbox, l, mesh.fixed_points)
    mesh.triangles = utils.get_subsimplices(mesh.simplices)
    mesh.edges = utils.get_subsimplices(mesh.triangles)
    return mesh

def scipy_mesh3d(bbox, l, fixed_points):
    '''
    Generates a simplical complex K of dimension 3, a tetrahedralized mesh.

    :param float bbox: a bounding box, [:math:`x_{min}, y_{min}, z_{min}, x_{max}, y_{max}, z_{max}`].
    :param float l: initial edge length.
    :param float fixed_points: fixed points that should be in K.
    :returns: points, tetrahedras.
    :rtype: float, int.
    '''
    bbox = np.array(bbox).reshape(2, -1)
    dim = bbox.shape[1]
    points = np.mgrid[tuple(slice(min, max+l, l) for min, max in bbox.T)]
    points = points.reshape(dim, -1).T
    ax = plt.gca(projection='3d')
    ax.scatter(points[:,0],points[:,1], points[:,2])
    return points, Delaunay(points).simplices

#def meshpy_cuboid3d(bbox, fixed_points, max_volume):
#    mesh_info = MeshInfo()
#    mesh_info.set_points(fixed_points)
#    mesh_info.set_facets([[0,1,2,3],\
#                            [4,5,6,7],\
#                            [0,4,5,1],\
#                            [1,5,6,2],\
#                            [2,6,7,3],\
#                            [3,7,4,0],\
#                                ])
#    mesh = build(mesh_info, volume_constraints=True, max_volume=max_volume)
#    return np.array(mesh.points), np.array(mesh.elements)

def distmesh2d(bbox, l, fixed_points=None, shape="square"):
    '''
    Generates a simplical complex K of dimension 2, a triangulated mesh.

    :param float bbox: a bounding box, [:math:`x_{min}, y_{min}, x_{max}, y_{max}`].
    :param float l: initial edge length.
    :param float fixed_points: fixed points that should be in K.
    :param str shape: the shape of K.
    :returns: points, triangles.
    '''
    if shape == "square":
        return square_mesh(bbox, l, fixed_points)

def distmesh3d(bbox, l=0.1, fixed_points=None, shape="ball", **kwargs):
    '''
    Generates tetrahedral mesh, K in 3D with a given shape using Distmesh.

    :param float bbox: a bounding box, [:math:`x_{min}, y_{min}, z_{min}, x_{max}, y_{max}, z_{max}`].
    :param float l: initial edge length.
    :param float fixed_points: fixed points that should be in K.
    :param str shape: a shape of K
    :returns: points, tetrahedras.
    '''
    #distmesh.distmeshnd accepts None(not []) if there is no fixed points.
    if  fixed_points is not None:
        if len(fixed_points) == 0:
            fixed_points = None
    #if shape == "cuboid":
        #return cuboid_mesh(bbox, fixed_points, l)
    if shape == "ball":
        r = np.abs(bbox[3] - bbox[0])*1.0/2  
        center = [(bbox[0]+bbox[3])*1.0/2, (bbox[1] + bbox[4])*1.0/2, (bbox[2]+bbox[5])*1.0/2]
        dist_function = lambda p: dm.dsphere(p, center[0], center[1], center[2], r)
        points, simplices = dm.distmeshnd(dist_function, dm.huniform, l, bbox, fixed_points, fig=None) 
    if shape == "sphere":
        r = np.abs(bbox[3] - bbox[0])*1.0/2  
        center = [(bbox[0]+bbox[3])*1.0/2, (bbox[1] + bbox[4])*1.0/2, (bbox[2]+bbox[5])*1.0/2]
        dist_function = lambda p: dm.dsphere(p, center[0], center[1], center[2],r);
        points, simplices = dm.distmeshnd(dist_function, dm.huniform, l, bbox, fixed_points, fig=None) 
    if shape == "torus":
        radius = np.abs(bbox[3] - bbox[0])*1.0/2  
        r = radius/4
        R = radius - r
        if "r" in kwargs:
            r = kwargs["r"]
        if "R" in kwargs:
            R = kwargs["R"]
        dist_function = lambda p: ((p**2).sum(axis=1)+R**2-r**2)**2-4*R**2*(p[:,0]**2+p[:,1]**2);
        points, simplices=dm.distmeshnd(dist_function, dm.huniform, l, bbox, fig=None);
    return points, simplices

def square_mesh(bbox, l, fixed_points=None):
    '''
    Generates a simplical complex K of dimension 2, a triangulated mesh,
    with square shape using Distmesh.

    :param float bbox: a bounding box, [:math:`x_{min}, y_{min}, x_{max}, y_{max}`].
    :param float l: initial edge length. 
    :param float fixed_points: fixed points that should be in K.
    :returns: points, triangles.
    '''
    #Square, with size function point and line sources#
    dist_function = lambda p: dm.drectangle(p, bbox[0], bbox[2], \
                                                bbox[1],bbox[3])
    points, triangles = dm.distmesh2d(dist_function, dm.huniform, l, bbox, fixed_points)
    return points, triangles 

def cuboid_mesh(bbox, l, fixed_points=None):
    '''
    Generates tetrahedral mesh in 3D using DistMesh with cubic shape using Distmesh.

    :param float bbox: a bounding box, [:math:`x_{min}, y_{min}, z_{min}, x_{max}, y_{max}, z_{max}`].
    :param float l: initial edge length.
    :param float fixed_points: fixed points that should be in K.
    :returns: points, tetrahedras.
    '''
    dist_function = lambda p: dcuboid(p, float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), float(bbox[4]), float(bbox[5])) 
    points, tetrahedras = dm.distmeshnd(dist_function, dm.huniform, l, bbox, fixed_points) 
    return points, tetrahedras

def dcuboid(p, x1, y1, z1, x2, y2, z2):
    """Signed distance function for cubiod with corners (x1,y1,z1), (x1,y1,z2),
    (x1,y2,z1), (x1,y2,z2), (x2,y1,z1), (x2,y2,z1), (x2,y1,z2), (x2,y2,z2).

    This has an incorrect distance to the four corners. 
    """
    d1 = x1 - p[:,0]
    d2 = -x2 + p[:,0]
    d3 = y1 - p[:,1]
    d4 = -y2  + p[:,1]
    d5 = z1 - p[:,2]
    d6 = -z2 + p[:,2]

    d135 = np.sqrt(d1**2+d3**2+d5**2)
    d136 = np.sqrt(d1**2+d3**2+d6**2)
    d145 = np.sqrt(d1**2+d4**2+d5**2)
    d146 = np.sqrt(d1**2+d4**2+d6**2)
    d235 = np.sqrt(d2**2+d3**2+d5**2)
    d236 = np.sqrt(d2**2+d3**2+d6**2)
    d245 = np.sqrt(d2**2+d4**2+d5**2)
    d246 = np.sqrt(d2**2+d4**2+d6**2)
    
    d = -np.minimum(np.minimum(np.minimum(np.minimum(np.minimum(-d1,-d2),-d3),-d4),-d5),-d6)
    ix = (d1>0)*(d3>0)*(d5>0)
    d[ix] = d135[ix]
    ix = (d1>0)*(d3>0)*(d6>0)
    d[ix] = d136[ix]
    ix = (d1>0)*(d4>0)*(d5>0)
    d[ix] = d145[ix]
    ix = (d1>0)*(d4>0)*(d6>0)
    d[ix] = d146[ix]
    ix = (d2>0)*(d3>0)*(d5>0)
    d[ix] = d235[ix]
    ix = (d2>0)*(d3>0)*(d6>0)
    d[ix] = d236[ix]
    ix = (d2>0)*(d4>0)*(d5>0)
    d[ix] = d245[ix]
    ix = (d2>0)*(d4>0)*(d4>0)
    d[ix] = d246[ix]
    return 
